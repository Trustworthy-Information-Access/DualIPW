import torch.nn as nn
import torch
import numpy as np
from baseline_model import ranking_model


from baseline_model.learning_algorithm.base_algorithm import BaseAlgorithm
import baseline_model.utils as utils
from args import config


def sigmoid_prob(logits):
    return torch.sigmoid(logits - torch.mean(logits, -1, keepdim=True))

class PropensityModel(nn.Module):
    """ Propensity model adapts from SetRank, which take both position and document into account
    """
    def __init__(self, feature_size, rank_list_size):
        super(PropensityModel, self).__init__()

        self.hparams = utils.hparams.HParams(
            n_layers=2,
            n_heads=8,
            hidden_size=config.pos_dim,
            inner_size=int(feature_size/2),
            hidden_dropout_prob=0.2,
            attn_dropout_prob=0.2,
            hidden_act='leaky_relu',
            layer_norm_eps=1e-12
        )

        self.feature_size = feature_size
        self.rank_list_size = rank_list_size

        self.position_embedding = nn.Embedding(rank_list_size, self.hparams.hidden_size)

        self.document_feature_layer = nn.Sequential(
            nn.Linear(feature_size, self.hparams.inner_size),
            nn.LeakyReLU(),
            nn.Linear(self.hparams.inner_size, self.hparams.hidden_size)
        )

        self.confounder_encoder = nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size)

        self.output_layer = nn.Sequential(
            nn.Linear(self.hparams.hidden_size, self.hparams.inner_size),
            nn.LeakyReLU(),
            nn.Linear(self.hparams.inner_size, 1),
        )

    def forward(self, document_tensor, add_position):
        """
        :param document_tensor: [batch_size, seq_len, feature_size]
        :param add_position: bool, True then add position_emb when generate output, False not.
        :return:
        """
        # [batch_size, rank_list_size, hidden_size]
        document_emb = self.document_feature_layer(document_tensor)
        output = self.confounder_encoder(document_emb)


        if add_position:
            position_ids = torch.arange(document_tensor.size(1), dtype=torch.long).cuda()
            position_ids = position_ids.unsqueeze(0).expand(document_tensor.size(0), -1)
            position_emb = self.position_embedding(position_ids)
            feed_forward_input = output + position_emb
        else:
            feed_forward_input = output
        output = self.output_layer(feed_forward_input)

        # [batch_size, rank_list_size, 1]
        return output.squeeze(dim=-1)


class UPE_DNN(BaseAlgorithm):
    """The Dual Learning Algorithm for unbiased learning to rank.

    This class implements the Dual Learning Algorithm (DLA) based on the input layer
    feed. See the following paper for more information on the algorithm.

    * Qingyao Ai, Keping Bi, Cheng Luo, Jiafeng Guo, W. Bruce Croft. 2018. Unbiased Learning to Rank with Unbiased Propensity Estimation. In Proceedings of SIGIR '18

    """

    def __init__(self, exp_settings, ranking_model):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
        """
        print('Build UPE_dnn')

        self.hparams = utils.hparams.HParams(
            learning_rate=exp_settings['lr'],                 # Learning rate.
            max_gradient_norm=0.5,            # Clip gradients to this norm.
            loss_func='softmax_loss',            # Select Loss function
            # the function used to convert logits to probability distributions
            logits_to_prob='softmax',
            # The learning rate for ranker (-1 means same with learning_rate).
            propensity_learning_rate=-1.0,
            ranker_loss_weight=1.0,            # Set the weight of unbiased ranking loss
            # Set strength for L2 regularization.
            l2_loss=0.0,
            max_propensity_weight=-1,      # Set maximum value for propensity weights
            constant_propensity_initialization=False,
            # Set true to initialize propensity with constants.
            grad_strategy='adamw',            # Select gradient strategy
        )

        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings
        self.max_candidate_num = exp_settings['max_candidate_num']

        self.feature_size = exp_settings['rank_feature_size']

        self.combine = exp_settings['combine']

        if 'selection_bias_cutoff' in exp_settings.keys():
            self.rank_list_size = self.exp_settings['selection_bias_cutoff']
            self.propensity_model = PropensityModel(self.feature_size, self.rank_list_size)

        self.model = ranking_model.cuda()
        self.propensity_model = self.propensity_model.cuda()
        self.loss_mode = config.loss_mode

        self.labels_name = []  # the labels for the documents (e.g., clicks)
        self.labels = []  # the labels for the documents (e.g., clicks)
        self.docid_inputs=[]
        self.docid_inputs_name=[]
        for i in range(self.max_candidate_num):
            self.labels_name.append("label{0}".format(i))
            self.docid_inputs_name.append("docid_input{0}".format(i))

        if self.hparams.propensity_learning_rate < 0:
            self.propensity_learning_rate = float(self.hparams.learning_rate)
        else:
            self.propensity_learning_rate = float(
                self.hparams.propensity_learning_rate)
        self.learning_rate = float(self.hparams.learning_rate)

        self.global_step = 0

        # Select logits to prob function
        self.logits_to_prob = nn.Softmax(dim=-1)
        if self.hparams.logits_to_prob == 'sigmoid':
            self.logits_to_prob = sigmoid_prob

        self.optimizer_func = torch.optim.AdamW
        if self.hparams.grad_strategy == 'sgd':
            self.optimizer_func = torch.optim.SGD

        print('Loss Function is ' + self.hparams.loss_func)
        # Select loss function
        self.loss_func = None
        if self.hparams.loss_func == 'sigmoid_loss':
            self.loss_func = self.sigmoid_loss_on_list
        elif self.hparams.loss_func == 'pairwise_loss':
            self.loss_func = self.pairwise_loss_on_list
        else:  # softmax loss without weighting
            self.loss_func = self.softmax_loss
        
        pretrain_params = list(self.propensity_model.document_feature_layer.parameters()) + \
                          list(self.propensity_model.confounder_encoder.parameters()) + \
                          list(self.propensity_model.output_layer.parameters())
        denoise_params = list(self.propensity_model.position_embedding.parameters())
        ranking_model_params = list(self.model.parameters())

        self.opt_pretrain = self.optimizer_func(pretrain_params, self.learning_rate)#, weight_decay=1e-4)
        self.opt_denoise = self.optimizer_func(denoise_params, self.propensity_learning_rate)
        self.opt_ranker = self.optimizer_func(ranking_model_params, self.learning_rate)


    def separate_gradient_update(self):
        denoise_params = self.propensity_model.parameters()
        ranking_model_params = self.model.parameters()
        # Select optimizer

        if self.hparams.l2_loss > 0:
            for p in ranking_model_params:
                self.rank_loss += self.hparams.l2_loss * self.l2_loss(p)
        self.loss = self.exam_loss + self.hparams.ranker_loss_weight * self.rank_loss + \
                    self.hparams.ranker_loss_weight*self.pt_loss

        self.opt_pretrain.zero_grad()
        self.opt_denoise.zero_grad()
        self.opt_ranker.zero_grad()

        self.loss.backward()

        if self.hparams.max_gradient_norm > 0:
            nn.utils.clip_grad_norm_(self.propensity_model.parameters(), self.hparams.max_gradient_norm)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.max_gradient_norm)

        self.opt_pretrain.step()
        self.opt_denoise.step()
        self.opt_ranker.step()

    def state_dict(self):
        return {'model': self.model.state_dict(), 'propensity_model': self.propensity_model.state_dict()}

    def load_state_dict(self, state_dict):

        self.model.load_state_dict(state_dict['model'])
        self.propensity_model.load_state_dict(state_dict['propensity_model'])

    def train(self, input_feed,epoch=0):
        """Run a step of the model feeding the given inputs.

        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.

        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a tf.summary containing related information about the step.

        """

        # Build model

        self.model.train()
        self.create_input_feed(input_feed, self.rank_list_size)
        features = input_feed['feat_input']

        input, train_output = self.model(features, 1)
        input=input.reshape(-1,self.rank_list_size,self.feature_size)
        self.propensity_model.train()
        
        # [bs seq_len]
        pt_scores=self.propensity_model(input,add_position=False)

        # initial_scores predicted by lgb
        initial_scores=torch.Tensor(list(input_feed['initial_scores'])).reshape(-1,self.rank_list_size).cuda()
        initial_scores=self.logits_to_prob(initial_scores)

        self.pt_loss=self.loss_func(pt_scores,initial_scores)
        
        train_output = train_output.reshape(-1, self.max_candidate_num)
        
        train_labels = self.labels

        
        # p(E|do(K))
        self.raw_propensity=self.propensity_model(input,add_position=True)
        # Compute examination loss
        with torch.no_grad():
            self.relevance_weights = self.get_normalized_weights(
                self.logits_to_prob(train_output[:, :self.rank_list_size]))
        self.exam_loss = self.loss_func(
            self.raw_propensity,
            train_labels[:, :self.rank_list_size],
            propensity_weights=self.relevance_weights
        )
        

        do_propensity = []
        # [seq_len bs]
        do_propensity.append(self.raw_propensity.transpose(0,1))

        for _ in range(16):
            random_indices = np.random.choice(self.rank_list_size, self.rank_list_size, replace=False)
            # [rank_list_size, batch_size]
            current_docid_inputs = self.docid_inputs[random_indices,:].transpose(0,1)
            # [bs seq_len feat_dim]
            cur_features,_=self.model(current_docid_inputs,1)
            current_do_propensity = self.propensity_model(cur_features,add_position=True)
            # current_do_propensity = torch.cat(current_do_propensity, 1)
            # [seq_len bs]
            do_propensity.append(current_do_propensity.transpose(0,1))
        # [sample_num, seq_len, bs]
        do_propensity = torch.stack(do_propensity, 0).transpose(-1,-2)
        with torch.no_grad():
            # [sample_num, batch_size, rank_list_size]
            do_propensity = self.logits_to_prob(do_propensity)
            # [batch_size, rank_list_size]
            self.propensity_weights = torch.mean(do_propensity, 0)
            # [1, rank_list_size] --> [batch_size, rank_list_size]
            self.propensity_weights = torch.mean(self.propensity_weights, 0, keepdim=True)
            self.propensity_weights = self.get_normalized_weights(self.propensity_weights).expand_as(train_output)
        
        self.rank_loss = self.loss_func(
            train_output, train_labels, propensity_weights=self.propensity_weights)

        self.loss = self.exam_loss + self.hparams.ranker_loss_weight * self.rank_loss +self.pt_loss
        self.separate_gradient_update()

        self.clip_grad_value(train_labels, clip_value_min=0, clip_value_max=1)
        self.global_step += 1

        return self.loss.item(), self.propensity_weights[0][:self.rank_list_size]

    def get_scores(self, input_feed):
        self.model.eval()
        features = input_feed['feat_input']
        scores = self.model(features)
        return scores

    def get_normalized_weights(self, propensity):
        """Computes listwise softmax loss with propensity weighting.

        Args:
            propensity: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (tf.Tensor) A tensor containing the propensity weights.
        """
        propensity_list = torch.unbind(
            propensity, dim=1)  # Compute propensity weights
        pw_list = []
        for i in range(len(propensity_list)):
            pw_i = propensity_list[0] / propensity_list[i]
            pw_list.append(pw_i)
        propensity_weights = torch.stack(pw_list, dim=1)
        if self.hparams.max_propensity_weight > 0:
            self.clip_grad_value(propensity_weights, clip_value_min=0,
                                 clip_value_max=self.hparams.max_propensity_weight)
        return propensity_weights
