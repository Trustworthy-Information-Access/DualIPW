'''
Qingyao Ai, Keping Bi, Cheng Luo, Jiafeng Guo, and W Bruce Croft. 2018. Unbi-
ased learning to rank with unbiased propensity estimation. In SIGIR’18. 385–394.
'''
import torch.nn as nn
import torch
import numpy as np
import time

from args import config
import torch.nn.functional as F
import sys

from baseline_model.learning_algorithm.base_algorithm import BaseAlgorithm
import baseline_model.utils as utils
from baseline_model.utils.sys_tools import find_class


def sigmoid_prob(logits):
    return torch.sigmoid(logits - torch.mean(logits, -1, keepdim=True))

class IOBM(nn.Module):

    def __init__(self,list_size,click_dim):

        super(IOBM,self).__init__()
        self.nlayers=config.s_nlayers
        self.list_size=list_size
        self.click_dim=click_dim
        self.context_dim=config.rank_feature_size
        self.intra_dim=self.click_dim*2+self.context_dim
        
        self.hidden_size=config.s_nhidden
        self.inter_dim=self.hidden_size*2+self.context_dim
        
        self.logits_to_prob=nn.Softmax(dim=-1)
        
        self.bi=config.bi
        if self.bi:
            self.bi=True
        else:
            self.bi=False
        
        self.pos_emb=nn.Embedding(self.list_size,self.click_dim)
        self.click_emb=nn.Embedding(2,self.click_dim)
        
        
        
        self.intra_w=nn.Sequential()
        self.intra_w.add_module('layer',nn.Linear(self.intra_dim,self.intra_dim-self.context_dim))
        self.intra_w.add_module('act',nn.Tanh())
        
        self.model=nn.LSTM(input_size=self.intra_dim-self.context_dim,hidden_size=self.hidden_size,batch_first=True,
            num_layers=self.nlayers,bidirectional=self.bi)
        
        self.inter_w=nn.Sequential()
        self.inter_w.add_module('layer',nn.Linear(self.inter_dim,self.inter_dim-self.context_dim))
        self.inter_w.add_module('act',nn.Tanh())
        

        
        self.output=nn.Linear(self.inter_dim-self.context_dim,1)

    
    def additive_attention(self,context,seq,att_type=''):

        x=torch.cat([context,seq],dim=-1)
        # W: m*n -> [bs seqL [x;p;c]]->[bs seqL [p;c]] n<-> lx
        n=seq.shape[-1]
        if att_type=='intra':
            x=self.intra_w(x)
        elif att_type=='inter':
            x=self.inter_w(x)
        x=self.logits_to_prob(x)*n
        return x,x*seq # 各分量weight*分量坐标
        
    def forward(self,pos_input,click_input,context_input=None):

        bs=pos_input.shape[0]
        # [bs seqL dim]->avg: [bs dim] -> copy: [bs seqL dim]
        context=context_input.mean(dim=1,keepdim=True).repeat(1,self.list_size,1)
        pos_emb=self.pos_emb(pos_input.long())
        click_emb=self.click_emb(click_input.long())
        
        # intra: [bs seqL dim+dim] -> [bs seqL hidden]
        x=torch.cat([pos_emb,click_emb],dim=-1)
        _,x=self.additive_attention(context,x,att_type='intra')
        start=torch.zeros(bs,1,x.shape[-1]).cuda()
        end=torch.zeros_like(start).cuda()
        x=torch.cat([start,x,end],dim=1)
        
        # [bs seqL dim*2]
        output,_=self.model(x)
        output=output.reshape(bs,self.list_size+2,2,-1)
        forward=output[:,:-2,0,:]
        backward=output[:,2:,1,:]
        
        x=torch.cat([forward,backward],dim=-1)

        # [bs seqL dim*2]
        _,x=self.additive_attention(context,x,att_type='inter')
        
        return self.output(x).squeeze(dim=-1) # [bs seqL]


class DenoisingNet(nn.Module):
    def __init__(self, input_vec_size,click_dim=0):
        super(DenoisingNet, self).__init__()
        self.linear_layer = nn.Linear(input_vec_size, 1)
        self.elu_layer = nn.ELU()
        self.propensity_net = nn.Sequential(self.linear_layer, self.elu_layer)
        self.list_size = input_vec_size

    def forward(self, input_list):
        output_propensity_list = []
        for i in range(self.list_size):
            # Add position information (one-hot vector)
            click_feature = [
                torch.unsqueeze(
                    torch.zeros_like(
                        input_list[i]), -1) for _ in range(self.list_size)]
            click_feature[i] = torch.unsqueeze(
                torch.ones_like(input_list[i]), -1)
            # Predict propensity with a simple network
            output_propensity_list.append(
                self.propensity_net(
                    torch.cat(
                        click_feature, 1)))

        return torch.cat(output_propensity_list, 1)

class DLA_DNN(BaseAlgorithm):
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
        print('Build DLA')

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
        if 'selection_bias_cutoff' in self.exp_settings.keys():
            self.rank_list_size = self.exp_settings['selection_bias_cutoff']

        self.rank_feature_size = exp_settings['rank_feature_size']
        self.p_model = config.p_model
        self.click_dim = config.click_dim
        self.propensity_model = find_class(f'baseline_model.learning_algorithm.dla_dnn.{self.p_model}')(
            self.max_candidate_num, self.click_dim)

        # single gpu trained directly
        self.model = ranking_model.cuda()

        self.propensity_model = self.propensity_model.cuda()
        self.labels_name = []  # the labels for the documents (e.g., clicks)
        self.labels = []  # the labels for the documents (e.g., clicks)
        for i in range(self.rank_list_size):
            self.labels_name.append("label{0}".format(i))

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

        self.opt_denoise = self.optimizer_func(
            self.propensity_model.parameters(), self.propensity_learning_rate)

        self.opt_ranker = self.optimizer_func(
            [{'params':self.model.parameters(),'lr':self.learning_rate}])

    def separate_gradient_update(self):
        # Select optimizer

        if self.hparams.l2_loss > 0:
            for p in ranking_model_params:
                self.rank_loss += self.hparams.l2_loss * self.l2_loss(p)
        self.loss = self.exam_loss + self.hparams.ranker_loss_weight * self.rank_loss

        self.opt_denoise.zero_grad()
        self.opt_ranker.zero_grad()

        self.loss.backward()

        if self.hparams.max_gradient_norm > 0:
            nn.utils.clip_grad_norm_(
                self.propensity_model.parameters(), self.hparams.max_gradient_norm)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.hparams.max_gradient_norm)

        self.opt_denoise.step()
        self.opt_ranker.step()

    # 原dla
    def train(self, input_feed, epoch=0):
        """Run a step of the model feeding the given inputs.

        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.

        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a tf.summary containing related information about the step.

        """

        # Build model
        self.model.train()
        self.propensity_model.train()
        self.create_input_feed(input_feed, self.rank_list_size)
        self.rank_loss = 0

        # start train
        features = input_feed['feat_input']
        feat_emb, train_output = self.model(features, 1)
        train_output = train_output.reshape(-1, self.rank_list_size)

        train_labels = self.labels
        bs = train_labels.shape[0]
        # [bs 10]
        
        if self.p_model=='IOBM':
            pos_input = torch.arange(self.rank_list_size).repeat(
                bs).reshape([bs, -1]).cuda()
            click_input = train_labels.detach()
            context=feat_emb.reshape(-1,self.rank_list_size,self.rank_feature_size).detach()

            self.propensity = self.propensity_model(
                pos_input, click_input,context)
        else:
            propensity_labels = torch.transpose(train_labels,0,1)
            self.propensity = self.propensity_model(
                propensity_labels)

        with torch.no_grad():

            self.propensity_weights = self.get_normalized_weights(
                self.logits_to_prob(self.propensity))
            self.relevance_weights = self.get_normalized_weights(
                self.logits_to_prob(train_output))
        
        self.rank_loss = self.rank_loss + self.loss_func(
            train_output, train_labels, propensity_weights=self.propensity_weights)
        
        self.exam_loss = self.loss_func(
            self.propensity,
            train_labels,
            propensity_weights=self.relevance_weights,

        )
        
        self.loss = self.exam_loss + self.hparams.ranker_loss_weight * self.rank_loss
        self.separate_gradient_update()

        self.clip_grad_value(train_labels, clip_value_min=0, clip_value_max=1)
        self.global_step += 1
        return self.loss.item()


    def get_scores(self, input_feed):
        self.model.eval()
        features = input_feed['feat_input']
        scores = self.model(features)
        return scores

    def state_dict(self):
        return {'model': self.model.state_dict(
        ), 'propensity_model': self.propensity_model.state_dict()}

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])
        self.propensity_model.load_state_dict(state_dict['propensity_model'])
        
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