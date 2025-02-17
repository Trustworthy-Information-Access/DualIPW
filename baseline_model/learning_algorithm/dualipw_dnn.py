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

single_dis=[[51.732158028887014, 16.448598130841123, 15.337723024638914, 3.7723024638912492, 2.916312659303313, 2.068819031435854, 2.341758708581138, 3.964528462192014, 0.8315632965165675, 0.5862361937128293],
            [30.345310312645825, 33.91273915072328, 13.838077461502564, 8.021465235650956, 2.8511432571161923, 2.305179654689687, 2.1278581427904806, 4.451703219785347, 1.3415772281847875, 0.8049463369108725],
            [32.085600290170476, 18.302502720348205, 27.319550235763508, 6.042800145085238, 3.4167573449401525, 3.1120783460282917, 2.4519405150525935, 4.6572361262241575, 1.682988755894088, 0.9285455204932898],
            [22.847826086956523, 17.19565217391304, 10.08695652173913, 38.097826086956516, 2.6195652173913047, 2.0543478260869565, 1.6847826086956523, 3.3695652173913047, 1.1630434782608696, 0.8804347826086957],
            [27.005291005291006, 18.708994708994712, 13.26984126984127, 18.941798941798943, 11.597883597883598, 2.2645502645502646, 1.8835978835978833, 3.8095238095238093, 1.4179894179894181, 1.1005291005291005],
            [26.503401360544217, 16.517006802721088, 16.13605442176871, 16.163265306122447, 4.054421768707483, 10.231292517006802, 1.9591836734693877, 5.605442176870748, 1.523809523809524, 1.306122448979592],
            [27.02078521939954, 17.41339491916859, 14.919168591224018, 11.224018475750578, 5.080831408775981, 2.9099307159353347, 11.824480369515012, 5.819861431870669, 1.8475750577367207, 1.9399538106235563],
            [23.829787234042556, 16.117021276595743, 12.553191489361703, 8.351063829787233, 3.351063829787234, 2.340425531914893, 1.9680851063829787, 20.744680851063833, 9.414893617021276, 1.3297872340425534],
            [28.75, 19.72222222222222, 16.180555555555554, 9.166666666666666, 4.027777777777777, 4.305555555555555, 2.0138888888888893, 4.513888888888889, 7.708333333333334, 3.6111111111111107],
            [25.13157894736842, 21.44736842105263, 14.605263157894736, 9.868421052631579, 6.315789473684211, 5.789473684210526, 2.7631578947368416, 4.342105263157895, 1.7105263157894737, 8.026315789473683]]
for i,dis in enumerate(single_dis):
    if config.f_mode=='KL':
        single_dis[i]=[d/100 for d in dis]
    else:
        single_dis[i]=[d/max(dis) for d in dis]
single_dis=torch.Tensor(single_dis).cuda()
# single_dis=nn.Softmax(dim=-1)(single_dis/0.1)
# print(single_dis)

class FFN(nn.Module):

    def __init__(self, layers=[], layer_size=64, norm='layer',act='elu'):
        super(FFN, self).__init__()
        if act=='elu':
            self.activation = nn.ELU()
        elif act=='relu':
            self.activation=nn.ReLU()
        self.layers = layers
        self.sequential = nn.Sequential().to(dtype=torch.float32)

        # easy to add multi-dense_layer
        for i in range(len(self.layers)):
            if norm=='layer':
                self.sequential.add_module('layer_norm{}'.format(
                    i), nn.LayerNorm(layer_size).to(dtype=torch.float32))
            elif norm=='batch2d':
                self.sequential.add_module('batch_norm{}'.format(
                    i), nn.BatchNorm2d(layer_size).to(dtype=torch.float32))
            elif norm=='batch1d':
                self.sequential.add_module('batch_norm{}'.format(
                    i), nn.BatchNorm1d(layer_size).to(dtype=torch.float32))
            self.sequential.add_module('linear{}'.format(
                i), nn.Linear(layer_size, self.layers[i]))
            if i != len(self.layers)-1:
                self.sequential.add_module('act{}'.format(i), self.activation)
            layer_size = self.layers[i]

    def forward(self, x):

        return self.sequential(x).squeeze(dim=-1)

# listwise(单序列 -> soft) + 多点击各个位置的直接相加
# LSTM作为seq_model
class QLDenoise(nn.Module):
    
    def __init__(self,seq_len=10):
        super(QLDenoise,self).__init__()
        self.tau1=nn.Parameter(torch.FloatTensor([1]))
        self.hidden_size=config.s1_hidden
        
        self.seq_single=nn.LSTM(input_size=1,hidden_size=self.hidden_size,
                                num_layers=config.s1_nlayers,batch_first=True,bidirectional=False)
        
        self.output=FFN([int(self.hidden_size/2),1],layer_size=self.hidden_size,act='relu')

        self.act=nn.Sigmoid()
        self.soft=nn.Softmax(dim=-1)
    
    
    # 单点击序列weight -> 多点击序列weight平均
    def forward(self,ori_clicks,dis):
        # clicks: [bs seq_len]; dis: [seq_len seq_len]
        click_one=torch.zeros(1,ori_clicks.shape[1]).cuda()
        click_one[:,0]=1
        # [bs+1 seq_len]
        ori_clicks=torch.cat([click_one,ori_clicks],dim=0) # row 0 -> used for normalization
        clicks=ori_clicks/ori_clicks.sum(dim=-1,keepdim=True)
        
        bs=clicks.shape[0]
        seq_len=clicks.shape[1]
        single_click=torch.eye(seq_len).cuda()
        # # tau1
        single_click=self.soft(single_click/self.tau1)
        single_kl=-single_click*torch.log(dis)+single_click*torch.log(single_click)
        
        print(single_kl)
        single_kl=single_kl.unsqueeze(dim=-1)
        seq_single,_=self.seq_single(single_kl)
        seq_single=seq_single[:,-1,:]

        # # [bs dim]
        output=self.output(seq_single)

        # print(output)
        output=self.soft(output).unsqueeze(dim=-1)
        
        print(output)
        weight=torch.matmul(clicks,output).squeeze()
        weight=weight/weight[0].detach()

        # [1:]
        return weight[1:],0
    
class DenoisingNet(nn.Module):
    def __init__(self, input_vec_size):
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

class DualIPW_DNN(BaseAlgorithm):
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
        print('Build DualIPW')

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
        self.propensity_model = DenoisingNet(self.max_candidate_num)

        # single gpu trained directly
        self.model = ranking_model.cuda()

        self.propensity_model = self.propensity_model.cuda()
        self.w_model=QLDenoise().cuda()

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
        self.opt_weight=self.optimizer_func(self.w_model.parameters(),self.learning_rate)#*10)

    def separate_gradient_update(self):
        denoise_params = self.propensity_model.parameters()
        ranking_model_params = self.model.parameters()
        # Select optimizer

        if self.hparams.l2_loss > 0:
            for p in ranking_model_params:
                self.rank_loss += self.hparams.l2_loss * self.l2_loss(p)
        
        self.loss = self.exam_loss + self.hparams.ranker_loss_weight * self.rank_loss + self.w_loss

        self.opt_denoise.zero_grad()
        self.opt_ranker.zero_grad()

        self.opt_weight.zero_grad()

        self.loss.backward()

        if self.hparams.max_gradient_norm > 0:
            nn.utils.clip_grad_norm_(
                self.propensity_model.parameters(), self.hparams.max_gradient_norm)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.hparams.max_gradient_norm)
            nn.utils.clip_grad_norm_(self.w_model.parameters(),self.hparams.max_gradient_norm)

        self.opt_denoise.step()
        self.opt_ranker.step()
        self.opt_weight.step()

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
        self.w_model.train()
        self.propensity_model.train()
        self.create_input_feed(input_feed, self.rank_list_size)
        self.rank_loss = 0

        # print(input_feed)

        # start train
        features = input_feed['feat_input']
        feat_emb, train_output = self.model(features, 1)
        train_output = train_output.reshape(-1, self.rank_list_size)

        train_labels = self.labels

        bs = train_labels.shape[0]
        # [bs 10]
        propensity_labels = torch.transpose(train_labels,0,1)
        self.propensity = self.propensity_model(
            propensity_labels)

        with torch.no_grad():
            self.propensity_weights = self.get_normalized_weights(
                self.logits_to_prob(self.propensity))
            self.relevance_weights = self.get_normalized_weights(
                self.logits_to_prob(train_output[:, :self.max_candidate_num]))

        fs=single_dis
        self.w_loss=0
        w1,self.w_loss=self.w_model(train_labels.detach(),fs)
        print(w1)

        self.rank_loss = self.rank_loss + self.loss_func(
            train_output, train_labels,propensity_weights=self.propensity_weights, weight=w1)
        self.exam_loss = self.loss_func(
            self.propensity,
            train_labels,
            propensity_weights=self.relevance_weights,
            weight=w1.detach())
      
        self.loss = self.exam_loss + self.hparams.ranker_loss_weight * self.rank_loss + self.w_loss
        print(self.rank_loss.item(),self.exam_loss.item())
        self.separate_gradient_update()

        self.clip_grad_value(train_labels, clip_value_min=0, clip_value_max=1)
        self.global_step += 1
        return self.loss.item()

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
    
    def state_dict(self):
        return {'model': self.model.state_dict(
        ), 'propensity_model': self.propensity_model.state_dict(),'w_model':self.w_model.state_dict()}

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])
        self.propensity_model.load_state_dict(state_dict['propensity_model'])
        self.w_model.load_state_dict(state_dict['w_model'])