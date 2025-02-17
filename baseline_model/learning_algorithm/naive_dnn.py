import torch
import torch.nn as nn
import torch.nn.functional as F
from args import config
import numpy as np
import json


from baseline_model.learning_algorithm.base_algorithm import BaseAlgorithm
import baseline_model.utils as utils

class NaiveAlgorithm_DNN(BaseAlgorithm):
    """The navie algorithm that directly trains ranking models with input labels.

    """

    def __init__(self, exp_settings, ranking_model):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
        """
        print('Build NaiveAlgorithm')

        self.hparams = utils.hparams.HParams(
            learning_rate=exp_settings['lr'],                 # Learning rate.
            max_gradient_norm=0.5,            # Clip gradients to this norm.
            loss_func='softmax_loss',            # Select Loss function
            # Set strength for L2 regularization.
            l2_loss=0.0,
            grad_strategy='ada',            # Select gradient strategy
        )
        print(exp_settings['learning_algorithm_hparams'])
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings

        self.max_candidate_num = exp_settings['max_candidate_num']
        self.rank_feature_size = exp_settings['rank_feature_size']
        if 'selection_bias_cutoff' in self.exp_settings.keys():
            self.rank_list_size = self.exp_settings['selection_bias_cutoff']

        # single gpu trained directly
        self.model = ranking_model
        self.model.cuda()

        self.learning_rate = float(self.hparams.learning_rate)
        self.global_step = 0

        # Feeds for inputs.
        self.labels_name = []  # the labels for the documents (e.g., clicks)
        self.labels = []  # the labels for the documents (e.g., clicks)
        for i in range(self.rank_list_size):
            self.labels_name.append("label{0}".format(i))

        self.optimizer_func = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate)

        if self.hparams.grad_strategy == 'sgd':
            self.optimizer_func = torch.optim.SGD(
                self.model.parameters(), lr=self.learning_rate)

    def train(self, input_feed, epoch=0):
        """Run a step of the model feeding the given inputs for training process.

        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.

        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a tf.summary containing related information about the step.

        """
        self.global_step += 1
        self.model.train()
        self.create_input_feed(input_feed, self.rank_list_size)

        # Gradients and SGD update operation for training the model.
        features = input_feed['feat_input']
        cnt=features.shape[0]
        train_output = self.model(features,0)
        train_output = train_output.reshape(-1, self.rank_list_size)
        train_labels = self.labels
        
        propensity_weight = None

        # 当前data得到
        if config.ipw:
            propensity_weight = json.load(open('./baseline_model/pw.json','r'))
            propensity_weight=propensity_weight['IPW_list'][:self.max_candidate_num]
            propensity_weight = torch.FloatTensor(propensity_weight).cuda()
          
            propensity_weight=propensity_weight.clip(min=1,max=config.limit)
       
        self.loss = None

        if self.hparams.loss_func == 'sigmoid_loss':
            self.loss=self.pointwise_ips(train_output,train_labels,propensity_weights=propensity_weight)
            # self.loss = self.sigmoid_loss_on_list(
            #      train_output, train_labels, propensity_weights=propensity_weight)
        elif self.hparams.loss_func == 'pairwise_loss':
            self.loss = self.pairwise_loss_on_list(
                train_output, train_labels, propensity_weights=propensity_weight)
        else:
            self.loss = self.softmax_loss(
                train_output, train_labels, propensity_weights=propensity_weight)

        if self.hparams.loss_func != 'softmax_loss':
            self.loss=self.loss/cnt
        
        params = self.model.parameters()
        if self.hparams.l2_loss > 0:
            loss_l2 = 0.0
            for p in params:
                loss_l2 += self.l2_loss(p)
            self.loss += self.hparams.l2_loss * loss_l2

        self.opt_step(self.optimizer_func, params)

        nn.utils.clip_grad_value_(train_labels, 1)
        return self.loss.item()

    def get_scores(self, input_feed):
        self.model.eval()
        features = input_feed['feat_input']
        scores = self.model(features)
        return scores
    
    def get_loss(self,input_feed):
        self.model.eval()
        features=input_feed['feat_input']
        return self.model(features)

    def state_dict(self):
        return {'model': self.model.state_dict()}

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])
        