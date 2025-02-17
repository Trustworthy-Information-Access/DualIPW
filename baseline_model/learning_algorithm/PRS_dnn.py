"""
Nan Wang, Zhen Qin, Xuanhui Wang, and Hongning Wang. 2021. Non-clicks mean irrelevant? propensity ratio scoring as a correction. In WSDM’21. 481–489.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from args import config
import numpy as np
import json

from baseline_model.learning_algorithm.base_algorithm import BaseAlgorithm
import baseline_model.utils as utils


class PRS_DNN(BaseAlgorithm):
    """The navie algorithm that directly trains ranking models with input labels.

    """

    def __init__(self, exp_settings, ranking_model):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
        """
        print('Build PRS_DNN')

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
        train_output = self.model(features)
        train_output=train_output.reshape(-1,self.rank_list_size)
        train_labels = self.labels
        bs=train_labels.shape[0]
        propensity_weight = json.load(open('./baseline_model/pw.json','r'))
        propensity_weight=propensity_weight['IPW_list'][:self.max_candidate_num]
        propensity_weight = torch.FloatTensor(propensity_weight).cuda()
        propensity_weight=propensity_weight.repeat(bs).reshape([-1,self.rank_list_size])

        self.loss = None

        ipw=propensity_weight
        pw=1/propensity_weight
        
        preds_sorted, preds_sorted_inds = torch.sort(train_output, dim=1, descending=True)
        labels_sorted_via_preds = torch.gather(train_labels, dim=1, index=preds_sorted_inds)
        ipw_sorted_via_preds = torch.gather(ipw, dim=1, index=preds_sorted_inds)
        pw_sorted_via_preds = torch.gather(pw, dim=1, index=preds_sorted_inds)

        #calculate the prs score using the pw of unclick document and ipw of clicked document
        prs = torch.unsqueeze(ipw_sorted_via_preds, dim=2) * torch.unsqueeze(pw_sorted_via_preds, dim=1)

        std_diffs = torch.unsqueeze(labels_sorted_via_preds, dim=2) - torch.unsqueeze(
            labels_sorted_via_preds, dim=1)  # standard pairwise differences, i.e., S_{ij}
        std_Sij = torch.clamp(std_diffs, min=-1.0, max=1.0)  # ensuring S_{ij} \in {-1, 0, 1}
        std_p_ij = 0.5 * (1.0 + std_Sij)
        
        prs=torch.where(std_p_ij==torch.ones_like(std_p_ij),prs,1/prs)
        prs = torch.triu(prs, diagonal=1)
        
        s_ij = torch.unsqueeze(preds_sorted, dim=2) - torch.unsqueeze(preds_sorted,dim=1)  # computing pairwise differences, i.e., s_i - s_j
        p_ij = 1.0 / (torch.exp(-s_ij) + 1.0)
        ideally_sorted_labels, _ = torch.sort(train_labels, dim =1, descending=True)
        delta_NDCG = self.compute_delta_ndcg(ideally_sorted_labels, labels_sorted_via_preds)
        self.loss = F.binary_cross_entropy(torch.triu(p_ij, diagonal=1),
                                            torch.triu(std_p_ij,diagonal=1),reduction='none')#,torch.triu(delta_NDCG,diagonal=1), reduction='none')
        
        self.loss = self.loss * prs
        self.loss = torch.sum(self.loss)/bs
        
        params = self.model.parameters()
        if self.hparams.l2_loss > 0:
            loss_l2 = 0.0
            for p in params:
                loss_l2 += self.l2_loss(p)
            self.loss += self.hparams.l2_loss * loss_l2

        self.opt_step(self.optimizer_func, params)

        nn.utils.clip_grad_value_(train_labels, 1)
        return self.loss.item()

    def dcg(self, labels):
        """Computes discounted cumulative gain (DCG).

        DCG =  SUM((2^label -1) / (log(1+rank))).

        Args:
         labels: The relevance `Tensor` of shape [batch_size, list_size]. For the
           ideal ranking, the examples are sorted by relevance in reverse order.
          weights: A `Tensor` of the same shape as labels or [batch_size, 1]. The
            former case is per-example and the latter case is per-list.

        Returns:
          A `Tensor` as the weighted discounted cumulative gain per-list. The
          tensor shape is [batch_size, 1].
        """
        list_size = labels.shape[1]
        position = torch.arange(1, list_size + 1,dtype=torch.float32).cuda()
        denominator = torch.log(position + 1)
        numerator = torch.pow(torch.tensor(2.0).cuda(), labels.to(torch.float32)) - 1.0
        return torch.sum(numerator/denominator)

    def compute_delta_ndcg(self, ideally_sorted_stds, stds_sorted_via_preds):
        '''
        Delta-nDCG w.r.t. pairwise swapping of the currently predicted ltr_adhoc
        :param batch_stds: the standard labels sorted in a descending order
        :param batch_stds_sorted_via_preds: the standard labels sorted based on the corresponding predictions
        :return:
        '''
        # ideal discount cumulative gains
        batch_idcgs = self.dcg(ideally_sorted_stds)

        batch_gains = torch.pow(2.0, stds_sorted_via_preds) - 1.0

        batch_n_gains = batch_gains / batch_idcgs  # normalised gains
        batch_ng_diffs = torch.unsqueeze(batch_n_gains, dim=2) - torch.unsqueeze(batch_n_gains, dim=1)

        batch_std_ranks = torch.arange(stds_sorted_via_preds.size(1)).type(torch.cuda.FloatTensor).cuda()
        batch_dists = 1.0 / torch.log2(batch_std_ranks + 2.0)  # discount co-efficients
        batch_dists = torch.unsqueeze(batch_dists, dim=0)
        batch_dists_diffs = torch.unsqueeze(batch_dists, dim=2) - torch.unsqueeze(batch_dists, dim=1)
        batch_delta_ndcg = torch.abs(batch_ng_diffs) * torch.abs(
            batch_dists_diffs)  # absolute changes w.r.t. pairwise swapping

        return batch_delta_ndcg

    def get_scores(self, input_feed):
        self.model.eval()
        features = input_feed['feat_input']
        scores = self.model(features)
        return scores

    def state_dict(self):
        return {'model': self.model.state_dict()}

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])
