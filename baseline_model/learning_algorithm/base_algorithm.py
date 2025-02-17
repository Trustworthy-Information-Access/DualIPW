"""The basic class that contains all the API needed for the implementation of an unbiased learning to rank algorithm.

"""
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from args import config

import baseline_model.utils as utils


def kl_loss(logits, labels):
    '''
    [bs seq]
    '''
    return -(F.softmax(labels)*F.log_softmax(logits)).sum(dim=1).mean()


def softmax_cross_entropy_with_logits(logits, labels):
    """Computes softmax cross entropy between logits and labels.

    Args:
        output: A tensor with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
        labels: A tensor of the same shape as `output`. A value >= 1 means a
        relevant example.
    Returns:
        A single value tensor containing the loss.
    """
    loss = torch.sum(- labels * F.log_softmax(logits, -1), -1)
    return loss

class BaseAlgorithm(ABC):
    """The basic class that contains all the API needed for the
        implementation of an unbiased learning to rank algorithm.

    """
    PADDING_SCORE = -100000

    @abstractmethod
    def __init__(self, exp_settings, encoder_model):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
        """
        self.is_training = None
        self.labels = None  # the labels for the documents (e.g., clicks)
        self.output = None  # the ranking scores of the inputs
        # the number of documents considered in each rank list.
        self.rank_list_size = None
        # the maximum number of candidates for each query.
        self.max_candidate_num = None
        self.optimizer_func = torch.optim.adagrad()
        self.best_model = None
        self.click_model = None
        self.model = encoder_model

    @abstractmethod
    def train(self, input_feed):
        """Run a step of the model feeding the given inputs for training.

        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.

        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a summary containing related information about the step.

        """
        pass

    def load_best_model(self, best_model):
        self.best_model = best_model
        

    def load_click_model(self, click_model):
        self.click_model = click_model

    def adjust(self,output,propensity):

        res_output,res_propensity=[],[]
        output=output.reshape([-1,self.rank_list_size])
        propensity=propensity.reshape([-1,self.rank_list_size])

        bs=output.shape[0]
        labels=self.labels.cpu().numpy().tolist()
        for i in range(bs):
            pos_idx,neg_idx=[],[]
            for j in range(self.rank_list_size):
                if labels[i][j]==1:
                    pos_idx.append(j)
                else:
                    neg_idx.append(j)
            for pos in pos_idx:
                for neg in neg_idx:
                    res_output.append(output[i,pos])
                    res_output.append(output[i,neg])
                    res_propensity.append(propensity[i,pos])
                    res_propensity.append(propensity[i,neg])
        return torch.stack(res_output),torch.stack(res_propensity)

    def create_input_feed(self, input_feed, list_size):
        self.labels = []
        self.docid_inputs = []
        for i in range(list_size):
            self.labels.append(input_feed[self.labels_name[i]])
            if 'UPE' in config.method_name:
                self.docid_inputs.append(input_feed[self.docid_inputs_name[i]])
        self.labels = np.transpose(self.labels)
        self.labels = torch.FloatTensor(self.labels).cuda()
        ones = torch.ones_like(self.labels)
        zeros = torch.zeros_like(self.labels)
        self.labels = torch.where(
            self.labels == -1 * ones, zeros, self.labels)
        if 'UPE' in config.method_name:
            self.docid_inputs = torch.FloatTensor(self.docid_inputs).cuda()

    def opt_step(self, opt, params):
        """ Perform an optimization step

        Args:
            opt: Optimization Function to use
            params_list: Model's parameters=>[encoder_model][encoder_model,ranking_model]

        Returns
            The ranking model that will be used to computer the ranking score.

        """
        opt.zero_grad()
        self.loss.backward()
        if self.hparams.max_gradient_norm > 0:
            if not isinstance(params, list):
                torch.nn.utils.clip_grad_norm_(
                    params, self.hparams.max_gradient_norm)
            else:
                for p in params:
                    torch.nn.utils.clip_grad_norm_(
                        p, self.hparams.max_gradient_norm)
        opt.step()

    def pairwise_cross_entropy_loss(self, pos_scores, neg_scores, propensity_weights=None):
        """Computes pairwise softmax loss without propensity weighting.

        Args:
            pos_scores: (torch.Tensor) A tensor with shape [batch_size, 1]. Each value is
            the ranking score of a positive example.
            neg_scores: (torch.Tensor) A tensor with shape [batch_size, 1]. Each value is
            the ranking score of a negative example.
            propensity_weights: (torch.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (torch.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = torch.ones_like(pos_scores)
        label_dis = torch.cat(
            [torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=1)
        loss = softmax_cross_entropy_with_logits(
            logits=torch.cat([pos_scores, neg_scores], dim=1), labels=label_dis) * propensity_weights
        return loss

    def pointwise_ips(self,output,labels,propensity_weights=None):
        if propensity_weights is None:
            propensity_weights = torch.ones_like(labels)
        sig=nn.Sigmoid()
        y1=sig(output)
        y0=1-y1
        log_p=torch.log(y1)
        log_not_p=torch.log(y0)
        loss=-(propensity_weights*labels)*log_p-(1-propensity_weights*labels)*log_not_p
        return loss.sum()
    
    def sigmoid_loss_on_list(self, output, labels,
                             propensity_weights=None):
        """Computes pointwise sigmoid loss without propensity weighting.

        Args:
            output: (torch.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (torch.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.
            propensity_weights: (torch.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (torch.Tensor) A single value tensor containing the loss.
        """
        zeros=torch.zeros_like(labels)
        ones=torch.ones_like(labels)
        if propensity_weights is None:
            propensity_weights = ones
        # 仅对click做IPW
        # propensity_weights=torch.where(labels==zeros,ones,propensity_weights)
        # 对non-click做PW
        # propensity_weights=torch.where(labels==zeros,1/propensity_weights,propensity_weights)
        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        # * propensity_weights
        loss = criterion(output, labels)* propensity_weights

        return loss.sum()

    def pairwise_loss_on_list(self, output, labels,
                              propensity_weights=None):
        """Computes pairwise entropy loss.

        Args:
            output: (torch.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (torch.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
                relevant example.
            propensity_weights: (torch.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (torch.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = torch.ones_like(labels)

        loss = None
        sliced_output = torch.unbind(output, dim=1)
        sliced_label = torch.unbind(labels, dim=1)
        sliced_propensity = torch.unbind(propensity_weights, dim=1)
        for i in range(len(sliced_output)):
            for j in range(i + 1, len(sliced_output)):
                cur_label_weight = torch.sign(
                    sliced_label[i] - sliced_label[j])
                cur_propensity = sliced_propensity[i] * \
                    sliced_label[i] + \
                    sliced_propensity[j] * sliced_label[j]
                cur_pair_loss = - \
                    torch.exp(
                        sliced_output[i]) / (torch.exp(sliced_output[i]) + torch.exp(sliced_output[j]))
                if loss is None:
                    loss = cur_label_weight * cur_pair_loss
                loss += cur_label_weight * cur_pair_loss * cur_propensity
        batch_size = labels.size()[0]
        return torch.sum(loss) / batch_size.type(torch.float32)

    def softmax_loss(self, output, labels, propensity_weights=None,weight=None):
        """Computes listwise softmax loss without propensity weighting.

        Args:
            output: (torch.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (torch.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.
            propensity_weights: (torch.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (torch.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = torch.ones_like(labels)
        # [bs seq] 每个query的num_candidates个rel_score
        # 不考虑click0
        weighted_labels = (labels + 0.0000001) * propensity_weights  # 平滑

        # weighted_labels = labels * propensity_weights # 无平滑
        label_dis = weighted_labels / \
            torch.sum(weighted_labels, 1, keepdim=True)
        label_dis = torch.nan_to_num(label_dis)
        loss = softmax_cross_entropy_with_logits(
            logits=output, labels=label_dis) * torch.sum(weighted_labels, 1)
        if weight is None:
            weight=torch.ones(labels.shape[0]).cuda()
        loss=loss*weight
        return torch.sum(loss) / torch.sum(weighted_labels).detach()


    def l2_loss(self, input):
        return torch.sum(input ** 2)/2

    def clip_grad_value(self, parameters, clip_value_min, clip_value_max) -> None:
        """Clips gradient of an iterable of parameters at specified value.

        Gradients are modified in-place.

        Args:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            clip_value (float or int): maximum allowed value of the gradients.
                The gradients are clipped in the range
                :math:`\left[\text{-clip\_value}, \text{clip\_value}\right]`
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        clip_value_min = float(clip_value_min)
        clip_value_max = float(clip_value_max)
        for p in filter(lambda p: p.grad is not None, parameters):
            p.grad.data.clamp_(min=clip_value_min, max=clip_value_max)

    def state_dict(self):
        return {'model': self.model.state_dict()}

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])
