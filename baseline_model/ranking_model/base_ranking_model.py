"""The basic class that contains all the API needed for the implementation of a ranking model.

"""
from abc import ABC, abstractmethod

import torch.nn as nn
import torch.nn.functional as F
import torch


def selu(x):
    """ Create the scaled exponential linear unit (SELU) activation function. More information can be found in
            Klambauer, G., Unterthiner, T., Mayr, A. and Hochreiter, S., 2017. Self-normalizing neural networks. In Advances in neural information processing systems (pp. 971-980).

        Args:
            x: (tf.Tensor) A tensor containing a set of numbers

        Returns:
            The tf.Tensor produced by applying SELU on each element in x.
        """
    # with tf.name_scope('selu') as scope:
    #     alpha = 1.6732632423543772848170429916717
    #     scale = 1.0507009873554804934193349852946
    #     return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * torch.where(x >= 0.0, x, alpha * F.elu(x))


class ActivationFunctions(object):
    """Activation Functions key strings."""

    ELU = 'elu'

    RELU = 'relu'

    SELU = 'selu'

    TANH = 'tanh'

    SIGMOID = 'sigmoid'


class NormalizationFunctions(object):
    """Normalization Functions key strings."""

    BATCH = 'batch'

    LAYER = 'layer'


class Initializer(object):
    """Initializer key strings."""

    CONSTANT = 'constant'


class BaseRankingModel(ABC, nn.Module):

    ACT_FUNC_DIC = {
        ActivationFunctions.ELU: nn.ELU(),
        ActivationFunctions.RELU: nn.ReLU(),
        ActivationFunctions.SELU: selu,
        ActivationFunctions.TANH: nn.Tanh(),
        ActivationFunctions.SIGMOID: nn.Sigmoid()
    }

    NORM_FUNC_DIC = {
        NormalizationFunctions.BATCH: F.batch_norm,
        NormalizationFunctions.LAYER: F.layer_norm
    }

    model_parameters = {}

    @abstractmethod
    def __init__(self, hparams_str=None, **kwargs):
        """Create the network.

        Args:
            hparams_str: (string) The hyper-parameters used to build the network.
        """
        pass

    @abstractmethod
    def build(self, input_list, noisy_params=None,
              noise_rate=0.05, is_training=False, **kwargs):
        """ Create the model

        Args:
            input_list: (list<tf.tensor>) A list of tensors containing the features
                        for a list of documents.
            noisy_params: (dict<parameter_name, tf.variable>) A dictionary of noisy parameters to add.
            noise_rate: (float) A value specify how much noise to add.
            is_training: (bool) A flag indicating whether the model is running in training mode.

        Returns:
            A list of tf.Tensor containing the ranking scores for each instance in input_list.
        """
        pass
