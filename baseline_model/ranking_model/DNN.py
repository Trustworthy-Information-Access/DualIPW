import torch
from torch import nn
from .base_ranking_model import BaseRankingModel
import baseline_model.utils as utils
from args import config


class DNN(nn.Module):

    def __init__(self, projection, feature_size):
        super(DNN, self).__init__()
        self.hparams = utils.hparams.HParams(
            activation_func='elu',
            norm='layer'
        )
        self.initializer = None
        self.act_func = None
        self.projection = projection
        self.feature_size = feature_size

        if feature_size == 64:
            self.hidden_layer_size = [32, 16, 8]
        elif feature_size == 72 or feature_size == 128:
            self.hidden_layer_size = [64, 32, 16]
        elif feature_size>512:
            self.hidden_layer_size = [512, 256, 128]
        elif feature_size>256:
            self.hidden_layer_size = [256, 128, 64]

        self.output_sizes = self.hidden_layer_size+[1]
        self.layer_norm = None
        self.projection = projection
        self.pro_bef = config.pro_bef
        self.pro_aft = config.pro_aft

        if projection:
            # embedding+feat | feat
            self.projection_feat = nn.Linear(self.pro_bef, self.pro_aft)

        self.sequential = nn.Sequential().to(dtype=torch.float32)
        if self.hparams.activation_func in BaseRankingModel.ACT_FUNC_DIC:
            self.act_func = BaseRankingModel.ACT_FUNC_DIC[self.hparams.activation_func]
        for i in range(len(self.output_sizes)):
            if self.layer_norm is None and self.hparams.norm in BaseRankingModel.NORM_FUNC_DIC:
                if self.hparams.norm == 'layer':
                    self.sequential.add_module('layer_norm{}'.format(i),
                                                nn.LayerNorm(feature_size).to(dtype=torch.float32))
                else:
                    self.sequential.add_module('batch_norm{}'.format(i),
                                                nn.BatchNorm2d(feature_size).to(dtype=torch.float32))
            self.sequential.add_module('linear{}'.format(
                i), nn.Linear(feature_size, self.output_sizes[i]))
            if i != len(self.output_sizes)-1:
                self.sequential.add_module(
                    'act{}'.format(i), self.act_func)
            feature_size = self.output_sizes[i]

    def forward(self, input,need_input=0,noisy_params=None, noise_rate=0.05, grad=True, **kwargs):
        """ Create the DNN model

        Args:
            input: [batch_size,feature_size]
            noisy_params: (dict<parameter_name, torch.variable>) A dictionary of noisy parameters to add.
            noise_rate: (float) A value specify how much noise to add.
            is_training: (bool) A flag indicating whether the model is running in training mode.

        Returns:
            [batch_size,1] containing the ranking scores for each instance in input.
        """
        input = input.cuda()
        if self.projection and input.shape[-1]==self.pro_bef:
            input = self.projection_feat(input[:, :self.pro_bef])
        if noisy_params == None:
            if grad:
                output = self.sequential(input)
            else:
                output=self.sequential(input.detach())
        else:
            for name, parameter in self.sequential.named_parameters():
                if name in noisy_params:
                    with torch.no_grad():
                        noise = noisy_params[name]*noise_rate
                        if torch.cuda.is_available():
                            noise = noise.to(input.device)
                        parameter += noise
        # return input, output.squeeze(dim=-1)
        if need_input:
            return input, output.squeeze(dim=-1)
        else:
            return output.squeeze(dim=-1)

