"""The basic class that contains all the API needed for the implementation of a ranking model.

"""

from __future__ import print_function
from __future__ import absolute_import
from abc import ABC, abstractmethod

import paddle.nn as nn
import paddle.nn.functional as F
import paddle

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


class BaseRankingModel(ABC, nn.Layer):

    ACT_FUNC_DIC = {
        ActivationFunctions.ELU: nn.ELU(),
        ActivationFunctions.RELU: nn.ReLU(),
        ActivationFunctions.SELU: nn.SELU(),
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
            input_list: (list<paddle.tensor>) A list of tensors containing the features
                        for a list of documents.
            noisy_params: (dict<parameter_name, paddle.variable>) A dictionary of noisy parameters to add.
            noise_rate: (float) A value specify how much noise to add.
            is_training: (bool) A flag indicating whether the model is running in training mode.

        Returns:
            A list of paddle.Tensor containing the ranking scores for each instance in input_list.
        """
        pass
