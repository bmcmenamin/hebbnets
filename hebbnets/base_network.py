"""
    Class contains a pure-python/Numpy implementation for a multilayer network
    of feedforward Hebbian/Anti-Hebbian (HAH) neurons
"""
import abc
import random

import numpy as np

from hebbnets import base_layer

class BaseNetwork(abc.ABC):
    """Abstract baseclass for networks"""

    LAYER_TYPE = base_layer.BaseLayer

    def __init__(self, input_size, layer_sizes, **kwargs):
        self.input_value = np.zeros(input_size,)
        self.input_size = input_size
        self.layers = []
        for idx, layer_size in enumerate(layer_sizes):
            if idx == 0:
                _layer = self.LAYER_TYPE(layer_size, input_size, **kwargs)
            else:
                _layer = self.LAYER_TYPE(layer_size, self.layers[-1], **kwargs)
            self.layers.append(_layer)

    def _set_input_value(self, input_value):
        """Helper function to check and set input values"""
        if isinstance(input_value, list):
            input_value = np.array(input_value)

        if not isinstance(input_value, np.ndarray):
            raise TypeError(
                "Layer fed input of type {}, expecting size list or numpy array".
                format(type(input_value))
            )

        self.input_value = input_value

    def train(self, data_set, num_epochs=3):
        """Perform an epoch-worth of model updates

        Args:
            data_set: an iterable of training data
            num_epochs: number of epochs to run training
        Returns:
            None
        """

        _data = data_set.copy()
        for _ in range(num_epochs):
            random.shuffle(_data)
            self._train_epoch(_data)

    @abc.abstractmethod
    def _train_epoch(self, data_set):
        """Perform an epoch-worth of model updates

        Args:
            data_set: an iterable of training data
        Returns:
            None
        """
        raise NotImplementedError

    def propogate_input(self, input_value, layer_attr='layers'):
        """Given an input value, propgate activation forward through the
        whole network

        Args:
            input_value: numpy array of list of input activation vlaues
        Returns:
            None, sets activations in place
        """
        self._set_input_value(input_value)
        for idx, layer in enumerate(getattr(self, layer_attr)):
            if idx == 0:
                layer.update_activation(input_value=self.input_value)
            else:
                layer.update_activation()