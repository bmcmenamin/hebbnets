"""
    Class contains a pure-python/Numpy implementation for a multilayer network
    of feedforward Hebbian/Anti-Hebbian (HAH) neurons
"""
import abc
import itertools
import random

import numpy as np

from hebbnets.hebbnets.layers import HahLayer

class Network(abc.ABC):
    """Abstract baseclass for networks"""

    def __init__(self, input_size, layer_sizes, **kwargs):
        self.input_value = np.zeros(input_size,)
        self.input_size = input_size
        self.layers = []
        for idx, layer_size in enumerate(layer_sizes):
            if idx == 0:
                _layer = HahLayer(layer_size, input_size, **kwargs)
            else:
                _layer = HahLayer(layer_size, self.layers[-1], **kwargs)
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

        if len(input_value) != self.input_size:
            raise ValueError(
                "Layer fed input of size {}, expecting size {}".
                format(len(input_value), self.input_size)
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

    def propogate_input(self, input_value):
        """Given an input value, propgate activation forward through the
        whole network

        Args:
            input_value: numpy array of list of input activation vlaues
        Returns:
            None, sets activations in place
        """
        self._set_input_value(input_value)
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                layer.update_activation(input_value=self.input_value)
            else:
                layer.update_activation()


class MultilayerHahNetwork(Network):
    """A network built from one or more layers of HAH layers"""

    def _train_epoch(self, data_set):
        """Perform an epoch-worth of model updates

        Args:
            data_set: an iterable of training data
        Returns:
            None
        """
        for samp in data_set:
            self.propogate_input(samp)
            self.update_weights()

    def update_weights(self):
        """Update model weights based on current activation

        Args:
            None
        Returns:
            None, modifies weights in place
        """
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                layer.update_weights(input_value=self.input_value)
            else:
                layer.update_weights()


class MultilayerHahEmbedding(Network):
    """A network built from one or more layers of HAH layers"""

    def _train_epoch(self, data_set, num_pairs_per_sample=10):
        """Perform an epoch-worth of model updates

        Args:
            data_set: an iterable of tuples of (trainingdata, classlabel) pairs
            num_pairs_per_sample: integer number of other train pairs to use per sample
        Returns:
            None
        """

        for samp1, class1 in data_set:
            for samp2, class2 in random.sample(data_set, num_pairs_per_sample):

                class_match = 2 * ((class1 == class2) - 0.5)

                self.propogate_input(samp1)
                target_activations = [
                    class_match * layer.activation.copy()
                    for layer in self.layers
                ]
                self.propogate_input(samp2)
                self.update_weights(target_activations)

    def update_weights(self, target_activations):
        """Update model weights based on current activation

        Args:
            target_activations: list of target activations for each layer
        Returns:
            None, modifies weights in place
        """
        for idx, (layer, target) in enumerate(zip(self.layers, target_activations)):
            if idx == 0:
                layer.update_weights(
                    input_value=self.input_value,
                    target_value=target)
            else:
                layer.update_weights(target_value=target)
