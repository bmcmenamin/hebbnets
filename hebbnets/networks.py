"""
    Class contains a pure-python/Numpy implementation for a multilayer network
    of feedforward Hebbian/Anti-Hebbian (HAH) neurons
"""
import collections
import random

import numpy as np

from hebbnets import base_network
from hebbnets import layers
from hebbnets import utils


class MultilayerHahNetwork(base_network.BaseNetwork):
    """A network built from one or more layers of HAH layers"""

    LAYER_TYPE = layers.HahLayer

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


class MultilayerDahEmbedding(base_network.BaseNetwork):
    """A network built from one or more layers of Delta-Anti/Hebbian layers"""

    LAYER_TYPE = layers.HahLayer

    def _train_epoch(self, data_set, num_pairs_per_sample=5):
        """Perform an epoch-worth of model updates

        Args:
            data_set: an iterable of tuples of (trainingdata, classlabel) pairs
            num_pairs_per_sample: integer number of other train pairs to use per sample
        Returns:
            None
        """

        for targ_value1, targ_class1 in data_set:

            targ_dataset = random.sample(data_set, num_pairs_per_sample)
            random.shuffle(targ_dataset)

            for targ_value2, targ_class2 in targ_dataset:

                class_match = 4.0 * ((targ_class1 == targ_class2) - 0.25)

                self.propogate_input(targ_value1)
                target_activations = [
                    class_match * layer.activation.copy() for layer in self.layers
                ]

                self.propogate_input(targ_value2)
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


class MultilayerQAGRELNetwork(base_network.BaseNetwork):
    """A network built from one or more layers of Q-AGREL layers"""

    LAYER_TYPE = layers.QagrelLayer

    def _build_layers(self, input_size, layer_sizes, **kwargs):

        kwargs_final = collections.ChainMap({'act_type': 'linear'}, kwargs)

        layer_list = []
        for idx, layer_size in enumerate(layer_sizes):
            if idx == 0:
                _layer = self.LAYER_TYPE(layer_size, input_size, **kwargs)
            elif idx == len(layer_sizes) - 1:
                _layer = self.LAYER_TYPE(layer_size, layer_list[-1], **kwargs_final)
            else:
                _layer = self.LAYER_TYPE(layer_size, layer_list[-1], **kwargs)
            layer_list.append(_layer)
        return layer_list

    def __init__(self, input_size, layer_sizes, **kwargs):

        kwargs = collections.ChainMap({'has_bias': False}, kwargs)
        self.rew_prederr = 0
        self.action_temperature = kwargs.get('action_temperature', 0.5)
        self.learning_rate = kwargs.get('learning_rate', 0.01)

        super().__init__(input_size, layer_sizes, **kwargs)

        all_layer_sizes = [input_size] + layer_sizes
        fb_input_size, fb_layer_sizes = all_layer_sizes[-1], all_layer_sizes[:-1]

        self.layers = self._build_layers(input_size, layer_sizes)
        self.fb_layers = self._build_layers(fb_input_size, fb_layer_sizes[::-1])

    def select_action(self):
        """Look over layer activations and select the action that wins
        Args:
            None
        Returns:
            action_idx: integer indicating which action is selected
            action_vector: numpy one-hot vector of selected action
        """

        activation = utils.softmax(self.layers[-1].activation, temp=0.1)

        if random.random() < self.action_temperature:
            top_choice = np.random.choice(len(activation), p=activation.ravel())
        else:
            top_choice = np.random.choice(np.flatnonzero(activation == activation.max()))

        action_vector = np.zeros(activation.shape)
        action_vector[top_choice] = 1.0
        return top_choice, action_vector

    def _unidir_weight_update(self, in_layers, gate_layers, init_layer_val, init_gate_val):
        """Update the weights for one direction in the network (i.e. for the feedforward
        of feedback network

        Args:
            in_layers: iterable of layers, starting with an input layer and progressing
              through the whole net. These are the layers with the weights to be updated.
            gate_layers: iterable of layers that matches the length and node-sizes of layers
              in in_layers. The activation values in these layers are used for attention gating.
            init_layer_val: the input pattern to feed into `in_layers` and propogate forward
            init_gate_val: the inpust pattern to feed into `gate_layers` and propogate backward.
        """
        for idx, (layer, gate_layer) in enumerate(zip(in_layers, gate_layers[::-1][1:])):
            layer.update_weights(
                gate_layer.activation,
                self.rew_prederr,
                self.learning_rate,
                layer_input_val=init_layer_val if idx == 0 else None
            )

        in_layers[-1].update_weights(
            init_gate_val,
            self.rew_prederr,
            self.learning_rate,
            layer_input_val=None
        )

    def _train_epoch(self, data_set):
        """Perform an epoch-worth of model updates

        Args:
            data_set: an iterable of tuples of (trainingdata, classlabel) pairs
            num_pairs_per_sample: integer number of other train pairs to use per sample
        Returns:
            None
        """

        for in_value, target_idx in data_set:

            # Forward pass
            self.propogate_input(in_value, layer_attr='layers')

            # select action, measure it's error
            action_idx, action_vector = self.select_action()
            self.rew_prederr = float(action_idx == target_idx) - self.layers[-1].activation[action_idx]

            # backward pass
            self.propogate_input(action_vector, layer_attr='fb_layers')

            # Update FF, FB weights
            self._unidir_weight_update(
                self.layers, self.fb_layers,
                in_value, action_vector)

            self._unidir_weight_update(
                self.fb_layers, self.layers,
                action_vector, in_value)
