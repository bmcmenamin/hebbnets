"""
    Class contains a pure-python/Numpy implementation for the layer of a feedforward
    network that learns using Hebbian/Anti-Hebbian (HAH) weight updates and a
    layer for Q-AQREL reinforcement learning
"""
import numpy as np
import random

from hebbnets import base_layer
from hebbnets import utils


class FeedforwardLayer(base_layer.BaseLayer):
    """Layer of neurons that use simple feedforward weights
    """

    def __init__(self, num_nodes, prev_layer, **kwargs):
        """Initialize layer
        Args:
            num_nodes: Number of nodes in this layer
            prev_layer: The previous layer of the network
                If this is an input layer, set to with integer for input
                size
            act_type: String naming the type of activation function
                to use
            has_bias: Bool indicating whether layer has bias
        Returns:
            None
        """

        super().__init__(num_nodes, prev_layer, **kwargs)

        self.input_weights = self.glorot_init(
            self.layer_input_size + self.params['bias'],
            self.num_nodes
        )

    def _rescale_weights(self):
        np.clip(
            self.input_weights,
            -self.MAX_WEIGHT, self.MAX_WEIGHT,
            out=self.input_weights
        )

    def update_activation(self, input_value=None):
        """Update activation  in forward pass for Q-AGREL layer
        Args:
            input_value: set to list/numpy array being fed as input into this
               layer, otherwise input will be inferred from a previous layer
        Returns:
            None. Updates self.activation in place
        """
        input_value = self._get_input_values(input_value)
        input_value_times_weights = np.matmul(self.input_weights.T, input_value)
        self.activation = self.activation_func(input_value_times_weights)


class QagrelLayer(FeedforwardLayer):
    """Layer of neurons that use simple feedforward weights
    and Q-AGREL weight updating
    """

    def update_weights(self, gate_value, rew_prederr, learning_rate, layer_input_val=None):
        """Update weights using Q-AGREL rules

        Args:
            gate_value: vector with size matching number of units in this layer
             indicating gating strength
            rew_prederr: scalar prediction error
            learning_rate: scalar learning rate
            layer_input_val: input value for this layer (or none to use prev lyaer activation)
        Returns:
            None, updates weight in place
        """

        target_vec = self.activation_deriv_func(self.activation) * gate_value.reshape(-1, 1)
        outer_prod = np.matmul(
            self._get_input_values(layer_input_val),
            target_vec.T
        )
        self.input_weights += learning_rate * rew_prederr * outer_prod
        self._rescale_weights()


class HahLayer(base_layer.BaseLayer):
    """Layer of neurons that are activated and updated using a
    Hebbian/AntHebbian (HAH) pattern
    """

    MAX_ACTIVATION_TIME_STEPS = 200
    ACTIVATION_ERROR_TOL = 1.0e-2
    CUMSCORE_LR = 0.01

    def __init__(self, num_nodes, prev_layer, **kwargs):
        """Initialize layer
        Args:
            num_nodes: Number of nodes in this layer
            prev_layer: The previous layer of the network
                If this is an input layer, set to with integer for input
                size
            act_type: String naming the type of activation function
                to use
            has_bias: Bool indicating whether layer has bias
            noise_var: variance for guassian noise applied to all activations

        Returns:
            None
        """

        super().__init__(num_nodes, prev_layer, **kwargs)

        self.input_weights = self.glorot_init(
            self.layer_input_size + self.params['bias'],
            self.num_nodes
        )

        self.lateral_weights = self.glorot_init(
            self.num_nodes,
            self.num_nodes,
        )
        np.fill_diagonal(self.lateral_weights, 0.0)

        self._cum_sqr_activation = np.tile(1000.0, (num_nodes, 1))

    def update_activation(self, input_value=None):
        """Update activation value for Hebb/Antihebb layer
        Args:
            input_value: set to list/numpy array being fed as input into this
               layer, otherwise input will be inferred from a previous layer
        Returns:
            None. Upates self.activation in place
        """
        input_value = self._get_input_values(input_value)
        self.activation = np.zeros((self.num_nodes, 1), dtype='float64')

        input_value_times_weights = np.matmul(self.input_weights.T, input_value)
        if self.params['act_type'] == 'linear':
            _lateral_inv = np.linalg.pinv(self.lateral_weights.T + np.eye(self.num_nodes))
            self.activation = np.matmul(_lateral_inv, input_value_times_weights)
        else:
            for _ in range(self.MAX_ACTIVATION_TIME_STEPS):
                next_activation = self.activation_func(
                    input_value_times_weights - np.matmul(self.lateral_weights.T, self.activation)
                )

                error = utils.max_abs_reldiff(self.activation, next_activation)
                self.activation = next_activation
                if error < self.ACTIVATION_ERROR_TOL:
                    break

    def _rescale_weights(self):
        np.clip(
            self.input_weights,
            -self.MAX_WEIGHT, self.MAX_WEIGHT,
            out=self.input_weights
        )

        np.clip(
            self.lateral_weights,
            -self.MAX_WEIGHT, self.MAX_WEIGHT,
            out=self.lateral_weights
        )

    def _calc_weight_delta(self, input_value, target_value):
        target_norm = target_value / self._cum_sqr_activation
        outer_prod = np.outer(input_value, target_norm)
        ident = np.eye(len(target_norm))
        return np.matmul(outer_prod, ident - target_norm.T)

    def update_weights(self, input_value=None, target_value=None):
        """Update input weights with Hebbian/Antihebbian rule
        Args:
            input_value: set to list/numpy array being fed as input into this
               layer, otherwise input will be inferred from a previous layer
           target_value: set to list/numpy array with length equal to number of
               nodes for supervised learning via delta rule. Leave as none for
               unsupervised hebbian updates
        Returns:
            None. Upates self.input_weights, self.lateral_weights in place
        """

        # Get weight deltas using modified Oja rule
        input_value = self._get_input_values(input_value)

        if target_value is None:
            _target_value = self.activation
        else:
            _target_value = target_value - self.activation

        if self.params.get('noise_var', 0) > 0:
            _target_value += (
                self.params['noise_var'] * np.random.randn(*_target_value.shape).astype('float64')
            )

        # update cumulative activation per node, used for adaptive learning-rate scaling
        self._cum_sqr_activation += self.CUMSCORE_LR * self.activation ** 2

        self.input_weights += self._calc_weight_delta(input_value, _target_value)
        self.lateral_weights += self._calc_weight_delta(self.activation, self.activation)

        # Enforce constraints on weights
        np.fill_diagonal(self.lateral_weights, 0.0)
        self._rescale_weights()
