"""
    Class contains a pure-python/Numpy implementation for the layer of a feedforward
    network that learns using Hebbian/Anti-Hebbian (HAH) weight updates
"""

import numpy as np

ABSMAX_WEIGHT_LIMIT = 1.0e4


def rescale_weights_in_place(weights):
    """Rescale every element of a wieght matrix in place if any
    of the values exceed the ABSMAX_WEIGHT_LIMIT
    """
    absmax_weight = max(weights.max(), -weights.min())
    if absmax_weight > ABSMAX_WEIGHT_LIMIT:
        weights /= (absmax_weight / ABSMAX_WEIGHT_LIMIT)


class ActivationFunctions(object):
    """Class to be used as mixin for providing activation functions in layer
    """

    @staticmethod
    def _relu(activation):
        return np.maximum(activation, 0, activation)

    @staticmethod
    def _sigmoid(activation):
        return 1.0 / (1 + np.exp(-activation))

    @staticmethod
    def _linear(activation):
        return activation

    def _soft_thresh(self, activation):
        abs_thresh = np.maximum(np.abs(activation) - self._soft_thresh_val, 0.0, activation)
        return np.sign(activation) * abs_thresh


class HahLayer(ActivationFunctions, object):
    """Layer of neurons that are activated and updated using a 
    Hebbian/AntHebbian (HAH) pattern
    """

    ACTIVATION_TIME_STEPS = 50
    MINIMUM_LEARNING_RATE = 1.0e-8

    def __repr__(self):

        to_repr = []
        for key, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                to_repr.append(
                    "{}={} size {}".format(key, type(val), val.shape)
                )
            else:
                to_repr.append(
                    "{}={}".format(key, val)
                )

        return "<{}>".format(",".join(to_repr))

    def __init__(self, num_nodes, prev_layer, act_type='linear', has_bias=True, reg_lambda=1.0, noise_var=0.001):
        """Set layer activation based on previous layer
        Args:
            num_nodes: Number of nodes in this layer
            prev_layer: The previous layer of the network
                If this is an input layer, set to with integer for input
                size
            act_type: String naming the type of activation function
                to use
            has_bias: Bool indicating whether layer has bias
            reg_lambda: Strength of soft-threshold regularization parameter
            noise_var: variance for guassian noise applied to all activations

        Returns:
            None
        """
        self.num_nodes = num_nodes
        self.prev_layer = prev_layer

        if isinstance(prev_layer, int):
            self.layer_input_size = prev_layer           
        elif isinstance(prev_layer, HahLayer):
            self.layer_input_size = prev_layer.num_nodes
        else:
            raise TypeError(
                "prev_layer is of type {}, expecting HahLayer or int".
                format(type(prev_layer))
            )

        self.input_weights = np.random.randn(
            self.layer_input_size + has_bias,
            self.num_nodes
        )
        if self.num_nodes > 1:
            self.input_weights /= self.num_nodes * np.linalg.norm(
                self.input_weights,
                axis=1, keepdims=True)

        self.lateral_weights = np.random.randn(
            self.num_nodes,
            self.num_nodes
        )
        np.fill_diagonal(self.lateral_weights, 0.0)
        if self.num_nodes > 1:
            self.lateral_weights /= self.num_nodes * np.linalg.norm(
                self.lateral_weights,
                axis=1, keepdims=True)

        self.ACTIVATION_FUNCS = {
            'relu': self._relu,
            'sigmoid': self._sigmoid,
            'linear': self._linear,
            'soft_thresh': self._soft_thresh
        }

        self.params = {
            'reg_lambda': reg_lambda,
            'bias': has_bias,
            'act_type': act_type,
            'noise_var': noise_var
        }

        if act_type not in self.ACTIVATION_FUNCS:
            raise NameError("Activation function named %s not defined", act_type)

        self.activation = np.zeros(self.num_nodes)

        self._cum_activation_l1 = np.tile(0.1, num_nodes)
        self._cum_activation_l2 = np.tile(0.1, num_nodes)
        self._soft_thresh_val = np.tile(0.0, num_nodes)

    def _get_input_value_format(self, input_value):

        if not isinstance(input_value, np.ndarray):
            try:
                input_value = np.array(input_value)
            except:
                raise TypeError(
                    "Layer fed input of type {}, expecting something that "
                    "could be cast to numpy array".
                    format(type(input_value))
                )

        if len(input_value) != self.layer_input_size:
            raise ValueError(
                "Layer fed input of size {}, expecting size {}".
                format(len(input_value), self.layer_input_size)
            )

        return input_value

    def _get_input_from_prev_layer(self):
        if not isinstance(self.prev_layer, HahLayer):
            raise TypeError(
                "Layer requires previous layer to be of type Layer. It's {}".
                format(type(self.prev_layer))
            )
        return self.prev_layer.activation

    def _get_input_values(self, input_value):
        """Get the input to this layer in the correct format
        """

        if input_value is None:
            input_value = self._get_input_from_prev_layer()
        else:
            input_value = self._get_input_value_format(input_value)

        if self.params['bias']:
            input_value = np.append(input_value, 1.0)

        return input_value

    def update_activation(self, input_value=None):
        """Update input weights with hebb rule
        Args:
            input_value: set to list/numpy array being fed as input into this
               layer, otherwise input will be inferred from a previous layer
        Returns:
            None. Upates self.activation in place
        """
        input_value = self._get_input_values(input_value)

        self.activation = np.zeros(self.num_nodes)
        for _ in range(self.ACTIVATION_TIME_STEPS):

            _next_step = self.ACTIVATION_FUNCS[self.params['act_type']](
                input_value.dot(self.input_weights) - self.activation.dot(self.lateral_weights)
            )

            self.activation = _next_step

            if np.allclose(_next_step, self.activation):
                break


    def _update_activation_counters(self, pad=1.0e-8):
        self._cum_activation_l1 += np.abs(self.activation)
        self._cum_activation_l2 += self.activation ** 2

        scale = 0.5 * self.params.get('reg_lambda', 0.0)
        self._soft_thresh_val = scale * self._cum_activation_l1 / (self._cum_activation_l2 + pad)


    def update_weights(self, input_value=None):
        """Update input weights with Hebbian/Antihebbian rule
        Args:
            input_value: set to list/numpy array being fed as input into this
               layer, otherwise input will be inferred from a previous layer
        Returns:
            None. Upates self.input_weights, self.lateral_weights in place
        """

        # update cumulative activation per node, used for adaptive learning-rate scaling
        self._update_activation_counters()

        # Get weight deltas using modified Oja rule
        input_value = self._get_input_values(input_value)

        # Add a little random noise to prevent gradients from dying in
        # thesholded activation functions
        if self.params['noise_var'] > 0:
            self.activation += self.params['noise_var'] * np.random.randn(self.num_nodes,)

        _lr = np.maximum(1.0 / self._cum_activation_l2, self.MINIMUM_LEARNING_RATE)
        lr_scaled_activation = self.activation * _lr

        delta_input = np.outer(
            input_value - self.input_weights.dot(self.activation),
            lr_scaled_activation)

        delta_lateral = np.outer(
            self.activation - self.lateral_weights.dot(self.activation),
            lr_scaled_activation)

        # Apply deltas
        self.input_weights += delta_input
        self.lateral_weights -= delta_lateral

        # Enforce constraings on weights
        np.fill_diagonal(self.lateral_weights, 0.0)
        rescale_weights_in_place(self.input_weights)
        rescale_weights_in_place(self.lateral_weights)
