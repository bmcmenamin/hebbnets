"""
    Class contains a pure-python/Numpy implementation for the layer of a feedforward
    network that learns using Hebbian/Anti-Hebbian (HAH) weight updates
"""

import numpy as np

ABSMAX_WEIGHT_LIMIT = 1.0e4
SOFT_THRESH = 0.1

# Neural activation functions 
ACTIVATION_FUNCS = {
    'relu': lambda x: np.maximum(x, 0, x),
    'sigmoid': lambda x: 1.0 / (1 + np.exp(-x)),
    'linear': lambda x: x,
    'soft_thresh': lambda x: np.sign(x) * np.maximum(np.abs(x) - SOFT_THRESH, 0.0, x)
}




def _rescale_weights_in_place(weights):
    """Rescale every element of a wieght matrix in place if any
    of the values exceed the ABSMAX_WEIGHT_LIMIT
    """
    absmax_weight = max(weights.max(), -weights.min())
    if absmax_weight > ABSMAX_WEIGHT_LIMIT:
        weights /= (absmax_weight / ABSMAX_WEIGHT_LIMIT)


class HahLayer(object):
    """Layer of neurons that are activated and updated using a 
    Hebbian/AntHebbian (HAH) pattern
    """
    ACTIVATION_TIME_STEPS = 50
    INITIAL_CUM_L2 = 1.0
    MINIMUM_LEARNING_RATE = 1.0e-8
    _MAX_CUM_L2 = 1.0 / MINIMUM_LEARNING_RATE

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

    def __init__(self, num_nodes, prev_layer, act_type='linear', has_bias=True, gamma=0.0):
        """Set layer activation based on previous layer
        Args:
            num_nodes: Number of nodes in this layer
            prev_layer: The previous layer of the network
                If this is an input layer, set to with integer for input
                size
            act_type: String naming the type of activation function
                to use
            has_bias: Bool indicating whether layer has bias
            gamma: Strength of within-layer decorrelation parameter

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
        self.input_weights /= 0.1 * np.linalg.norm(
            self.input_weights,
            axis=1, keepdims=True)

        self.lateral_weights = np.random.randn(
            self.num_nodes,
            self.num_nodes
        )
        np.fill_diagonal(self.lateral_weights, 0.0)
        self.lateral_weights /= 0.1 * np.linalg.norm(
            self.lateral_weights,
            axis=1, keepdims=True)

        self.params = {
            'gamma': gamma,
            'bias': has_bias,
            'act_type': act_type
        }

        if act_type not in ACTIVATION_FUNCS:
            raise NameError("Activation function named %s not defined", act_type)

        self.activation = np.zeros(self.num_nodes)
        self._cum_activation_l2 = np.tile(self.INITIAL_CUM_L2, num_nodes)

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

            _next_step = ACTIVATION_FUNCS[self.params['act_type']](
                input_value.dot(self.input_weights) - self.activation.dot(self.lateral_weights)
            )

            self.activation = _next_step

            if np.allclose(_next_step, self.activation):
                break

    def update_weights(self, input_value=None):
        """Update input weights with Hebbian/Antihebbian rule
        Args:
            input_value: set to list/numpy array being fed as input into this
               layer, otherwise input will be inferred from a previous layer
        Returns:
            None. Upates self.input_weights, self.lateral_weights in place
        """

        # update cumulative activation per node, used for adaptive learning-rate scaling
        self._cum_activation_l2 += self.activation ** 2
        np.minimum(
            self._cum_activation_l2,
            self._MAX_CUM_L2,
            out=self._cum_activation_l2
        )

        # Get weight deltas using modified Oja rule
        input_value = self._get_input_values(input_value)
        lr_scaled_activation = self.activation / self._cum_activation_l2

        delta_input = np.outer(
            input_value - self.input_weights.dot(self.activation),
            lr_scaled_activation)

        gamma_times_ident = np.diag(np.tile(1 + self.params['gamma'], self.num_nodes))
        delta_lateral = np.outer(
            (gamma_times_ident - self.lateral_weights).dot(self.activation),
            lr_scaled_activation)

        # Apply deltas
        self.input_weights += delta_input
        self.lateral_weights += delta_lateral

        # Enforce constraings on weights
        np.fill_diagonal(self.lateral_weights, 0.0)
        _rescale_weights_in_place(self.input_weights)
        _rescale_weights_in_place(self.lateral_weights)
