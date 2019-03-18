"""
    Class contains a pure-python/Numpy implementation for the layer of a feedforward
    network that learns using Hebbian/Anti-Hebbian (HAH) weight updates and a
    layer for Q-AQREL reinforcement learning
"""

import abc
import numpy as np


from hebbnets import activation


ACTIVATION_REGISTRY = {
    'linear': activation.LinearActivation,
    'sigmoid': activation.SigmoidActivation,
    'relu': activation.ReLUActivation
}


class BaseLayer(abc.ABC):
    """Abstract base class for a Network of layers"""

    MAX_WEIGHT = 1.0e6

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

    @staticmethod
    def glorot_init(in_dim, out_dim):
        scale = np.sqrt(6 / (in_dim + out_dim))
        uni_rand_vals = np.random.rand(in_dim, out_dim)
        scaled_vals = scale * 2 * (uni_rand_vals - 0.5)
        return scaled_vals.astype('float64')

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

        return input_value

    def _get_attr_from_prev_layer(self, attr_name):
        """Get attribute by name from previous layer"""
        if not isinstance(self.prev_layer, BaseLayer):
            raise TypeError(
                "Layer requires previous layer to be of type Layer. It's {}".
                format(type(self.prev_layer))
            )
        return getattr(self.prev_layer, attr_name)

    def _get_input_values(self, input_value, input_attr_name='activation'):
        """Get the input to this layer in the correct format
        """

        if input_value is None:
            input_value = self._get_attr_from_prev_layer(input_attr_name)
        else:
            input_value = self._get_input_value_format(input_value)

        if self.params['bias']:
            input_value = np.append(input_value, 1.0)

        return input_value.reshape(-1, 1)

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
        act_type = kwargs.get('act_type', 'linear')
        has_bias = kwargs.get('has_bias', False)
        noise_var = kwargs.get('noise_var', 0.0)

        self.num_nodes = num_nodes
        self.prev_layer = prev_layer

        if isinstance(prev_layer, int):
            self.layer_input_size = prev_layer
        elif isinstance(prev_layer, BaseLayer):
            self.layer_input_size = prev_layer.num_nodes
        else:
            raise TypeError(
                "prev_layer is of type {}, expecting BaseLayer or int".
                format(type(prev_layer))
            )

        if act_type not in ACTIVATION_REGISTRY:
            raise NameError("Activation function named %s not defined", act_type)
        self.activation_type = ACTIVATION_REGISTRY[act_type]()

        self.params = {
            'bias': has_bias,
            'act_type': act_type,
            'noise_var': noise_var
        }

        self.activation = np.zeros((self.num_nodes, 1))
