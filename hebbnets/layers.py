"""
    Class contains a pure-python/Numpy implementation for the layer of a feedforward
    network that learns using Hebbian/Anti-Hebbian (HAH) weight updates
"""
import abc
import numpy as np

from hebbnets.hebbnets import utils


class Layer(abc.ABC):
    """Abstract base class for a layer"""

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
    def _relu(activation):
        return np.maximum(activation, 0,)

    @staticmethod
    def _sigmoid(activation):
        return 1.0 / (1 + np.exp(-activation))

    @staticmethod
    def _linear(activation):
        return activation

    def _soft_thresh(self, activation):
        abs_thresh = np.maximum(np.abs(activation) - self._soft_thresh_val, 0.0)
        return np.sign(activation) * abs_thresh

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
            reg_lambda: Strength of soft-threshold regularization parameter
            noise_var: variance for guassian noise applied to all activations

        Returns:
            None
        """
        act_type = kwargs.get('act_type', 'linear')
        has_bias = kwargs.get('has_bias', False)
        reg_lambda = kwargs.get('reg_lambda', 0.0)
        noise_var = kwargs.get('noise_var', 0.0)

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

        _activation_func_names = {
            'relu': self._relu,
            'sigmoid': self._sigmoid,
            'linear': self._linear,
            'soft_thresh': self._soft_thresh
        }

        if act_type not in _activation_func_names:
            raise NameError("Activation function named %s not defined", act_type)
        self.activation_func = _activation_func_names[act_type]

        self.params = {
            'reg_lambda': reg_lambda,
            'bias': has_bias,
            'act_type': act_type,
            'noise_var': noise_var
        }

        self.activation = np.zeros((self.num_nodes, 1))
        self._soft_thresh_val = np.tile(1.0, (num_nodes, 1))



class HahLayer(Layer):
    """Layer of neurons that are activated and updated using a 
    Hebbian/AntHebbian (HAH) pattern
    """

    MAX_ACTIVATION_TIME_STEPS = 200
    ACTIVATION_ERROR_TOL = 1.0e-3

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
            reg_lambda: Strength of soft-threshold regularization parameter
            noise_var: variance for guassian noise applied to all activations

        Returns:
            None
        """

        super().__init__(num_nodes, prev_layer, **kwargs)

        self.input_weights = np.random.randn(
            self.layer_input_size + self.params['bias'],
            self.num_nodes,
        ).astype('float64')
        self.input_weights /= np.sqrt(np.prod(self.input_weights.shape))

        self.lateral_weights = np.random.randn(
            self.num_nodes,
            self.num_nodes,
        ).astype('float64')
        np.fill_diagonal(self.lateral_weights, 0.0)
        self.lateral_weights /= np.sqrt(np.prod(self.lateral_weights.shape))

        self._cum_sqr_activation = np.tile(1000.0, (num_nodes, 1))
        self._cum_abs_activation = np.tile(1000.0, (num_nodes, 1))


    def update_activation(self, input_value=None):
        """Update activation value for Hebb/Antihebb layer
        Args:
            input_value: set to list/numpy array being fed as input into this
               layer, otherwise input will be inferred from a previous layer
        Returns:
            None. Upates self.activation in place
        """
        input_value = self._get_input_values(input_value)

        input_value_times_weights = self.input_weights.T.dot(input_value)
        self.activation = np.zeros((self.num_nodes, 1), dtype='float64')

        for i in range(self.MAX_ACTIVATION_TIME_STEPS):

            _next_activation = self.activation_func(
                input_value_times_weights - self.lateral_weights.T.dot(self.activation)
            )
            utils.rescale_absmax_in_place(_next_activation, absmax_limit=100.0)

            error = utils.max_abs_reldiff(self.activation, _next_activation)
            self.activation = _next_activation

            if error < self.ACTIVATION_ERROR_TOL:
                break


    def _update_cums(self):
        self._cum_abs_activation = np.nan_to_num(self._cum_abs_activation)
        self._cum_sqr_activation = np.nan_to_num(self._cum_sqr_activation)
        self._soft_thresh_val = np.nan_to_num(self._soft_thresh_val)

        self._cum_abs_activation += self.CUMSCORE_LR * np.abs(self.activation)
        self._cum_sqr_activation += self.CUMSCORE_LR * self.activation ** 2
        self._soft_thresh_val = (
            0.5 * self.params.get('reg_lambda', 1.0) * 
            self._cum_abs_activation / self._cum_sqr_activation
        )

    def update_weights(self, input_value=None):
        """Update input weights with Hebbian/Antihebbian rule
        Args:
            input_value: set to list/numpy array being fed as input into this
               layer, otherwise input will be inferred from a previous layer
        Returns:
            None. Upates self.input_weights, self.lateral_weights in place
        """

        # Get weight deltas using modified Oja rule
        input_value = self._get_input_values(input_value)

        # Add a little random noise to prevent gradients from dying in
        # thesholded activation functions
        if self.params.get('noise_var', 0) > 0:
            self.activation += self.params['noise_var'] * np.random.randn(*self.activation.shape).astype('float64')

        # update cumulative activation per node, used for adaptive learning-rate scaling
        self._update_cums()

        # Calculate update deltas
        activation_norm = self.activation / self._cum_sqr_activation
        activation_sqr_norm = self.activation * activation_norm

        self.input_weights += (
            input_value.dot(activation_norm.T) -
            self.input_weights * activation_sqr_norm.T
        )

        self.lateral_weights += (
            self.activation.dot(activation_norm.T) -
            self.lateral_weights * activation_sqr_norm.T
        )

        # Enforce constraints on weights
        np.fill_diagonal(self.lateral_weights, 0.0)
        utils.rescale_absmax_in_place(self.input_weights, absmax_limit=10.0)
        utils.rescale_absmax_in_place(self.lateral_weights, absmax_limit=10.0)