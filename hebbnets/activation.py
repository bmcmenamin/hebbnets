"""
    Class contains activation funtions
"""

import abc
import numpy as np


class BaseActivationFunction(abc.ABC):
    """Base class for neural activation function
    """

    @abc.abstractmethod
    def apply(self, raw_activation):
        """Apply activation function to raw inputs"""
        return NotImplemented

    @abc.abstractmethod
    def apply_deriv(self, raw_activation):
        """Apply the derivative activation function to raw inputs"""
        return NotImplemented


class LinearActivation(BaseActivationFunction):
    """Linear activation function"""

    @staticmethod
    def apply(raw_activation):
        return raw_activation

    @staticmethod
    def apply_deriv(raw_activation):
        return np.ones(raw_activation.shape)


class SigmoidActivation(BaseActivationFunction):
    """Sigmoid activation function"""

    @staticmethod
    def _sigmoid(x_value):
        return 1.0 / (1 + np.exp(-x_value))

    @classmethod
    def apply(cls, raw_activation):
        """Apply activation function to raw inputs"""
        return cls._sigmoid(raw_activation)

    @classmethod
    def apply_deriv(cls, raw_activation):
        """Apply the derivative activation function to raw inputs"""
        sig = cls._sigmoid(raw_activation)
        return sig * (1 - sig)


class ReLUActivation(BaseActivationFunction):
    """ReLU activation function"""

    @staticmethod
    def apply(raw_activation):
        """Apply activation function to raw inputs"""
        return np.maximum(raw_activation, 0,)

    @staticmethod
    def apply_deriv(raw_activation):
        """Apply the derivative activation function to raw inputs"""
        return (raw_activation > 0).astype(float)
