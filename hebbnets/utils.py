"""
    Utility functions
"""

import numpy as np


def max_abs_reldiff(value0, value1, pad=0.001):
    """ Given two vectors, return the max absolute relative difference

    Args:
        value0, value1: two numpy vectors of the same size
        pad: optional value to add to denominator for numerical stability
    Returns:
        scalar of max absolute relative difference
    """
    diff = np.abs(value0 - value1)
    diff /= np.abs(value0 + pad)
    return diff.max()


def softmax(values, temp=0.01):
    """ Apply softmax to a vector of values

    Args:
        softmax: numpy vectors of values
        temp: temperature parameter
    Returns:
        scalar of max absolute relative difference
    """
    max_val = values.max()
    exp_val = np.exp(values + temp - max_val)
    return exp_val / exp_val.sum(axis=0)
