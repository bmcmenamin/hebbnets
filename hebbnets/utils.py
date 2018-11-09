"""
    Utility functions
"""

import numpy as np


def rescale_absmax_in_place(data_matrix, absmax_limit=10.0):
    """If any element of an input matrix exceeds the value
    absmax_limit rescale the whole input matrix so it doesn't

    Args:
        data_matrix: numpy array to rescale
        absmax_limit: absmax limit

    Returns:
        None, works in place
    """

    data_matrix = np.nan_to_num(data_matrix)

    absmax_val = max(data_matrix.max(), -data_matrix.min())
    if absmax_val > absmax_limit:
        scale = absmax_limit / absmax_val
        data_matrix *= scale


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
