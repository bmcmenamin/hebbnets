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
