#!/usr/bin/env python3
"""Module for concatenating NumPy arrays along a specific axis."""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenate two NumPy arrays along a specific axis.

    Args:
        mat1 (numpy.ndarray): The first array.
        mat2 (numpy.ndarray): The second array.
        axis (int): The axis to concatenate along.

    Returns:
        numpy.ndarray: The concatenated array.
    """
    return np.concatenate((mat1, mat2), axis=axis)
