#!/usr/bin/env python3
"""Module for concatenating NumPy arrays along a specific axis."""


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
    if axis == 0:
        return np.array(list(mat1) + list(mat2))
    if mat1.ndim == 1:
        return mat1.reshape(1, -1).T + mat2.reshape(1, -1).T
    return np.array([a + b for a, b in zip(mat1, mat2)])
