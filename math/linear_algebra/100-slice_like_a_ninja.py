#!/usr/bin/env python3
"""Advanced slicing with dictionary of axes."""


def np_slice(matrix, axes={}):
    """
    Slice a NumPy array along specified axes using a dictionary.

    Args:
        matrix (numpy.ndarray): Input array to slice.
        axes (dict): {axis: (start, stop, step)} or partial slices.

    Returns:
        numpy.ndarray: New sliced array.
    """
    slices = [slice(None)] * matrix.ndim

    for axis, slc in axes.items():
        slices[axis] = slice(*slc)

    return matrix[tuple(slices)]
