#!/usr/bin/env python3
"""Module for concatenating two 2D matrices along a specific axis."""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenate two 2D matrices along a specific axis.

    Args:
        mat1 (list): The first matrix.
        mat2 (list): The second matrix.
        axis (int): The axis to concatenate along (0 or 1).

    Returns:
        list or None: The concatenated matrix or None if incompatible.
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [r1[:] + r2[:] for r1, r2 in zip(mat1, mat2)]

    return None
