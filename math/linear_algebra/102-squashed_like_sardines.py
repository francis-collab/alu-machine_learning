#!/usr/bin/env python3
"""Recursive matrix concatenation along any axis."""


def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenate two n-dimensional matrices along the specified axis.

    Args:
        mat1: first matrix (nested lists)
        mat2: second matrix (nested lists)
        axis: dimension along which to concatenate (default 0)

    Returns:
        New concatenated matrix or None if incompatible
    """
    if axis == 0:
        if not isinstance(mat1, list) or not isinstance(mat2, list):
            return None
        if len(mat1) == 0 or len(mat2) == 0:
            return None
        # Check that sub-dimensions match
        if not same_shape(mat1[0], mat2[0]):
            return None
        return mat1 + mat2

    # axis > 0 → both must be lists and same length
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None
    if len(mat1) != len(mat2):
        return None

    result = []
    for a, b in zip(mat1, mat2):
        concatenated = cat_matrices(a, b, axis - 1)
        if concatenated is None:
            return None
        result.append(concatenated)

    return result


def same_shape(a, b):
    """Helper: check if two nested structures have the same shape"""
    if type(a) is not type(b):
        return False
    if not isinstance(a, list):
        return True
    if len(a) != len(b):
        return False
    return all(same_shape(x, y) for x, y in zip(a, b))
