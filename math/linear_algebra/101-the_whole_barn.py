#!/usr/bin/env python3
"""Recursive matrix addition — any number of dimensions."""


def add_matrices(mat1, mat2):
    """
    Add two n-dimensional matrices element-wise.

    Args:
        mat1: first matrix (list or nested lists)
        mat2: second matrix (list or nested lists)

    Returns:
        New matrix with same structure or None if shapes incompatible
    """
    if type(mat1) is not type(mat2):
        return None

    if isinstance(mat1, (int, float)):
        return mat1 + mat2

    if len(mat1) != len(mat2):
        return None

    result = []
    for a, b in zip(mat1, mat2):
        added = add_matrices(a, b)
        if added is None:
            return None
        result.append(added)

    return result
