#!/usr/bin/env python3
"""Returns the shape of a matrix as list of integers."""


def matrix_shape(matrix):
    """
    Calculate the shape of a matrix.

    Args:
        matrix: nested list representing a matrix

    Returns:
        list: shape dimensions (e.g. [2, 3] for 2×3 matrix)
    """
    shape = []
    current = matrix

    while isinstance(current, list):
        shape.append(len(current))
        if len(current) == 0:
            break
        current = current[0]

    return shape
