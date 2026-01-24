#!/usr/bin/env python3
"""Module for calculating the shape of a matrix."""


def matrix_shape(matrix):
    """
    Calculate the shape of a matrix.

    Args:
        matrix (list): The input matrix as a nested list.

    Returns:
        list: A list of integers representing the shape.
    """
    if not isinstance(matrix, list):
        return []

    return [len(matrix)] + matrix_shape(matrix[0])
