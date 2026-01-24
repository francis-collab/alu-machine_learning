#!/usr/bin/env python3
"""Module for adding two 2D matrices element-wise."""

matrix_shape = __import__('2-size_me_please').matrix_shape


def add_matrices2D(mat1, mat2):
    """
    Add two 2D matrices element-wise.

    Args:
        mat1 (list): The first matrix.
        mat2 (list): The second matrix.

    Returns:
        list or None: The sum matrix or None if shapes differ.
    """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    return [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(mat1, mat2)]
