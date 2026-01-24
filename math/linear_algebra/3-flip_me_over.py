#!/usr/bin/env python3

"""Module for transposing a 2D matrix."""

def matrix_transpose(matrix):
    """
    Transpose a 2D matrix.

    Args:
        matrix (list): The input 2D matrix.

    Returns:
        list: The transposed matrix.
    """
    # Use zip to transpose, convert tuples back to lists
    return [list(row) for row in zip(*matrix)]
