#!/usr/bin/env python3
"""Module for performing matrix multiplication with NumPy."""

import numpy as np


def np_matmul(mat1, mat2):
    """
    Perform matrix multiplication with NumPy.

    Args:
        mat1 (numpy.ndarray): The first matrix.
        mat2 (numpy.ndarray): The second matrix.

    Returns:
        numpy.ndarray: The product matrix.
    """
    return np.matmul(mat1, mat2)
