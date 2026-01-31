#!/usr/bin/env python3
"""Module for calculating the definiteness of a matrix."""

import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a matrix.

    Args:
        matrix (numpy.ndarray): The matrix of shape (n, n) to analyze.

    Returns:
        str: The definiteness string or None.

    Raises:
        TypeError: If matrix is not a numpy.ndarray.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    n = matrix.shape[0]
    if n == 0:
        return None

    eigs = np.linalg.eigvals(matrix)
    if not np.all(np.isreal(eigs)):
        return None
    eigs = np.real(eigs)

    if np.all(eigs > 0):
        return "Positive definite"
    elif np.all(eigs >= 0):
        return "Positive semi-definite"
    elif np.all(eigs < 0):
        return "Negative definite"
    elif np.all(eigs <= 0):
        return "Negative semi-definite"
    elif np.any(eigs > 0) and np.any(eigs < 0):
        return "Indefinite"
    else:
        return None
