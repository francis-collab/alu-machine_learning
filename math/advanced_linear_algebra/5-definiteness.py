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

    if matrix.shape[0] == 0:
        return None

    eigenvalues = np.linalg.eigvals(matrix)

    if not np.all(np.isreal(eigenvalues)):
        return None

    eigenvalues = np.real(eigenvalues)

    if np.all(eigenvalues > 0):
        return "Positive definite"
    if np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    if np.all(eigenvalues < 0):
        return "Negative definite"
    if np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    if np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Indefinite"

    return None
