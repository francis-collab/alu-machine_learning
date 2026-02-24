#!/usr/bin/env python3
"""
This module contains the mean_cov function that calculates
the mean and covariance matrix of a given data set.
"""
import numpy as np


def mean_cov(X):
    """Calculates the mean and covariance of a data set.

    Args:
        X (numpy.ndarray): shape (n, d) containing the data set
            n: number of data points
            d: number of dimensions

    Returns:
        mean (numpy.ndarray): shape (1, d) containing the mean
        cov (numpy.ndarray): shape (d, d) containing the covariance matrix

    Raises:
        TypeError: if X is not a 2D numpy.ndarray
        ValueError: if n < 2
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - mean
    cov = np.dot(X_centered.T, X_centered) / (n - 1)
    return mean, cov