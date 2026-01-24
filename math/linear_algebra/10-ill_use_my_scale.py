#!/usr/bin/env python3
"""Module for calculating the shape of a NumPy array."""

import numpy as np


def np_shape(matrix):
    """
    Calculate the shape of a NumPy array.

    Args:
        matrix (numpy.ndarray): The input array.

    Returns:
        tuple: The shape as a tuple of integers.
    """
    return matrix.shape
