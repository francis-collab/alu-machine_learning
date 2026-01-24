#!/usr/bin/env python3
"""Module for element-wise operations on NumPy arrays."""

import numpy as np


def np_elementwise(mat1, mat2):
    """
    Perform element-wise addition, subtraction, multiplication, and division.

    Args:
        mat1 (numpy.ndarray): The first array.
        mat2 (numpy.ndarray or scalar): The second array or scalar.

    Returns:
        tuple: (sum, difference, product, quotient).
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
