#!/usr/bin/env python3
"""Module for performing matrix multiplication."""


def mat_mul(mat1, mat2):
    """
    Perform matrix multiplication.

    Args:
        mat1 (list): The first matrix.
        mat2 (list): The second matrix.

    Returns:
        list or None: The product matrix or None if incompatible.
    """
    rows1 = len(mat1)
    cols1 = len(mat1[0])
    rows2 = len(mat2)
    cols2 = len(mat2[0])

    if cols1 != rows2:
        return None

    result = [[0] * cols2 for _ in range(rows1)]

    for i in range(rows1):
        for j in range(cols2):
            for k in range(cols1):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
