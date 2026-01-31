#!/usr/bin/env python3
"""Module for calculating the determinant of a matrix."""


def determinant(matrix):
    """Calculates the determinant of a matrix.

    Args:
        matrix (list of list): The matrix to calculate the determinant for.

    Returns:
        int/float: The determinant of the matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square.
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Special case for 0x0 matrix [[]]
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    def det_helper(m):
        """Helper function to compute determinant recursively,
        handles empty for submatrices.
        """
        if len(m) == 0:
            return 1
        k = len(m)
        if k == 1:
            return m[0][0]
        det = 0
        sign = 1
        for j in range(k):
            sub = [row[:j] + row[j+1:] for row in m[1:]]
            det += sign * m[0][j] * det_helper(sub)
            sign = -sign
        return det

    return det_helper(matrix)
