#!/usr/bin/env python3
"""Module for calculating the minor matrix of a matrix."""


def minor(matrix):
    """Calculates the minor matrix of a matrix.

    Args:
        matrix (list of list): The matrix to calculate the minor for.

    Returns:
        list of list: The minor matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    if n == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    def det_helper(m):
        """Helper function to compute determinant recursively, handles empty for submatrices."""
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

    min_mat = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            sub = [r[:j] + r[j+1:] for r in matrix[:i] + matrix[i+1:]]
            min_mat[i][j] = det_helper(sub)
    return min_mat
