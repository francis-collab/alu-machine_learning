#!/usr/bin/env python3
"""Summation of squares from 1 to n without using loops."""


def summation_i_squared(n):
    """Calculate the sum of squares from 1 to n using closed form formula.

    Args:
        n (int): positive integer upper bound

    Returns:
        int or None: sum of i² if n is valid positive integer, else None
    """
    if type(n) is not int or n < 1:
        return None

    return n * (n + 1) * (2 * n + 1) // 6