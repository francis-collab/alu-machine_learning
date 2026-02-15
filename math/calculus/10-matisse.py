#!/usr/bin/env python3
"""Polynomial derivative calculator."""


def poly_derivative(poly):
    """Compute coefficients of the derivative of a polynomial.

    The input list poly represents coefficients where poly[i] is the
    coefficient of x^i.

    Args:
        poly (list): list of int or float coefficients

    Returns:
        list or None: coefficients of the derivative
                      [0] if result is the zero polynomial
                      None if input is invalid
    """
    if type(poly) is not list or len(poly) == 0:
        return None

    if not all(isinstance(c, (int, float)) for c in poly):
        return None

    if len(poly) <= 1:
        return [0]

    derivative = []
    for power in range(1, len(poly)):
        derivative.append(poly[power] * power)

    # remove trailing zeros but keep at least one element
    while len(derivative) > 1 and derivative[-1] == 0:
        derivative.pop()

    return derivative