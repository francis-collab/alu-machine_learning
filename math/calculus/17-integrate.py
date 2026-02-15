#!/usr/bin/env python3
"""Polynomial indefinite integral calculator."""


def poly_integral(poly, C=0):
    """Compute coefficients of the indefinite integral of a polynomial + constant C.

    poly[i] is coefficient of x^i.
    Result includes the constant term first (C), then coefficients of x^1, x^2, ...

    Args:
        poly (list): list of int/float coefficients
        C (int): integration constant (must be integer)

    Returns:
        list or None: coefficients of antiderivative (compact form)
                      integers used when possible
                      None if input invalid
    """
    if type(poly) is not list or len(poly) == 0:
        return None

    if type(C) is not int:
        return None

    if not all(isinstance(coef, (int, float)) for coef in poly):
        return None

    result = [C]

    for power in range(len(poly)):
        coef = poly[power]
        if coef == 0:
            result.append(0)
        else:
            new_coef = coef / (power + 1)
            if new_coef == int(new_coef):
                result.append(int(new_coef))
            else:
                result.append(new_coef)

    # Remove trailing zeros (keep at least the constant term)
    while len(result) > 1 and result[-1] == 0:
        result.pop()

    return result