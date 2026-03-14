#!/usr/bin/env python3
"""Intersection of likelihood and prior (unnormalized posterior)."""

import numpy as np
from scipy.special import comb      # only used for clarity in docstring, not in computation


def intersection(x, n, P, Pr):
    """
    Calculate the intersection (likelihood × prior) of obtaining x successes
    in n trials for each hypothetical probability in P given prior Pr.

    Parameters:
    -----------
    x : int
        Number of patients that develop severe side effects
    n : int
        Total number of patients observed
    P : numpy.ndarray
        1D array of hypothetical probabilities
    Pr : numpy.ndarray
        1D array of prior probabilities (same shape as P)

    Returns:
    --------
    numpy.ndarray
        1D array containing intersection values (unnormalized posterior)

    Raises:
    -------
    ValueError / TypeError
        Same checks as in likelihood() + prior checks
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any((P < 0) | (P > 1)) or np.any((Pr < 0) | (Pr > 1)):
        if np.any((P < 0) | (P > 1)):
            raise ValueError("All values in P must be in the range [0, 1]")
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(Pr.sum(), 1.0):
        raise ValueError("Pr must sum to 1")

    lik = (P ** x) * ((1 - P) ** (n - x))
    inter = lik * Pr

    return inter