#!/usr/bin/env python3
"""Likelihood function for binomial data."""

import numpy as np


def likelihood(x, n, P):
    """
    Calculate the likelihood of observing x successes in n trials
    for each hypothetical probability in P.

    Parameters:
    -----------
    x : int
        Number of patients that develop severe side effects
    n : int
        Total number of patients observed
    P : numpy.ndarray
        1D array of hypothetical probabilities

    Returns:
    --------
    numpy.ndarray
        1D array containing the likelihood for each p in P

    Raises:
    -------
    ValueError
        If n is not positive integer, x invalid, or x > n
    TypeError
        If P is not 1D numpy array
    ValueError
        If any value in P is outside [0,1]
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Binomial likelihood:  C(n,x) * p^x * (1-p)^(n-x)
    # We only need the part that depends on p (constant factor can be omitted for many purposes)
    lik = (P ** x) * ((1 - P) ** (n - x))

    return lik