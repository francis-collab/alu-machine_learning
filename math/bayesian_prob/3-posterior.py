#!/usr/bin/env python3
"""Posterior probability distribution."""

import numpy as np


def posterior(x, n, P, Pr):
    """
    Calculate the posterior probability distribution P(p | data)
    for each value in P given the observed data and prior.

    Parameters:
    -----------
    x, n, P, Pr : same as above

    Returns:
    --------
    numpy.ndarray
        1D array of posterior probabilities (normalized)
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
    marg = np.sum(inter)

    # Avoid division by zero (though marginal should be >0 in valid cases)
    post = inter / marg if marg > 0 else np.zeros_like(inter)

    return post
