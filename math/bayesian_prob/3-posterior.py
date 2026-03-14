#!/usr/bin/env python3
"""Posterior probability distribution."""

import numpy as np
from scipy.special import comb


def posterior(x, n, P, Pr):
    """Compute posterior P(p | x,n) for each p in P.

    Args:
        x (int): number of successes
        n (int): number of trials
        P (np.ndarray): array of hypothetical probabilities
        Pr (np.ndarray): prior probabilities (same shape as P)

    Returns:
        np.ndarray: posterior probabilities
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer >= 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with same shape as P")

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in [0, 1]")

    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in [0, 1]")

    if not np.isclose(np.sum(Pr), 1.0):
        raise ValueError("Pr must sum to 1")

    coeff = comb(n, x)
    lik = coeff * (P ** x) * ((1 - P) ** (n - x))
    inter = lik * Pr
    marg = np.sum(inter)

    if marg == 0:
        return np.zeros_like(inter)

    return inter / marg
