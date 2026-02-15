#!/usr/bin/env python3
"""Poisson distribution module"""


class Poisson:
    """Represents a Poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize Poisson distribution

        Args:
            data (list, optional): list of data to estimate lambtha
            lambtha (float): expected number of occurrences

        Raises:
            ValueError: lambtha <= 0 when data is None
            TypeError: data is not a list
            ValueError: data has fewer than 2 values
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes

        Args:
            k (int/float): number of successes

        Returns:
            float: PMF value for k
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0.0

        e = 2.718281828285
        lam = self.lambtha
        fact = 1.0
        for i in range(1, k + 1):
            fact *= i
        return (e ** (-lam) * (lam ** k)) / fact

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of successes

        Args:
            k (int/float): number of successes

        Returns:
            float: CDF value for k
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0.0

        e = 2.718281828285
        lam = self.lambtha
        cum = 0.0
        term = e ** (-lam)
        for i in range(k + 1):
            cum += term
            term *= lam / (i + 1)
        return cum