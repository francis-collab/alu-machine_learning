#!/usr/bin/env python3
"""Binomial distribution module"""


class Binomial:
    """Represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize Binomial distribution

        Args:
            data (list, optional): list of data to estimate n and p
            n (int): number of trials
            p (float): probability of success

        Raises:
            ValueError: n <= 0 or p not in (0,1) when data is None
            TypeError: data is not a list
            ValueError: data has fewer than 2 values
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)

            # Count number of successes (values == 1)
            successes = sum(1 for val in data if val == 1)
            p_est = successes / len(data)

            if p_est == 0:
                # Edge case: no successes → avoid division by zero
                # Use very small p and estimate n from mean
                p_est = 1e-9          # very small positive value
                n_est = round(mean / p_est) if mean > 0 else 1
            else:
                n_est = round(mean / p_est)

            # Recalculate p using estimated n
            p_est = mean / n_est if n_est > 0 else 0.5

            # Make sure p stays in (0,1)
            if not (0 < p_est < 1):
                p_est = 0.5

            self.n = int(round(n_est))
            self.p = float(p_est)

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
        if k < 0 or k > self.n:
            return 0.0

        # Binomial coefficient C(n,k) computed iteratively
        c = 1.0
        for i in range(k):
            c *= (self.n - i) / (i + 1)

        return c * (self.p ** k) * ((1 - self.p) ** (self.n - k))

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
        if k >= self.n:
            return 1.0

        cum = 0.0
        for i in range(k + 1):
            cum += self.pmf(i)
        return cum