#!/usr/bin/env python3
"""Normal distribution module"""


class Normal:
    """Represents a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize Normal distribution

        Args:
            data (list, optional): list of data to estimate mean/stddev
            mean (float): mean of the distribution
            stddev (float): standard deviation of the distribution

        Raises:
            ValueError: stddev <= 0 when data is None
            TypeError: data is not a list
            ValueError: data has fewer than 2 values
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            n = len(data)
            self.mean = sum(data) / n
            var = sum((x - self.mean) ** 2 for x in data) / n
            self.stddev = var ** 0.5

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return self.mean + z * self.stddev

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value

        Args:
            x (float): x-value

        Returns:
            float: PDF value for x
        """
        e = 2.718281828285
        pi = 3.141592653589793
        exp = -((x - self.mean) ** 2) / (2 * self.stddev ** 2)
        denom = self.stddev * (2 * pi) ** 0.5
        return (1 / denom) * (e ** exp)

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value

        Args:
            x (float): x-value

        Returns:
            float: CDF value for x
        """
        pi = 3.141592653589793
        sqrt_pi = pi ** 0.5
        z = (x - self.mean) / (self.stddev * 2 ** 0.5)

        # erf approximation using given series
        erf = (2 / sqrt_pi) * (z - (z**3)/3 + (z**5)/10 - (z**7)/42 + (z**9)/216)

        return 0.5 * (1 + erf)