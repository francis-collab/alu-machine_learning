#!/usr/bin/env python3
"""Exponential distribution module"""


class Exponential:
    """Represents an exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize Exponential distribution

        Args:
            data (list, optional): list of data to estimate lambtha
            lambtha (float): rate parameter

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
            self.lambtha = 1.0 / (sum(data) / len(data))

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period

        Args:
            x (float): time period

        Returns:
            float: PDF value for x
        """
        if x < 0:
            return 0.0
        e = 2.718281828285
        return self.lambtha * (e ** (-self.lambtha * x))

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period

        Args:
            x (float): time period

        Returns:
            float: CDF value for x
        """
        if x < 0:
            return 0.0
        e = 2.718281828285
        return 1.0 - (e ** (-self.lambtha * x))