#!/usr/bin/env python3
"""
This module defines the MultiNormal class that represents
a Multivariate Normal distribution.
"""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution.

    Public instance variables:
        mean (numpy.ndarray): shape (d, 1) containing the mean
        cov (numpy.ndarray): shape (d, d) containing the covariance matrix
    """

    def __init__(self, data):
        """Initializes a MultiNormal instance.

        Args:
            data (numpy.ndarray): shape (d, n) containing the data set
                d: number of dimensions
                n: number of data points

        Raises:
            TypeError: if data is not a 2D numpy.ndarray
            ValueError: if n < 2
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        centered = data - self.mean
        self.cov = np.dot(centered, centered.T) / (n - 1)

    def pdf(self, x):
        """Calculates the PDF of the multivariate normal distribution at a point x.

        Args:
            x (numpy.ndarray): shape (d, 1) containing the data point

        Returns:
            float: value of the PDF

        Raises:
            TypeError: if x is not a numpy.ndarray
            ValueError: if x is not of shape (d, 1)
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        diff = x - self.mean
        inv_cov = np.linalg.inv(self.cov)
        mahalanobis = np.dot(diff.T, np.dot(inv_cov, diff))[0, 0]
        det_cov = np.linalg.det(self.cov)
        norm_factor = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(det_cov))
        pdf_value = norm_factor * np.exp(-0.5 * mahalanobis)
        return pdf_value