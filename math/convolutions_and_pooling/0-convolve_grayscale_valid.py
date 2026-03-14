#!/usr/bin/env python3
"""Valid convolution on grayscale images."""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Perform valid convolution on grayscale images.

    Args:
        images (np.ndarray): shape (m, h, w) - m images
        kernel (np.ndarray): shape (kh, kw) - convolution kernel

    Returns:
        np.ndarray: convolved images (m, h-kh+1, w-kw+1)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    out_h = h - kh + 1
    out_w = w - kw + 1

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = images[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
