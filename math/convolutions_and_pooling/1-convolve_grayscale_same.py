#!/usr/bin/env python3
"""Same convolution on grayscale images."""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """Perform same convolution on grayscale images (pad with zeros).

    Args:
        images (np.ndarray): shape (m, h, w)
        kernel (np.ndarray): shape (kh, kw)

    Returns:
        np.ndarray: convolved images with same shape as input (m, h, w)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph = (kh - 1) // 2
    pw = (kw - 1) // 2

    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            region = padded[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
