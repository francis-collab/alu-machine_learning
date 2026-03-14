#!/usr/bin/env python3
"""Convolution with custom padding on grayscale images."""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Perform convolution on grayscale images with given padding.

    Args:
        images (np.ndarray): shape (m, h, w)
        kernel (np.ndarray): shape (kh, kw)
        padding (tuple): (ph, pw) - padding height and width

    Returns:
        np.ndarray: convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    out_h = h + 2 * ph - kh + 1
    out_w = w + 2 * pw - kw + 1

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = padded[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
