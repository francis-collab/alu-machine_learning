#!/usr/bin/env python3
"""Convolution on grayscale images with padding & stride."""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Perform convolution on grayscale images (valid/same/custom padding).

    Args:
        images (np.ndarray): shape (m, h, w)
        kernel (np.ndarray): shape (kh, kw)
        padding (str or tuple): 'valid', 'same', or (ph, pw)
        stride (tuple): (sh, sw)

    Returns:
        np.ndarray: convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'valid':
        ph = pw = 0
    elif padding == 'same':
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
    else:
        ph, pw = padding

    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    # Integer arithmetic equivalent to floor division + 1
    out_h = ((h + 2 * ph - kh) // sh) + 1
    out_w = ((w + 2 * pw - kw) // sw) + 1

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            row_start = i * sh
            col_start = j * sw
            region = padded[:, row_start:row_start+kh, col_start:col_start+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
