#!/usr/bin/env python3
"""Convolution with multiple kernels."""

import numpy as np
from math import ceil, floor


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Perform convolution using multiple kernels (produces feature maps).

    Args:
        images (np.ndarray): shape (m, h, w, c)
        kernels (np.ndarray): shape (kh, kw, c, nc)
        padding (str or tuple): 'valid', 'same', or (ph, pw)
        stride (tuple): (sh, sw)

    Returns:
        np.ndarray: convolved images (m, out_h, out_w, nc)
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    if padding == 'valid':
        ph = pw = 0
    elif padding == 'same':
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
    else:
        ph, pw = padding

    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, out_h, out_w, nc))

    for i in range(out_h):
        for j in range(out_w):
            row_start = i * sh
            col_start = j * sw
            region = padded[:, row_start:row_start+kh, col_start:col_start+kw, :]
            for k in range(nc):
                output[:, i, j, k] = np.sum(region * kernels[..., k], axis=(1, 2, 3))

    return output
