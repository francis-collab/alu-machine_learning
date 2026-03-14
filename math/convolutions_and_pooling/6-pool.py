#!/usr/bin/env python3
"""Pooling operation on images."""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Perform pooling (max or average) on images.

    Args:
        images (np.ndarray): shape (m, h, w, c)
        kernel_shape (tuple): (kh, kw)
        stride (tuple): (sh, sw)
        mode (str): 'max' or 'avg'

    Returns:
        np.ndarray: pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1

    output = np.zeros((m, out_h, out_w, c))

    for i in range(out_h):
        for j in range(out_w):
            row_start = i * sh
            col_start = j * sw
            region = images[:, row_start:row_start+kh, col_start:col_start+kw, :]
            if mode == 'max':
                output[:, i, j, :] = np.max(region, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(region, axis=(1, 2))

    return output
