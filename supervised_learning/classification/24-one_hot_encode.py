#!/usr/bin/env python3
"""One-hot encode function."""

import numpy as np


def one_hot_encode(Y, classes):
    """Converts numeric label vector into one-hot matrix."""
    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or classes <= 0:
        return None

    m = Y.shape[0]
    one_hot = np.zeros((classes, m))
    one_hot[Y, np.arange(m)] = 1
    return one_hot
