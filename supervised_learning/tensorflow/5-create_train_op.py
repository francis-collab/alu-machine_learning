#!/usr/bin/env python3
"""Create training operation."""

import tensorflow as tf


def create_train_op(loss, alpha):
    """Creates the training operation.

    Args:
        loss (tensor): Loss of the network.
        alpha (float): Learning rate.

    Returns:
        Operation: Training operation.
    """
    train_op = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return train_op

