#!/usr/bin/env python3
"""Create placeholders for neural network."""

import tensorflow as tf


def create_placeholders(nx, classes):
    """Creates placeholders for the neural network.

    Args:
        nx (int): Number of feature columns.
        classes (int): Number of classes in the classifier.

    Returns:
        x (tf.placeholder): Placeholder for input data.
        y (tf.placeholder): Placeholder for one-hot labels.
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")
    return x, y

