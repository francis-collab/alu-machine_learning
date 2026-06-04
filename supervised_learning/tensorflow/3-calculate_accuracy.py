#!/usr/bin/env python3
"""Calculate accuracy."""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Calculates the accuracy of a prediction.

    Args:
        y (placeholder): True labels.
        y_pred (tensor): Predicted labels.

    Returns:
        tensor: Accuracy.
    """
    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy
