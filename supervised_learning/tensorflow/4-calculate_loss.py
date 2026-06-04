#!/usr/bin/env python3
"""Calculate loss."""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """Calculates the softmax cross-entropy loss of a prediction.

    Args:
        y: placeholder for the one-hot labels.
        y_pred: tensor containing the network’s predictions (logits).

    Returns:
        tensor containing the loss of the prediction.
    """
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=y, logits=y_pred))
    return loss

