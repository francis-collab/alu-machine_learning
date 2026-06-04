#!/usr/bin/env python3
"""Create a layer in the neural network."""

import tensorflow as tf


def create_layer(prev, n, activation):
    """Creates a layer in the neural network.

    Args:
        prev (tensor): Previous layer output.
        n (int): Number of nodes.
        activation (function): Activation function.

    Returns:
        tensor: Output of the layer.
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=initializer,
                            name="layer")
    return layer(prev)
