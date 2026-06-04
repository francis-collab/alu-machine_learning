#!/usr/bin/env python3
"""Forward propagation."""

import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Creates the forward propagation graph.

    Args:
        x (placeholder): Input data.
        layer_sizes (list): Number of nodes per layer.
        activations (list): Activation functions per layer.

    Returns:
        tensor: Prediction of the network.
    """
    A = x
    for i in range(len(layer_sizes)):
        A = create_layer(A, layer_sizes[i], activations[i])
    return A
