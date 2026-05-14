#!/usr/bin/env python3
"""DeepNeuralNetwork - Task 16"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        """Initialize DeepNeuralNetwork."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(1, self.L + 1):   # Only one loop allowed
            prev = nx if i == 1 else layers[i-2]
            self.weights[f'W{i}'] = np.random.randn(layers[i-1], prev) * np.sqrt(2 / prev)
            self.weights[f'b{i}'] = np.zeros((layers[i-1], 1))
            