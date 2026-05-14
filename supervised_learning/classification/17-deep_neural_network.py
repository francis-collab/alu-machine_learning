#!/usr/bin/env python3
"""DeepNeuralNetwork - Task 17 (Private)"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network."""

    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(1, self.__L + 1):   # Only one loop
            prev = nx if i == 1 else layers[i-2]
            self.__weights[f'W{i}'] = np.random.randn(layers[i-1], prev) * np.sqrt(2 / prev)
            self.__weights[f'b{i}'] = np.zeros((layers[i-1], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights
    