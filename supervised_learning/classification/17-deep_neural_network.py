#!/usr/bin/env python3
"""DeepNeuralNetwork with private attributes."""

import numpy as np


class DeepNeuralNetwork:
    """Class that defines a deep neural network."""

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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for l in range(1, self.__L + 1):
            layer_size = layers[l-1]
            prev_size = nx if l == 1 else layers[l-2]
            self.__weights[f'W{l}'] = np.random.randn(layer_size, prev_size) * np.sqrt(2 / prev_size)
            self.__weights[f'b{l}'] = np.zeros((layer_size, 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights
    