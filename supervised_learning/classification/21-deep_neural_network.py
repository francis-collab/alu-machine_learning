#!/usr/bin/env python3
"""DeepNeuralNetwork Gradient Descent - Task 21"""

import numpy as np


class DeepNeuralNetwork:
    """Class that defines a deep neural network."""

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

        for i in range(1, self.__L + 1):
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

    def forward_prop(self, X):
        self.__cache['A0'] = X
        A = X
        for i in range(1, self.__L + 1):
            Z = np.matmul(self.__weights[f'W{i}'], A) + self.__weights[f'b{i}']
            A = 1 / (1 + np.exp(-Z))
            self.__cache[f'A{i}'] = A
        return A, self.__cache

    def gradient_descent(self, Y, cache, alpha=0.05):
        """One pass of gradient descent - One loop only."""
        m = Y.shape[1]
        L = self.__L
        A = cache[f'A{L}']
        dZ = A - Y

        for i in range(L, 0, -1):
            A_prev = cache[f'A{i-1}']
            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            self.__weights[f'W{i}'] -= alpha * dW
            self.__weights[f'b{i}'] -= alpha * db

            if i > 1:
                dZ = np.matmul(self.__weights[f'W{i}'].T, dZ) * A_prev * (1 - A_prev)
                