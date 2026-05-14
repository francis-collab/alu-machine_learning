#!/usr/bin/env python3
"""DeepNeuralNetwork for multiclass classification."""

import numpy as np
import pickle


class DeepNeuralNetwork:
    """Class that defines a deep neural network performing multiclass classification."""

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

    def forward_prop(self, X):
        self.__cache['A0'] = X
        A = X
        for l in range(1, self.__L + 1):
            Z = np.matmul(self.__weights[f'W{l}'], A) + self.__weights[f'b{l}']
            A = 1 / (1 + np.exp(-Z))
            self.__cache[f'A{l}'] = A
        return A, self.__cache

    def cost(self, Y, A):
        """Cost for multiclass (one-hot)."""
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(Y * np.log(A) + (1.0000001 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluate for multiclass."""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.argmax(A, axis=0).reshape(1, -1)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        m = Y.shape[1]
        L = self.__L
        A = cache[f'A{L}']
        dZ = A - Y

        for l in range(L, 0, -1):
            A_prev = cache[f'A{l-1}']
            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            self.__weights[f'W{l}'] -= alpha * dW
            self.__weights[f'b{l}'] -= alpha * db

            if l > 1:
                dZ = np.matmul(self.__weights[f'W{l}'].T, dZ) * A_prev * (1 - A_prev)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

            if i % step == 0 or i == iterations or i == 0:
                cost = self.cost(Y, A)
                costs.append(cost)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")

        if graph:
            import matplotlib.pyplot as plt
            plt.plot(range(0, iterations + 1, step), costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None