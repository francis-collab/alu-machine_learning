#!/usr/bin/env python3
"""Sparse Autoencoder"""

import tensorflow.keras as keras
from tensorflow.keras import regularizers


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Creates a sparse autoencoder.

    Args:
        input_dims (int): Dimensions of the input.
        hidden_layers (list): Number of nodes in hidden layers.
        latent_dims (int): Dimensions of latent space.
        lambtha (float): Regularization parameter for L1.

    Returns:
        encoder, decoder, auto (keras models)
    """
    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    latent = keras.layers.Dense(latent_dims, activation='relu',
                                activity_regularizer=regularizers.l1(lambtha))(x)
    encoder = keras.Model(inputs, latent, name="encoder")

    # Decoder
    latent_input = keras.Input(shape=(latent_dims,))
    x = latent_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_input, outputs, name="decoder")

    # Autoencoder
    auto_outputs = decoder(encoder(inputs))
    auto = keras.Model(inputs, auto_outputs, name="autoencoder")
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
