#!/usr/bin/env python3
"""Variational Autoencoder"""

import tensorflow.keras as keras
import tensorflow.keras.backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder.

    Args:
        input_dims (int): Input dimensions.
        hidden_layers (list): Hidden layers sizes.
        latent_dims (int): Latent space dimensions.

    Returns:
        encoder, decoder, auto
    """
    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims, name='z_mean')(x)
    z_log_var = keras.layers.Dense(latent_dims, name='z_log_var')(x)

    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,), name='z')([z_mean, z_log_var])

    encoder = keras.Model(inputs, [z, z_mean, z_log_var], name="encoder")

    # Decoder
    latent_input = keras.Input(shape=(latent_dims,))
    x = latent_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_input, outputs, name="decoder")

    # Autoencoder
    auto_outputs = decoder(encoder(inputs)[0])
    auto = keras.Model(inputs, auto_outputs, name="autoencoder")
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
