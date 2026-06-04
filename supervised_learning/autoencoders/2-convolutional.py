#!/usr/bin/env python3
"""Convolutional Autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Creates a convolutional autoencoder.

    Args:
        input_dims (tuple): Input dimensions (height, width, channels).
        filters (list): Number of filters for each conv layer.
        latent_dims (tuple): Latent space dimensions.

    Returns:
        encoder, decoder, auto
    """
    # Encoder
    inputs = keras.Input(shape=input_dims)
    x = inputs
    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    encoder = keras.Model(inputs, x, name="encoder")

    # Decoder
    latent_input = keras.Input(shape=latent_dims)
    x = latent_input
    for f in reversed(filters[:-1]):
        x = keras.layers.Conv2D(f, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

    # Second to last convolution with valid padding
    x = keras.layers.Conv2D(filters[-1], (3, 3), activation='relu', padding='valid')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)

    # Final output
    outputs = keras.layers.Conv2D(input_dims[-1], (3, 3), activation='sigmoid', padding='same')(x)

    decoder = keras.Model(latent_input, outputs, name="decoder")

    # Autoencoder
    auto_outputs = decoder(encoder(inputs))
    auto = keras.Model(inputs, auto_outputs, name="autoencoder")
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
