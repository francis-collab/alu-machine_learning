# Unsupervised Learning - Autoencoders

This project implements different types of **Autoencoders** using **TensorFlow Keras**.

## Tasks Completed

| Task | File | Description |
|------|------|-----------|
| 0 | `0-vanilla.py` | **Vanilla Autoencoder** (fully connected) |
| 1 | `1-sparse.py` | **Sparse Autoencoder** (with L1 regularization) |
| 2 | `2-convolutional.py` | **Convolutional Autoencoder** |
| 3 | `3-variational.py` | **Variational Autoencoder (VAE)** |

## Requirements

- Python 3.5
- TensorFlow 1.12 (with Keras)
- NumPy
- Only allowed import: `import tensorflow.keras as keras`

## Files Description

### 0. Vanilla Autoencoder
- Standard fully connected autoencoder
- Uses ReLU activation (except sigmoid on final decoder layer)
- Compiled with Adam optimizer and binary cross-entropy loss

### 1. Sparse Autoencoder
- Adds **L1 regularization** on the latent space (`activity_regularizer`)
- Encourages sparsity in the encoded representation

### 2. Convolutional Autoencoder
- Uses Conv2D + MaxPooling2D in encoder
- Uses Conv2D + UpSampling2D in decoder
- Designed for image data (e.g., MNIST)

### 3. Variational Autoencoder (VAE)
- Probabilistic generative model
- Encoder outputs: `z`, `z_mean`, `z_log_var`
- Uses reparameterization trick (`sampling` function)
- Latent space of lower dimension for generation

## Usage

```bash
# Test Vanilla Autoencoder
./0-main.py

# Test Sparse Autoencoder
./1-main.py

# Test Convolutional Autoencoder
./2-main.py

# Test Variational Autoencoder
./3-main.py
