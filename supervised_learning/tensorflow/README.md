orFlow - Supervised Learning

This project implements a neural network from scratch using **TensorFlow 1.12** (without Keras).

## Tasks Completed

| Task | File | Description |
|------|------|-----------|
| 0 | `0-create_placeholders.py` | Create placeholders `x` and `y` |
| 1 | `1-create_layer.py` | Create a layer with He initialization |
| 2 | `2-forward_prop.py` | Forward propagation |
| 3 | `3-calculate_accuracy.py` | Calculate accuracy |
| 4 | `4-calculate_loss.py` | Calculate softmax cross-entropy loss |
| 5 | `5-create_train_op.py` | Create Gradient Descent training operation |
| 6 | `6-train.py` | Build, train, and save the full model |
| 7 | `7-evaluate.py` | Evaluate the model from a saved checkpoint |

## Requirements

- Python 3.5
- TensorFlow 1.12
- NumPy
- No use of `keras`
- Only `import tensorflow as tf` (except in main files)

## Files

All files are located in the `supervised_learning/tensorflow/` directory.

## Usage Example

```bash
# Train the model
./6-main.py

# Evaluate the model
./7-main.py
```

## Model Saving
The trained model is saved as:

- model.ckpt.data-00000-of-00001
- model.ckpt.index
- model.ckpt.meta

## Key Features

- Placeholders for input and labels
- He initialization (variance_scaling_initializer)
- nForward propagation with custom layers
- Softmax cross-entropy loss
- Gradient Descent optimization
- Training with validation monitoring
- Model persistence and restoration
- Accuracy and loss evaluation
