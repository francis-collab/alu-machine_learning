# Supervised Learning - Classification

This project implements various **Classification models from scratch** using only `numpy`

## Project Structure

| Task | File | Description |
|------|------|-----------|
| 0 | `0-neuron.py` | Single Neuron (public attributes) |
| 1 | `1-neuron.py` | Single Neuron (private attributes + getters) |
| 2 | `2-neuron.py` | Forward Propagation + Sigmoid |
| 3 | `3-neuron.py` | Cost Calculation (Logistic Regression) |
| 4 | `4-neuron.py` | Evaluate predictions |
| 5 | `5-neuron.py` | Gradient Descent |
| 6 | `6-neuron.py` | Training the Neuron |
| 7 | `7-neuron.py` | **Upgraded Train** (verbose + graph) |
| 8 | `8-neural_network.py` | Neural Network (1 hidden layer) - public |
| 9 | `9-neural_network.py` | Neural Network - private attributes |
| 10 | `10-neural_network.py` | Forward Propagation (1 hidden layer) |
| 11 | `11-neural_network.py` | Cost function |
| 12 | `12-neural_network.py` | Evaluate |
| 13 | `13-neural_network.py` | Gradient Descent |
| 14 | `14-neural_network.py` | Training |
| 15 | `15-neural_network.py` | **Upgraded Train** (verbose + graph) |
| 16 | `16-deep_neural_network.py` | Deep Neural Network (public) |
| 17 | `17-deep_neural_network.py` | Deep Neural Network (private) |
| 18 | `18-deep_neural_network.py` | Forward Propagation (Deep) |
| 19 | `19-deep_neural_network.py` | Cost |
| 20 | `20-deep_neural_network.py` | Evaluate |
| 21 | `21-deep_neural_network.py` | Gradient Descent (backpropagation) |
| 22 | `22-deep_neural_network.py` | Training |
| 23 | `23-deep_neural_network.py` | **Upgraded Train** (verbose + graph) |
| 24 | `24-one_hot_encode.py` | One-Hot Encoding |
| 25 | `25-one_hot_decode.py` | One-Hot Decoding |
| 26 | `26-deep_neural_network.py` | Save & Load model (pickle) |
| 27 | `27-deep_neural_network.py` | Multiclass Classification |
| 28 | `28-deep_neural_network.py` | **Different Activations** (`sigmoid` & `tanh`) |

## Features Implemented

- **Binary Classification** (Single Neuron → Shallow NN → Deep NN)
- **Multiclass Classification** (with one-hot encoding)
- **Forward Propagation**
- **Backpropagation** (Gradient Descent)
- **Cost Function** (Binary Cross-Entropy)
- **Evaluation** (Accuracy, Predictions)
- **Training Loop** with early stopping visualization
- **Verbose output** and **Training Cost Graph** using matplotlib
- **Model Persistence** (Save/Load with pickle)
- **Flexible Activation Functions** (Sigmoid & Tanh)
- **He Initialization** for deep networks

## Constraints Respected

- Only `import numpy as np` (and matplotlib for graphs)
- Limited use of loops (only where explicitly allowed)
- Proper documentation for all modules, classes, and methods
- All files executable with shebang
- `pycodestyle` compliant

## How to Test

```bash
# Example: Test task 7
./0-main.py

# Test upgraded neuron
./7-main.py

# Test Deep Neural Network (multiclass)
./27-main.py
```

## Datasets Used

- **Binary_Train.npz / Binary_Dev.npz** — Binary classification
- **MNIST.npz** — Multiclass digit classification