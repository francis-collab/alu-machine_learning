# Convolutions and Pooling

This project implements fundamental operations used in **Convolutional Neural Networks (CNNs)** using only **NumPy**, without relying on high-level libraries such as TensorFlow or PyTorch.

The tasks cover:

- Valid, same, and custom-padding convolutions on grayscale images
- Strided convolution
- Convolution on multi-channel (RGB) images
- Convolution with multiple kernels (producing feature maps)
- Max and average pooling

All implementations respect the strict constraint of using a **maximum of two nested for loops** (three for multiple kernels in task 5), and do not use `np.convolve`.

---

## Project Requirements

- Python 3.5
- NumPy 1.15
- Ubuntu 16.04 LTS environment
- Code style: `pycodestyle` 2.5
- Allowed imports: `numpy` and `from math import ceil, floor` (when needed)
- No external libraries beyond the allowed ones
- All files must be executable and end with a newline

---

## Files

| File | Description |
|-----|-------------|
| `0-convolve_grayscale_valid.py` | Valid convolution on grayscale images |
| `1-convolve_grayscale_same.py` | Same convolution (zero-padding to preserve input size) |
| `2-convolve_grayscale_padding.py` | Convolution with custom padding (`ph`, `pw`) |
| `3-convolve_grayscale.py` | General grayscale convolution (padding mode + stride) |
| `4-convolve_channels.py` | Convolution on multi-channel images (e.g. RGB) |
| `5-convolve.py` | Convolution with multiple kernels (multiple output feature maps) |
| `6-pool.py` | Max and average pooling on multi-channel images |

---

## Usage Examples

Most tasks include a main file for demonstration (provided by the project):

```bash
# Example: task 0 - valid convolution
./0-main.py

# Example: task 3 - strided convolution
./3-main.py

# Example: task 6 - average pooling
./6-main.py
```

Typical output shapes (MNIST / animals dataset):

- Valid 3×3 kernel on 28×28 images → 26×26
- Same 3×3 kernel → 28×28
- Valid 3×3 kernel on 32×32×3 images → 30×30
- Pooling 2×2 stride 2 → halves spatial dimensions

---

## Learning Objectives

- Understand how convolution reduces spatial dimensions (**valid vs same**)
- Implement zero-padding manually
- Handle strides greater than 1
- Extend convolution from grayscale to multi-channel images
- Apply multiple filters to produce feature maps
- Implement max and average pooling operations
- Work within strict loop constraints to simulate low-level CNN operations

---

## Notes

- All convolutions are implemented using an explicit sliding window with two nested loops over output positions
- Padding is calculated correctly for `'same'`, `'valid'`, and custom cases
- Channel-wise summation is performed efficiently using NumPy broadcasting and axis reduction
- Pooling supports both `'max'` and `'avg'` modes

---

## Author

Francis Mutabazi  
ALU Machine Learning Program