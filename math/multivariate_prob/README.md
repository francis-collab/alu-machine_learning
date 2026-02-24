# Multivariate Probability

## Description
This project implements key concepts in multivariate probability using only NumPy:
- Calculation of mean and covariance matrix of a dataset
- Conversion of a covariance matrix to a correlation matrix
- A `MultiNormal` class representing a multivariate normal distribution (with mean, covariance, and PDF evaluation)

All code follows the project requirements:
- Python 3.5 + NumPy 1.15
- No external modules except `import numpy as np`
- Full documentation for modules, classes, and functions
- `pycodestyle` compliant
- Shebang, executable, ends with newline

## Files
- **0-mean_cov.py** – `mean_cov(X)` function  
- **1-correlation.py** – `correlation(C)` function  
- **multinormal.py** – `MultiNormal` class (used by tasks 2 & 3)  
- **README.md** – this file  

## Usage Examples
See the provided `*-main.py` files in the project description (they are the official testers).

Run them after placing the files in `math/multivariate_prob/`:
```bash
./0-main.py
./1-main.py
./2-main.py
./3-main.py
```
Expected outputs match the examples exactly (up to floating-point precision).

## Repository Structure (as required)

```
alu-machine_learning/
└── math/
    └── multivariate_prob/
        ├── 0-mean_cov.py
        ├── 1-correlation.py
        ├── multinormal.py
        └── README.md
```