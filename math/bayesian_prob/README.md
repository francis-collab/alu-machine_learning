# Bayesian Probability – Binomial Likelihood & Posterior Calculation

This project implements core components of **Bayesian inference** for binomial data (e.g., success/failure outcomes such as patients developing side effects after taking a drug).

The goal is to compute:

- Likelihood of observed data given different candidate probabilities
- Intersection (likelihood × prior)
- Marginal probability of the data
- Posterior probability distribution (updated beliefs after observing data)

All functions follow strict input validation and are implemented using only `numpy`.

---

## Files

| File | Description |
|-----|-------------|
| `0-likelihood.py` | Computes binomial likelihood `P(data \| p)` for multiple candidate values of `p` |
| `1-intersection.py` | Computes the unnormalized posterior = likelihood × prior |
| `2-marginal.py` | Computes the marginal probability `P(data) = Σ (likelihood × prior)` |
| `3-posterior.py` | Computes the normalized posterior `P(p \| data)` |

---

## Project Requirements

- Python 3.5  
- `numpy` 1.15  
- Ubuntu 16.04 LTS environment  
- Code style: `pycodestyle` 2.5  
- No external libraries except `numpy`

---

## Usage Example

```python
import numpy as np

# Example: 26 side effects observed out of 130 patients
x = 26
n = 130

# Grid of candidate probabilities
P = np.linspace(0, 1, 11)

# Uniform prior
Pr = np.ones_like(P) / len(P)

# 0. Likelihood
from likelihood import likelihood
L = likelihood(x, n, P)

# 1. Intersection (likelihood × prior)
from intersection import intersection
inter = intersection(x, n, P, Pr)

# 2. Marginal probability P(data)
from marginal import marginal
m = marginal(x, n, P, Pr)

# 3. Posterior P(p | data)
from posterior import posterior
post = posterior(x, n, P, Pr)
```

---

## Expected Output (uniform prior, x=26, n=130)

```
Likelihood:
[0.00000000e+00 2.71330957e-04 8.71800070e-02 ... 0.00000000e+00]

Intersection:
[0.00000000e+00 2.46664506e-05 7.92545518e-03 ... 0.00000000e+00]

Marginal probability:
0.008229580791426582

Posterior:
[0.00000000e+00 2.99729127e-03 9.63044824e-01 ... 0.00000000e+00]
```

---

## Learning Objectives

- Understand the four key terms in **Bayes' theorem**:
  - Prior `P(p)`
  - Likelihood `P(data | p)`
  - Marginal / Evidence `P(data)`
  - Posterior `P(p | data)`

- Implement proper input validation and meaningful error messages

- Work with discrete approximations of continuous probability distributions

- Normalize posterior distributions correctly

---

## Author

Francis Mutabazi  
ALU Machine Learning Program