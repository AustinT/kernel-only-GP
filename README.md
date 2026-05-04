# Kernel only GP

A minimal Python package for GP inference given only kernel matrices — not the underlying data points. Useful when working with data that cannot be stored as a tensor.

## Usage

```python
import jax.numpy as jnp
import kern_gp

# Kernel matrices (e.g. from an RBF kernel evaluated on your data)
# In this package, *these* are the GP inputs (not the data)
k_train_train = jnp.array([[1.0, 0.8], [0.8, 1.0]])
k_test_train  = jnp.array([[0.6, 0.9]])
k_test_test   = jnp.array([[1.0]])

y_train = jnp.array([1.2, 0.8])

# Hyperparameters: output scale and noise variance
a = 1.0   # output scale
s = 0.01  # noise variance

# Marginal log likelihood (useful for training / hyperparameter selection)
mll = kern_gp.mll_train(a, s, k_train_train, y_train)

# Posterior mean and covariance at test points (noise not added back)
mean, covar = kern_gp.noiseless_predict(a, s, k_train_train, k_test_train, k_test_test, y_train)
```

The kernel matrices should use the **base kernel** (without the output scale `a`). The full GP kernel is `a·k(x,x') + s·I`, where `s` is the noise variance.

## Installation

```bash
pip install kern-gp
```

## Development

```bash
# Install all deps (including dev group)
uv sync

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest
```
