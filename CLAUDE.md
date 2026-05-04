# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install all deps (including dev group)
uv sync

# Install pre-commit hooks (required before committing)
uv run pre-commit install

# Run all tests
uv run pytest

# Run a single test
uv run pytest test_kern_gp.py::test_mll_train

# Lint and format (also runs automatically on commit)
uv run black .
uv run ruff check . --fix
```

Line length is 120 (configured in `pyproject.toml` for both black and ruff).

## Architecture

The entire library lives in `kern_gp/__init__.py`. It implements a **zero-mean GP** with kernel `a·k(x,x) + s·I`, where `a` is the output scale and `s` is the noise variance. All functions operate on **pre-computed kernel matrices** rather than raw data points.

**Public API:**
- `mll_train(a, s, k_train_train, y_train)` — marginal log likelihood of training data
- `noiseless_predict(a, s, k_train_train, k_test_train, k_test_test, y_train, full_covar)` — posterior mean and covariance (or diagonal) at test points, without noise added back

**Private helpers** with `_L_` prefix accept a pre-computed Cholesky factor `L = chol(K + (s/a)·I)`. These exist so callers can compute Cholesky once and reuse it across MLL and prediction in a training loop.

**Tests** (`test_kern_gp.py`) validate against `gpytorch` as a reference implementation, using a `ScaleKernel(RBFKernel())` GP with matching hyperparameters. The base kernel matrix (without the output scale) is passed to our functions, with `a=outputscale` and `s=noise`.

## Key conventions

- JAX arrays throughout (`jax.numpy`); Cholesky is always lower-triangular (`LOWER = True`)
- Version is derived from git tags via `setuptools_scm`
