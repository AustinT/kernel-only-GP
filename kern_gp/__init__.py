"""
Contains code for zero-mean GP with kernel a*k(x,x) + s*I for some base kernel k.

Throughout, the following notation is used:
- L is the Cholesky decomposition of (K + (s/a)I), where K is the kernel matrix between training points.
"""
import logging

import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, cholesky, solve_triangular

LOWER = True
logger = logging.getLogger(__name__)


def mll_train(a, s, k_train_train, y_train):
    """Computes the marginal log likelihood of the training data."""
    L = _k_cholesky(k_train_train, s / a)
    return _L_mll_train(a, L, y_train)


def _L_mll_train(a, L, y_train):
    data_fit = _data_fit(L, a, y_train)
    complexity = _complexity(L, a)
    constant = -L.shape[0] / 2 * jnp.log(2 * jnp.pi)
    return data_fit + complexity + constant


def noiseless_predict(a, s, k_train_train, k_test_train, k_test_test, y_train, full_covar: bool = True):
    """
    Computes mean and [co]variance predictions for the test data given training data.

    Full covar means we return the full covariance matrix, otherwise we return the diagonal.
    """

    L = _k_cholesky(k_train_train, s / a)
    return _L_noiseless_predict(a, L, k_test_train, k_test_test, y_train, full_covar)


def _L_noiseless_predict(a, L, k_test_train, k_test_test, y_train, full_covar: bool = True):
    mean = jnp.dot(k_test_train, cho_solve((L, LOWER), y_train))
    covar_adj_sqrt = solve_triangular(L, k_test_train.T, lower=LOWER)
    if full_covar:
        covar_adj = covar_adj_sqrt.T @ covar_adj_sqrt
    else:
        covar_adj = jnp.sum(covar_adj_sqrt**2, axis=0)

    return mean, a * (k_test_test - covar_adj)


def _k_cholesky(k, s):
    """Computes cholesky of k+sI."""
    logger.debug(f"Computing cholesky of {k.shape[0]}x{k.shape[0]} matrix with s={s}")
    k2 = k + s * jnp.eye(k.shape[0])
    L = cholesky(k2, lower=LOWER)
    logger.debug("Done computing cholesky")
    return L


def _data_fit(L, a, y_train):
    return -0.5 / a * jnp.dot(y_train.T, cho_solve((L, LOWER), y_train))


def _complexity(L, a):
    """MLL complexity term for kernel a(L@L^T)"""
    log_det_L = -jnp.sum(jnp.log(jnp.diag(L)))  # because we use cholesky, the factor of 2 cancels so no -1/2
    a_adjustment = -jnp.log(a) * L.shape[0] / 2
    return log_det_L + a_adjustment
