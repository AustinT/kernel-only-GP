import math

import gpytorch
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from kern_gp import mll_train, noiseless_predict

X_train = jnp.array([[1, 2, 3], [4, 5, 6]]) / 5
y_train = jnp.array([0.5, 1.0])
X_test = jnp.array([[3, 2, 1], [6, 5, 4]]) / 5

OUTPUTSCALE_LIST = [0.1, 1.0, 10.0]
NOISE_LIST = list(OUTPUTSCALE_LIST)


class SimpleGP(gpytorch.models.ExactGP):
    def __init__(self, X_train, y_train, likelihood):
        super().__init__(
            torch.as_tensor(np.array(X_train)),
            torch.as_tensor(np.array(y_train)),
            likelihood,
        )
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def _setup_gp(outputscale, noise):
    """Returns a GP with the given outputscale and noise."""
    gp = SimpleGP(X_train, y_train, gpytorch.likelihoods.GaussianLikelihood())
    gp.covar_module.outputscale = outputscale
    gp.likelihood.noise = noise
    return gp


@pytest.mark.parametrize("outputscale", OUTPUTSCALE_LIST)
@pytest.mark.parametrize("noise", NOISE_LIST)
def test_mll_train(outputscale, noise):
    # Make gpytorch GP and get its marginal likelihood
    gp = _setup_gp(outputscale, noise)
    mll_obj = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    with gpytorch.settings.fast_computations(False, False, False), torch.no_grad():
        mll_gpytorch = mll_obj(gp(torch.as_tensor(np.array(X_train))), torch.as_tensor(np.array(y_train))).item()
        mll_gpytorch *= len(X_train)  # it returns the average, we want the sum

    # Make our GP and get its marginal likelihood
    with torch.no_grad():
        k_train_train = gp.covar_module.base_kernel(torch.as_tensor(np.array(X_train))).to_dense().numpy()
    our_mll = mll_train(outputscale, noise, k_train_train, y_train)

    # Test: are they close?
    assert math.isclose(mll_gpytorch, our_mll, rel_tol=1e-5)


@pytest.mark.parametrize("outputscale", OUTPUTSCALE_LIST)
@pytest.mark.parametrize("noise", NOISE_LIST)
@pytest.mark.parametrize("full_covar", [True, False])
def test_noiseless_predict(outputscale, noise, full_covar):
    # Make gpytorch GP and get its predictions
    gp = _setup_gp(outputscale, noise)
    gp.eval()
    with gpytorch.settings.fast_computations(False, False, False), torch.no_grad():
        output = gp(torch.as_tensor(np.array(X_test)))
        mu_true = output.mean.numpy()
        covar_true = output.covariance_matrix.numpy()

    # Get kernel matrices
    with torch.no_grad():
        X_train_t = torch.as_tensor(np.array(X_train))
        X_test_t = torch.as_tensor(np.array(X_test))
        k_train_train = gp.covar_module.base_kernel(X_train_t).to_dense().numpy()
        k_test_train = gp.covar_module.base_kernel(X_test_t, X_train_t).to_dense().numpy()
        k_test_test = gp.covar_module.base_kernel(X_test_t).to_dense().numpy()

    # Adjust matrices based on full covar
    if not full_covar:
        k_test_test = jnp.diag(k_test_test)
        covar_true = jnp.diag(covar_true)

    # Get predictions from our GP
    our_mu, our_covar = noiseless_predict(
        outputscale, noise, k_train_train, k_test_train, k_test_test, y_train, full_covar=full_covar
    )

    # Test: are they close?
    assert jnp.allclose(mu_true, our_mu)
    assert jnp.allclose(covar_true, our_covar, rtol=1e-5)
