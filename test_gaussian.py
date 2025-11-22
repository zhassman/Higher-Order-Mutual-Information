import numpy as np
import pytest
from src import estimate_mi


NUM_SAMPLES = 100_000
K_NEIGHBORS = 100
SEED = 1

np.random.seed(SEED)




@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
def test_gaussian_1d(sigma):

    X = np.random.randn(NUM_SAMPLES) * sigma
    X_noisy = X + .25 * np.random.randn(NUM_SAMPLES) # (.25 is a hyperparameter)

    est = estimate_mi([X, X_noisy], k_neighbors=K_NEIGHBORS)
    true = 0.5 * np.log(2 * np.pi * np.e * sigma**2)

    assert np.isclose(est, true, rtol=0.20, atol=0.05)




cov_2d_cases = [
    np.array([[1.0, 0.3],
              [0.3, 1.0]]),

    np.array([[1.0, 0.7],
              [0.7, 1.0]]),

    np.array([[2.0, 1.0],
              [1.0, 2.0]]),
]

@pytest.mark.parametrize("Sigma", cov_2d_cases)
def test_gaussian_2d(Sigma):

    L = np.linalg.cholesky(Sigma)
    Z = np.random.randn(2, NUM_SAMPLES)
    X = L @ Z
    X1, X2 = X

    est = estimate_mi([X1, X2], k_neighbors=K_NEIGHBORS)

    variances = np.diag(Sigma)
    true = 0.5 * np.log(np.prod(variances) / np.linalg.det(Sigma))

    assert np.isclose(est, true, rtol=0.20, atol=0.05)




cov_3d_cases = [
    np.array([
        [1.0, 0.4, 0.2],
        [0.4, 1.0, 0.1],
        [0.2, 0.1, 1.0]
    ]),
    np.array([
        [1.0, 0.7, 0.4],
        [0.7, 1.0, 0.2],
        [0.4, 0.2, 1.0]
    ])
]

@pytest.mark.parametrize("Sigma", cov_3d_cases)
def test_gaussian_3d(Sigma):

    L = np.linalg.cholesky(Sigma)
    Z = np.random.randn(3, NUM_SAMPLES)
    X = L @ Z
    X1, X2, X3 = X

    est = estimate_mi([X1, X2, X3], k_neighbors=K_NEIGHBORS)

    variances = np.diag(Sigma)
    true = 0.5 * np.log(np.prod(variances) / np.linalg.det(Sigma))

    assert np.isclose(est, true, rtol=0.20, atol=0.05)




cov_4d_cases = [
    np.eye(4),
    np.array([
        [1.0, 0.6, 0.3, 0.2],
        [0.6, 1.0, 0.4, 0.1],
        [0.3, 0.4, 1.0, 0.5],
        [0.2, 0.1, 0.5, 1.0]
    ])
]

@pytest.mark.parametrize("Sigma", cov_4d_cases)
def test_gaussian_4d(Sigma):

    L = np.linalg.cholesky(Sigma)
    Z = np.random.randn(4, NUM_SAMPLES)
    X = L @ Z
    X1, X2, X3, X4 = X

    est = estimate_mi([X1, X2, X3, X4], k_neighbors=K_NEIGHBORS)

    variances = np.diag(Sigma)
    true = 0.5 * np.log(np.prod(variances) / np.linalg.det(Sigma))

    assert np.isclose(est, true, rtol=0.20, atol=0.05)
