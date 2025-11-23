import numpy as np
from src.core import estimate_mi


def run_two_variable_tests(k_neighbors: int):
    rng = np.random.default_rng(0)

    # Identical variables
    X = rng.random(1000)
    mi_identical = estimate_mi([X, X], k_neighbors=k_neighbors)
    print("Identical variables:", mi_identical)

    # Completely independent variables
    X = rng.random(1000)
    Y = rng.random(1000)
    mi_independent = estimate_mi([X, Y], k_neighbors=k_neighbors)
    print("Independent variables:", mi_independent)

    # Linear dependence
    X = rng.random(1000)
    Y = 3 * X + rng.normal(0, 0.05, 1000)
    mi_linear = estimate_mi([X, Y], k_neighbors=k_neighbors)
    print("Linearly dependent variables:", mi_linear)

    # Quadratic dependence
    X = rng.random(1000)
    Y = X ** 2 + rng.normal(0, 0.05, 1000)
    mi_nonlinear = estimate_mi([X, Y], k_neighbors=k_neighbors)
    print("Nonlinear relationship:", mi_nonlinear)

    # Zero variance
    X = np.ones(1000)
    Y = rng.random(1000)
    mi_constant = estimate_mi([X, Y], k_neighbors=k_neighbors)
    print("Zero variance variable:", mi_constant)


if __name__ == "__main__":
    run_two_variable_tests(5)
