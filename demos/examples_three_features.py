import numpy as np
from src.core import estimate_mi


def run_three_variable_tests(k_neighbors: int):
    rng = np.random.default_rng(0)

    # Identical variables
    X = rng.random(1000)
    mi_identical = estimate_mi([X, X, X], k_neighbors=k_neighbors)
    print("Identical variables:", mi_identical)

    # Completely independent variables
    X = rng.random(1000)
    Y = rng.random(1000)
    Z = rng.random(1000)
    mi_independent = estimate_mi([X, Y, Z], k_neighbors=k_neighbors)
    print("Independent variables:", mi_independent)

    # Linear dependence
    X = rng.random(1000)
    Y = 2 * X + rng.normal(0, 0.05, 1000)
    Z = X + Y + rng.normal(0, 0.05, 1000)
    mi_linear = estimate_mi([X, Y, Z], k_neighbors=k_neighbors)
    print("Linearly dependent variables:", mi_linear)

    # Nonlinear dependence
    X = rng.random(1000)
    Y = X ** 2 + rng.normal(0, 0.05, 1000)
    Z = np.sin(np.pi * X) + rng.normal(0, 0.05, 1000)
    mi_nonlinear = estimate_mi([X, Y, Z], k_neighbors=k_neighbors)
    print("Nonlinear relationship:", mi_nonlinear)

    # XOR-style higher-order dependence
    A = rng.integers(0, 2, 2000).astype(float)
    B = rng.integers(0, 2, 2000).astype(float)
    C = np.bitwise_xor(A.astype(int), B.astype(int)).astype(float)
    eps = 1e-2
    Af = A + rng.normal(0, eps, 2000)
    Bf = B + rng.normal(0, eps, 2000)
    Cf = C + rng.normal(0, eps, 2000)

    mi_xor = estimate_mi([Af, Bf, Cf], k_neighbors=k_neighbors)
    print("XOR-style dependence:", mi_xor)


if __name__ == "__main__":
    run_three_variable_tests(5)
