import json
import numpy as np
from tqdm import tqdm
from src.core import estimate_mi

SEED = 1
np.random.seed(SEED)

Sigma = np.array([
    [1.0, 0.6, 0.3, 0.2],
    [0.6, 1.0, 0.4, 0.1],
    [0.3, 0.4, 1.0, 0.5],
    [0.2, 0.1, 0.5, 1.0]
])

L = np.linalg.cholesky(Sigma)
variances = np.diag(Sigma)
true_mi = 0.5 * np.log(np.prod(variances) / np.linalg.det(Sigma))

sample_sizes = [100_000, 10_000, 1_000]
k_values = list(range(10, 201, 10))

results = {
    "sample_sizes": sample_sizes,
    "k_values": k_values,
    "errors": {}
}

for N in sample_sizes:
    Z = np.random.randn(4, N)
    X = L @ Z
    X1, X2, X3, X4 = X

    errors = []
    for k in tqdm(k_values, desc=f"{N} samples"):
        est = estimate_mi([X1, X2, X3, X4], k_neighbors=k)
        rel_err = abs(true_mi - est) / true_mi
        errors.append(rel_err)

    results["errors"][str(N)] = errors

with open("k_errors.json", "w") as f:
    json.dump(results, f, indent=2)
