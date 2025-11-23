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

sizes = np.arange(1000, 100001, 1000).tolist()
k_list = [10, 50, 100]

results = {"sizes": sizes, "errors": {}}

for k in k_list:
    errors = []
    for n in tqdm(sizes, desc=f"k={k}"):
        Z = np.random.randn(4, n)
        X = L @ Z
        X1, X2, X3, X4 = X

        est = estimate_mi([X1, X2, X3, X4], k_neighbors=k)
        rel_err = abs(true_mi - est) / true_mi
        errors.append(rel_err)

    results["errors"][str(k)] = errors

with open("samples_errors.json", "w") as f:
    json.dump(results, f, indent=2)
