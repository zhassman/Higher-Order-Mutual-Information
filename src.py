import numpy as np
from scipy.special import digamma
from sklearn.neighbors import KDTree, NearestNeighbors
from typing import List
from numpy.typing import NDArray


def estimate_mi(
    X_list: List[NDArray[np.float64]],
    k_neighbors: int
) -> float:
    """    
    Args:
        X_list: List of 1D NumPy arrays, each representing samples of a variable.
        k_neighbors: Number of nearest neighbors to use in the MI estimation.

    Returns:
        mutual_information (float)

    Adapted from scikit-learn. 

    Implements equation (30) from `Estimating Mutual Information.`
    """

    m_features = len(X_list)
    N_samples = X_list[0].shape[0]

    for X in X_list:
        assert X.shape[0] == N_samples, "All variables must have the same number of samples."

    X_joint = np.hstack([X.reshape((-1, 1)) for X in X_list])

    nn = NearestNeighbors(metric="chebyshev", n_neighbors=k_neighbors)
    nn.fit(X_joint)
    radius = nn.kneighbors()[0][:, -1]

    n_x_j_list = []
    for X_j in X_list:
        kd = KDTree(X_j.reshape((-1,1)), metric="chebyshev")
        n_x_j = kd.query_radius(X_j.reshape((-1, 1)), radius, count_only=True)
        n_x_j_list.append(np.array(n_x_j) - 1)

    mutual_information = (
        digamma(k_neighbors) 
        - (m_features - 1) / k_neighbors
        + (m_features - 1) * digamma(N_samples)
        - sum(np.mean(digamma(n_x_j)) for n_x_j in n_x_j_list)

    )
    return max(0, mutual_information)
