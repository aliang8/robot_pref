import numba as nb
import numpy as np


@nb.njit
def get_distance_matrix(query: np.ndarray, reference: np.ndarray):
    """
    Compute the pairwise Euclidean distance matrix between two sequences.

    Args:
        query (np.ndarray): Shape (N, D)
        reference (np.ndarray): Shape (M, D)

    Returns:
        np.ndarray: Shape (N, M) distance matrix
    """
    query_squared = np.sum(query**2, axis=1)[:, np.newaxis]
    ref_squared = np.sum(reference**2, axis=1)[np.newaxis, :]
    cross_term = np.dot(query, reference.T)
    distance_matrix = np.sqrt(query_squared - 2 * cross_term + ref_squared)
    return distance_matrix


@nb.njit
def compute_accumulated_cost_matrix_dtw(C: np.ndarray):
    """
    Compute the accumulated cost matrix for standard DTW.

    Args:
        C (np.ndarray): Cost matrix of shape (N, M)

    Returns:
        np.ndarray: Accumulated cost matrix of shape (N, M)
    """
    N, M = C.shape
    D = np.full((N, M), np.inf)
    D[0, 0] = C[0, 0]
    for n in range(1, N):
        D[n, 0] = C[n, 0] + D[n - 1, 0]
    for m in range(1, M):
        D[0, m] = C[0, m] + D[0, m - 1]
    for n in range(1, N):
        for m in range(1, M):
            D[n, m] = C[n, m] + min(D[n - 1, m], D[n, m - 1], D[n - 1, m - 1])
    return D


@nb.njit
def compute_optimal_warping_path_dtw(D: np.ndarray):
    """
    Backtrack to find the optimal warping path for standard DTW.

    Args:
        D (np.ndarray): Accumulated cost matrix of shape (N, M)

    Returns:
        np.ndarray: Optimal warping path as array of (n, m) index pairs
    """
    N, M = D.shape
    n, m = N - 1, M - 1
    path = [(n, m)]
    while n > 0 or m > 0:
        if n == 0:
            m -= 1
        elif m == 0:
            n -= 1
        else:
            choices = np.array([D[n - 1, m], D[n, m - 1], D[n - 1, m - 1]])
            argmin = np.argmin(choices)
            if argmin == 0:
                n -= 1
                m -= 1
            elif argmin == 1:
                n -= 1
            else:
                m -= 1
        path.append((n, m))
    path.reverse()
    return np.array(path)


def get_single_match(query: np.ndarray, reference: np.ndarray):
    """
    Compute the standard DTW match between query and reference.

    Args:
        query (np.ndarray): Query trajectory, shape (N, D)
        reference (np.ndarray): Reference trajectory, shape (M, D)

    Returns:
        tuple: (cost, path)
            cost (float): DTW distance
            path (np.ndarray): Optimal warping path as array of (n, m) index pairs
    """
    distance_matrix = get_distance_matrix(query, reference)
    accumulated_cost_matrix = compute_accumulated_cost_matrix_dtw(distance_matrix)
    path = compute_optimal_warping_path_dtw(accumulated_cost_matrix)
    cost = float(accumulated_cost_matrix[-1, -1])
    return cost, path
