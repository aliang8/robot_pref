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
    query_squared = np.sum(query ** 2, axis=1)[:, np.newaxis]
    ref_squared = np.sum(reference ** 2, axis=1)[np.newaxis, :]
    cross_term = np.dot(query, reference.T)
    distance_matrix = np.sqrt(query_squared - 2 * cross_term + ref_squared)
    return distance_matrix

@nb.njit
def compute_accumulated_cost_matrix_subsequence_dtw_21(C: np.ndarray):
    """
    Compute the accumulated cost matrix for subsequence DTW.

    Args:
        C (np.ndarray): Cost matrix of shape (N, M)

    Returns:
        np.ndarray: Accumulated cost matrix of shape (N, M)
    """
    N, M = C.shape
    D = np.full((N + 1, M + 2), np.inf)
    D[1, 2:] = C[0, :]

    for n in range(1, N):
        for m in range(M):
            D[n + 1, m + 2] = C[n, m] + min(
                D[n, m + 1], D[n, m]
            )
    return D[1:, 2:]

@nb.njit
def compute_optimal_warping_path_subsequence_dtw_21(D: np.ndarray, m: int = -1):
    """
    Backtrack to find the optimal warping path for subsequence DTW.

    Args:
        D (np.ndarray): Accumulated cost matrix of shape (N, M)
        m (int): End index for backtracking (default: -1, use optimal endpoint)

    Returns:
        np.ndarray: Optimal warping path as array of (n, m) index pairs
    """
    N, M = D.shape
    n = N - 1
    if m < 0:
        m = np.argmin(D[N - 1, :])
    path = []
    path.append((n, m))
    while n > 0:
        if m == 0:
            cell = (n - 1, 0)
        else:
            if D[n - 1, m - 1] <= D[n - 1, m - 2]:
                cell = (n - 1, m - 1)
            else:
                cell = (n - 1, m - 2)
        path.append(cell)
        n, m = cell
    path.reverse()
    return np.array(path)

def get_single_match(query: np.ndarray, reference: np.ndarray):
    """
    Compute the best subsequence DTW match between query and reference.

    Args:
        query (np.ndarray): Query trajectory, shape (N, D)
        reference (np.ndarray): Reference trajectory, shape (M, D)

    Returns:
        tuple: (cost, start, end)
            cost (float): Cost of the match
            start (int): Start index in reference
            end (int): End index in reference (exclusive)
    """
    distance_matrix = get_distance_matrix(query, reference)
    accumulated_cost_matrix = compute_accumulated_cost_matrix_subsequence_dtw_21(distance_matrix)
    path = compute_optimal_warping_path_subsequence_dtw_21(accumulated_cost_matrix)
    start = int(path[0, 1])
    end = int(path[-1, 1])
    cost = float(accumulated_cost_matrix[-1, end])
    end = end + 1  # Make end exclusive for Python slicing
    return cost, start, end
