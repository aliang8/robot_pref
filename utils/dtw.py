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



@nb.jit(nopython=True)
def compute_optimal_warping_path_subsequence_dtw_21(D: np.ndarray, m=-1):
    """
    Args:
        D (np.ndarray): Accumulated cost matrix
        m (int): Index to start back tracking; if set to -1, optimal m is used (Default value = -1)

    Returns:
        P (np.ndarray): Optimal warping path (array of index pairs)
    """
    N, M = D.shape
    n = N - 1
    if m < 0:
        m = D[N - 1, :].argmin()
    P = [(n, m)]

    while n > 0:
        if m == 0:
            cell = (n - 1, 0)
        else:
            val = min(D[n - 1, m - 1], D[n - 1, m - 2])  # D[n-2, m-1],
            if val == D[n - 1, m - 1]:
                cell = (n - 1, m - 1)
            # elif val == D[n-2, m-1]:
            #     cell = (n-2, m-1)
            else:
                cell = (n - 1, m - 2)
        P.append(cell)
        n, m = cell
    P.reverse()
    P = np.array(P)
    return P

@nb.jit(nopython=True)
def compute_accumulated_cost_matrix_subsequence_dtw_21(C: np.ndarray):
    """
    Args:
        C (np.ndarray): Cost matrix
    Returns:
        D (np.ndarray): Accumulated cost matrix
    """
    N, M = C.shape
    D = np.zeros((N + 1, M + 2))
    D[0:1, :] = np.inf
    D[:, 0:2] = np.inf

    D[1, 2:] = C[0, :]

    for n in range(1, N):
        for m in range(0, M):
            if n == 0 and m == 0:
                continue
            D[n + 1, m + 2] = C[n, m] + min(
                D[n - 1 + 1, m - 1 + 2], D[n - 1 + 1, m - 2 + 2]
            )  # D[n-2+1, m-1+2],
    D = D[1:, 2:]
    return D

def get_single_match_subsequence(query: np.ndarray, play: np.ndarray):
    """Get single match using S-DTW."""
    """
    Args:
        query (np.ndarray): Query trajectory
        play (np.ndarray): Play trajectory
    Returns:
        cost (float): Cost of the match
        start (int): Start index of the match
        end (int): End index of the match
    """
    distance_matrix = get_distance_matrix(query, play)
    accumulated_cost_matrix = compute_accumulated_cost_matrix_subsequence_dtw_21(
        distance_matrix
    )
    path = compute_optimal_warping_path_subsequence_dtw_21(accumulated_cost_matrix)
    start = path[0, 1]
    if start < 0:
        # assert start == -1 # TODO: do we need this?
        start = 0
    end = path[-1, 1]
    cost = accumulated_cost_matrix[-1, end]

    end = (
        end + 1
    )  # Note that the actual end index is inclusive in this case so +1 to use python : based indexing

    return cost, path