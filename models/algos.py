import numpy as np

def floyd_warshall(adjacency_matrix):
    (nrows, ncols) = adjacency_matrix.shape
    n = nrows

    adj_mat_copy = adjacency_matrix.astype(np.float64, order='C', casting='safe', copy=True)
    M = adj_mat_copy
    path = np.zeros([n, n], dtype=np.float64)

    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 0

    for k in range(n):
        for i in range(n):
            M_ik = M[i][k]
            for j in range(n):
                cost_ikkj = M_ik + M[k][j]
                M_ij = M[i][j]
                if M_ij > cost_ikkj:
                    M[i][j] = cost_ikkj
                    path[i][j] = k

    return M, path.astype(int)


def get_all_edges(path, i, j):
    k = path[i][j]
    if k == 0:
        return []
    else:
        return get_all_edges(path, i, k) + [k] + get_all_edges(path, k, j)


def gen_edge_input(max_hop, path, edge_feat, region):
    (nrows, ncols) = path.shape
    assert nrows == ncols
    n = nrows
    max_hop_copy = max_hop

    path_copy = path.astype(np.int64, order='C', casting='safe', copy=True)
    edge_feat_copy = edge_feat.astype(np.float64, order='C', casting='safe', copy=True)

    edge_fea_all = -3 * np.ones([n, n, max_hop_copy, edge_feat.shape[-1]], dtype=np.float64)
    max_numpath = 0

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            path = [i] + get_all_edges(path_copy, i, j) + [j]
            num_path = len(path) - 1
            for k in range(num_path):
                flag = not (region[path[k]]==region[path[k+1]])
                first = min(path[k], path[k+1])
                second = max(path[k], path[k+1])
                flag = -1 if region[path[k]]==region[path[k+1]] else -2
                edge_fea_all[i, j, k, :] = np.array([flag, first, second])
            max_numpath = max(max_numpath, num_path)

    return edge_fea_all[:, :, :max_numpath, :] + 3