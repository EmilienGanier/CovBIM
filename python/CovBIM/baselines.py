"""Baselines sharing one signature: f(points, source, **kwargs) -> distances (N,).

To add one, write a function with that signature and put it in BASELINES.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


def _neighbors(X, k):
    dist, idx = cKDTree(X).query(X, k=k + 1)
    return idx[:, 1:], dist[:, 1:]          # drop the point itself


def _assemble(idx, weights, keep=None, mode="union"):
    """Directed kNN graph -> symmetric graph. 'union': edge if either endpoint lists the other;
    'intersection': only if both do."""
    N, k = idx.shape
    keep = np.ones(idx.shape, dtype=bool) if keep is None else keep
    rows = np.repeat(np.arange(N), k)[keep.ravel()]
    G = csr_matrix((weights.ravel()[keep.ravel()], (rows, idx.ravel()[keep.ravel()])), shape=(N, N))
    G = G.maximum(G.T) if mode == "union" else G.minimum(G.T)
    G.eliminate_zeros()
    return G


def euclidean_geodesics(points, source, k=15):
    """Isomap-style geodesics: Dijkstra on the Euclidean kNN graph."""
    idx, dist = _neighbors(np.asarray(points, dtype=float), k)
    return dijkstra(_assemble(idx, dist), directed=False, indices=source)


def power_weighted_distances(points, source, k=15, p=3.0, root=True):
    """Fermat-like distance on the kNN graph: shortest path for weights ||x_i - x_j||^p
    (then the p-th root if `root`). The kNN graph is an approximation of the complete graph."""
    idx, dist = _neighbors(np.asarray(points, dtype=float), k)
    d = dijkstra(_assemble(idx, dist**p), directed=False, indices=source)
    return d ** (1.0 / p) if root else d


def mcd_filtered_distances(points, source, k=15, quantile=0.975, random_state=0):
    """Euclidean kNN graph where an edge i-j is kept only if j is an inlier of the local MCD fit
    around i AND i is an inlier around j. (This is *a* definition of the baseline: adapt it to
    the one you want to cite.) Requires scikit-learn."""
    from scipy.stats import chi2
    from sklearn.covariance import MinCovDet

    X = np.asarray(points, dtype=float)
    N, n = X.shape
    dist, idx = cKDTree(X).query(X, k=k + 1)
    thr = chi2.ppf(quantile, df=n)
    keep = np.ones((N, k), dtype=bool)
    for i in range(N):
        try:
            mcd = MinCovDet(random_state=random_state).fit(X[idx[i]])
            keep[i] = mcd.mahalanobis(X[idx[i, 1:]]) <= thr     # squared Mahalanobis vs chi2 quantile
        except ValueError:                                       # degenerate neighborhood: keep all
            pass
    G = _assemble(idx[:, 1:], dist[:, 1:], keep=keep, mode="intersection")
    return dijkstra(G, directed=False, indices=source)


BASELINES = {
    "euclidean": euclidean_geodesics,
    "fermat": power_weighted_distances,
    "mcd": mcd_filtered_distances,
}