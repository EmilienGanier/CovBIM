"""High-level pipeline: points -> Tyler covariances -> weighted kNN graph -> geodesic distances."""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from . import _core

KERNELS = {"triweight": _core.Kernel.Triweight, "bump": _core.Kernel.Bump, "dumb": _core.Kernel.Dumb}
SPD_METRICS = {
    "frobenius": _core.SpdMetric.Frobenius,
    "log_euclidean": _core.SpdMetric.LogEuclidean,
    "airm": _core.SpdMetric.Airm,
}


def _as_points(X) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("points must have shape (N, n)")
    return X


# ----------------------------------------------------------------------------- thin wrappers

def knn(X, k: int, include_self: bool = False):
    """Exact kNN. Returns (indices (N, k) int64, distances (N, k))."""
    return _core.knn(_as_points(X), k, include_self)


@dataclass
class TylerResult:
    covariances: np.ndarray   # (N, n, n), trace-normalized
    locations: np.ndarray     # (N, n)
    radii: np.ndarray         # (N,)
    iterations: np.ndarray    # (N,)
    converged: np.ndarray     # (N,) bool


def tyler_covariances(X, k: int = 30, kernel: str = "triweight", joint_location: bool = True,
                      tol: float = 1e-6, max_iter: int = 50, ridge: float = 1e-10) -> TylerResult:
    X = _as_points(X)
    P, mu, radii, iters, conv = _core.tyler_covariances(
        X, k, KERNELS[kernel], joint_location, tol, max_iter, ridge)
    n = X.shape[1]
    return TylerResult(P.reshape(len(X), n, n), mu, radii, iters, conv.astype(bool))


# ----------------------------------------------------------------------------- configuration

@dataclass
class CovarianceConfig:
    k: int = 30
    kernel: str = "dumb"        # "triweight" | "bump" | "dumb"
    joint_location: bool = True
    tol: float = 1e-6
    max_iter: int = 50
    ridge: float = 1e-10


@dataclass
class MetricConfig:
    alpha: float = 1.0               # determinant-penalty exponent (0 = no penalty)
    beta: float = 1.0                # edge-length exponent (> 1: Fermat-style path selection)
    lam: float = 1.0                 # stiffness of the SPD part
    spd_metric: str = "log_euclidean"  # "frobenius" | "log_euclidean" | "airm"
    min_eig: float = 1e-12


@dataclass
class GraphConfig:
    k: int = 15                      # neighbors used to build the shortest-path graph


@dataclass
class PipelineConfig:
    covariance: CovarianceConfig = field(default_factory=CovarianceConfig)
    metric: MetricConfig = field(default_factory=MetricConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        """Build from a (YAML-loaded) dict, e.g. {"covariance": {"k": 40}, "metric": {"alpha": 2}}."""
        return cls(CovarianceConfig(**d.get("covariance", {})),
                   MetricConfig(**d.get("metric", {})),
                   GraphConfig(**d.get("graph", {})))

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------- pipeline

class GeodesicMetric:
    """fit(points) runs the expensive steps once; distances_from(source) is then cheap.

        gm = GeodesicMetric(PipelineConfig.from_dict({"metric": {"alpha": 2, "beta": 3}}))
        gm.fit(X)
        d = gm.distances_from(0)
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def fit(self, points) -> "GeodesicMetric":
        cfg = self.config
        X = _as_points(points)
        N = len(X)

        self.points_ = X
        self.tyler_ = tyler_covariances(X, **asdict(cfg.covariance))
        if not self.tyler_.converged.all():
            frac = 1.0 - self.tyler_.converged.mean()
            warnings.warn(f"Tyler iterations did not converge at {100 * frac:.1f}% of the points")

        flat = np.ascontiguousarray(self.tyler_.covariances.reshape(N, -1))
        m = cfg.metric
        self._weighter = _core.EdgeWeighter(
            X, flat, alpha=m.alpha, beta=m.beta, lam=m.lam,
            spd_metric=SPD_METRICS[m.spd_metric], min_eig=m.min_eig)

        nn_idx, _ = _core.knn(X, cfg.graph.k, False)
        indptr, indices, data = self._weighter.build_graph(np.ascontiguousarray(nn_idx, dtype=np.int64))
        self._graph = (np.ascontiguousarray(indptr, dtype=np.int64),
                       np.ascontiguousarray(indices, dtype=np.int64),
                       np.ascontiguousarray(data, dtype=np.float64))

        self.n_components_ = connected_components(self.csr_graph(), directed=False)[0]
        if self.n_components_ > 1:
            warnings.warn(f"the graph has {self.n_components_} connected components: "
                          "some distances will be infinite (increase graph.k)")
        return self

    # -- results ---------------------------------------------------------------------------

    @property
    def covariances(self) -> np.ndarray:
        return self.tyler_.covariances

    @property
    def determinants(self) -> np.ndarray:
        """det of the trace-normalized covariances: the scalar field that penalizes intersections."""
        return np.linalg.det(self.tyler_.covariances)

    def csr_graph(self) -> sp.csr_matrix:
        """The weighted graph as a scipy matrix (handy to cross-check with scipy's Dijkstra)."""
        indptr, indices, data = self._graph
        N = len(self.points_)
        return sp.csr_matrix((data, indices, indptr), shape=(N, N))

    def shortest_paths(self, source: int):
        """Dijkstra with the graph weights (exponent beta). Returns (distances, predecessors, order)."""
        return _core.dijkstra(*self._graph, int(source))

    def distances_from(self, source: int, recompute: bool = True) -> np.ndarray:
        """Distances from `source` to every point (inf if unreachable).

        With beta > 1 the paths are selected with the beta-weights; if `recompute` is True the returned
        value is the beta = 1 (Riemannian) length of those paths.
        """
        dist, pred, order = self.shortest_paths(source)
        if recompute and self.config.metric.beta != 1.0:
            dist = self._weighter.tree_lengths(pred, order, int(source), 1.0)
        return np.asarray(dist)

    def path(self, source: int, target: int) -> list[int]:
        """Node sequence of the selected shortest path (empty if unreachable)."""
        _, pred, _ = self.shortest_paths(source)
        path, v = [], int(target)
        while v != source:
            if v < 0 or len(path) > len(pred):
                return []
            path.append(v)
            v = int(pred[v])
        path.append(int(source))
        return path[::-1]

    def distance_matrix(self, sources) -> np.ndarray:
        """(S, N) matrix of distances from each source."""
        return np.stack([self.distances_from(s) for s in sources])