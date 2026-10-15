# from . import _core
# import numpy as np

# def tyler_covariances(X, k=30, **kw):
#     X = np.ascontiguousarray(X, dtype=np.float64)
#     P, mu, r, iters, conv = _core.tyler_covariances(X, k, **kw)
#     return P.reshape(len(X), X.shape[1], X.shape[1]), mu, r, iters, conv

from . import _core, baselines, datasets, viz
from .pipeline import (
    CovarianceConfig, GeodesicMetric, GraphConfig, MetricConfig,
    PipelineConfig, TylerResult, knn, tyler_covariances,
)

__all__ = [
    "GeodesicMetric", "PipelineConfig", "CovarianceConfig", "MetricConfig", "GraphConfig",
    "TylerResult", "knn", "tyler_covariances",
    "datasets", "baselines", "viz",
]