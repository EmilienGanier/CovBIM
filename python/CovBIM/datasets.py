"""Point-cloud datasets: synthetic generators, noise utilities, mesh / file loading.

To add a generator, write a function `(n_points, rng, **params) -> PointCloud` that returns
*noise-free* points, and decorate it with `@register("name")`. Noise is applied separately by
`make()` so experiments can vary it independently.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class PointCloud:
    points: np.ndarray                    # (N, n) float64
    labels: np.ndarray | None = None      # (N,) int branch ids, -1 for outliers; None if unknown
    name: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.points = np.ascontiguousarray(self.points, dtype=np.float64)
        if self.points.ndim != 2:
            raise ValueError("points must have shape (N, n)")
        if self.labels is not None:
            self.labels = np.asarray(self.labels, dtype=np.int64)
            if self.labels.shape != (len(self.points),):
                raise ValueError("labels must have shape (N,)")

    def __len__(self) -> int:
        return len(self.points)

    @property
    def dim(self) -> int:
        return self.points.shape[1]

    def replace(self, **changes) -> "PointCloud":
        return dataclasses.replace(self, **changes)


# ----------------------------------------------------------------------------- sampling helpers

def sample_sphere(n, rng, radius=1.0, center=(0.0, 0.0, 0.0)):
    dim = len(center)
    v = rng.normal(size=(n, dim))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return radius * v + np.asarray(center, dtype=float)


def sample_segment(n, rng, p0, p1):
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    return p0 + rng.random((n, 1)) * (p1 - p0)


def sample_triangles(vertices, faces, n, rng):
    """Uniform (area-weighted) samples on a triangle mesh."""
    tri = vertices[faces]                                          # (F, 3, 3)
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    f = rng.choice(len(faces), size=n, p=areas / areas.sum())
    u, v = rng.random((n, 1)), rng.random((n, 1))
    flip = (u + v) > 1.0                                           # fold back into the triangle
    u, v = np.where(flip, 1.0 - u, u), np.where(flip, 1.0 - v, v)
    a, b, c = tri[f, 0], tri[f, 1], tri[f, 2]
    return a + u * (b - a) + v * (c - a)


# ----------------------------------------------------------------------------- registry

_GENERATORS: dict[str, Callable[..., PointCloud]] = {}


def register(name: str):
    def deco(fn):
        _GENERATORS[name] = fn
        return fn
    return deco


def available() -> list[str]:
    return sorted(_GENERATORS)


def make(name: str, n_points: int = 2000, noise: float = 0.0, ambient: float = 0.0,
         seed: int = 0, **params) -> PointCloud:
    """Generate a registered dataset, then optionally add Gaussian and ambient noise (all seeded)."""
    if name not in _GENERATORS:
        raise KeyError(f"unknown dataset '{name}', available: {available()}")
    cloud = _GENERATORS[name](n_points=n_points, rng=np.random.default_rng(seed), **params)
    if noise > 0:
        cloud = add_gaussian_noise(cloud, noise, seed=seed + 1)
    if ambient > 0:
        cloud = add_ambient_noise(cloud, ambient, seed=seed + 2)
    return cloud


# ----------------------------------------------------------------------------- generators

@register("sphere_plane")
def sphere_plane(n_points, rng, radius=0.5, plane_size=2.0, sphere_fraction=None):
    """A sphere crossing a plane z = 0. Labels: 0 sphere, 1 plane.
    By default the split is proportional to area (uniform sampling density)."""
    if sphere_fraction is None:
        area_sphere = 4.0 * np.pi * radius**2
        sphere_fraction = area_sphere / (area_sphere + plane_size**2)
    n_s = int(round(n_points * sphere_fraction))
    n_p = n_points - n_s
    sphere = sample_sphere(n_s, rng, radius)
    plane = np.c_[(rng.random((n_p, 2)) - 0.5) * plane_size, np.zeros(n_p)]
    return PointCloud(np.vstack([sphere, plane]),
                      labels=np.r_[np.zeros(n_s), np.ones(n_p)],
                      name="sphere_plane")


@register("plane_with_pole")
def plane_with_pole(n_points, rng, width=2.0, height=1.0, pole_length=1.5, pole_fraction=0.15):
    """A rectangle in the plane y = 0 with a 1D pole attached below it (mixed dimensions: the
    fraction of points on the pole is a free parameter). Labels: 0 rectangle, 1 pole."""
    n_pole = int(round(n_points * pole_fraction))
    n_rect = n_points - n_pole
    rect = np.c_[(rng.random(n_rect) - 0.5) * width, np.zeros(n_rect), rng.random(n_rect) * height]
    pole = sample_segment(n_pole, rng, (0.0, 0.0, 0.0), (0.0, 0.0, -pole_length))
    return PointCloud(np.vstack([rect, pole]),
                      labels=np.r_[np.zeros(n_rect), np.ones(n_pole)],
                      name="plane_with_pole")


@register("crossing_lines")
def crossing_lines(n_points, rng, n_lines=2, length=2.0):
    """`n_lines` segments in 2D crossing at the origin. Label i = i-th line."""
    counts = np.full(n_lines, n_points // n_lines)
    counts[: n_points % n_lines] += 1
    parts, labels = [], []
    for i, c in enumerate(counts):
        d = 0.5 * length * np.array([np.cos(np.pi * i / n_lines), np.sin(np.pi * i / n_lines)])
        parts.append(sample_segment(c, rng, -d, d))
        labels.append(np.full(c, i))
    return PointCloud(np.vstack(parts), labels=np.concatenate(labels), name="crossing_lines")


@register("crossing_circles")
def crossing_circles(n_points, rng, n_lines=2, radius=1.0):
    """`n_circles` circles in 2D crossing at the origin. Label i = i-th circle."""
    if n_lines != 2:
        raise ValueError("crossing_circles only supports n_lines = 2")
    n_circles = 2
    counts = np.full(n_circles, n_points // n_circles)
    counts[: n_points % n_circles] += 1
    parts, labels = [], []
    for i, c in enumerate(counts):
        parts.append(sample_sphere(c, rng, radius=radius, center=(0.0 + (i - 0.5)*radius, 0.0)))
        labels.append(np.full(c, i))
    return PointCloud(np.vstack(parts), labels=np.concatenate(labels), name="crossing_circles")


# ----------------------------------------------------------------------------- noise

def add_gaussian_noise(cloud: PointCloud, sigma: float, seed: int = 0) -> PointCloud:
    """Isotropic Gaussian perturbation of standard deviation `sigma` in every ambient direction."""
    rng = np.random.default_rng(seed)
    pts = cloud.points + sigma * rng.normal(size=cloud.points.shape)
    return cloud.replace(points=pts, meta={**cloud.meta, "noise": sigma})


def add_ambient_noise(cloud: PointCloud, fraction: float, seed: int = 0, margin: float = 0.0) -> PointCloud:
    """Adds round(fraction * N) outliers drawn uniformly in the (inflated) bounding box.
    Outliers get label -1 when labels exist."""
    rng = np.random.default_rng(seed)
    lo, hi = cloud.points.min(axis=0), cloud.points.max(axis=0)
    pad = margin * (hi - lo)
    n_out = int(round(fraction * len(cloud)))
    outliers = rng.uniform(lo - pad, hi + pad, size=(n_out, cloud.dim))
    labels = None if cloud.labels is None else np.r_[cloud.labels, np.full(n_out, -1)]
    return cloud.replace(points=np.vstack([cloud.points, outliers]), labels=labels,
                         meta={**cloud.meta, "ambient": fraction})


# ----------------------------------------------------------------------------- files and meshes

_TEXT_SUFFIXES = {".xyz", ".txt", ".csv", ".pts"}


def normalize_unit(points: np.ndarray) -> np.ndarray:
    """Center the bounding box at the origin and scale its longest side to 1
    (so that noise levels are comparable across datasets)."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    scale = (hi - lo).max()
    return (points - 0.5 * (lo + hi)) / (scale if scale > 0 else 1.0)


def load_file(path, n_points: int | None = None, seed: int = 0, normalize: bool = True,
              usecols=None) -> PointCloud:
    """Load a point cloud from a file.

    - .npy, or .xyz/.txt/.csv/.pts (one point per row; `usecols` selects coordinate columns)
    - anything trimesh reads (.obj, .ply, .off, .stl, ...). For a triangle mesh, `n_points`
      area-weighted surface samples are drawn (vertices are used if `n_points` is None).
      For a point-only file, `n_points` subsamples the points.
    Requires `pip install trimesh` for mesh formats.
    """
    path = Path(path)
    rng = np.random.default_rng(seed)
    suffix = path.suffix.lower()
    kind = "points"

    if suffix == ".npy":
        pts = np.load(path)
    elif suffix in _TEXT_SUFFIXES:
        pts = np.loadtxt(path, delimiter="," if suffix == ".csv" else None, usecols=usecols)
    else:
        import trimesh                                         # optional dependency

        obj = trimesh.load(path)
        if isinstance(obj, trimesh.Scene):
            obj = obj.dump(concatenate=True)
        vertices = np.asarray(obj.vertices, dtype=float)
        faces = getattr(obj, "faces", None)
        if faces is not None and len(faces) > 0:
            kind = "mesh"
            pts = vertices if n_points is None else sample_triangles(vertices, np.asarray(faces), n_points, rng)
        else:
            pts = vertices

    pts = np.asarray(pts, dtype=np.float64)
    if kind == "points" and n_points is not None and n_points < len(pts):
        pts = pts[rng.choice(len(pts), size=n_points, replace=False)]
    if normalize:
        pts = normalize_unit(pts)
    return PointCloud(pts, labels=None, name=path.stem, meta={"source": str(path), "kind": kind})