"""Plotting helpers (matplotlib imported lazily; polyscope optional)."""
from __future__ import annotations

import numpy as np


def plot_field(points, values, ax=None, *, source=None, band=None, cmap="viridis",
               vmin=None, vmax=None, s=4.0, title=None, colorbar=True):
    """Scatter plot of a scalar field on a 2D or 3D point cloud.

    band=(lo, hi) shows only the points whose value is in [lo, hi] (an isolevel), the rest in gray.
    Non-finite values (unreachable points) are drawn in gray as well.
    """
    import matplotlib.pyplot as plt

    pts = np.asarray(points, dtype=float)
    vals = np.asarray(values, dtype=float)
    dim = pts.shape[1]
    if dim not in (2, 3):
        raise ValueError("plot_field supports 2D or 3D points: project first")
    if ax is None:
        ax = plt.figure(figsize=(5, 5)).add_subplot(projection="3d" if dim == 3 else None)

    finite = np.isfinite(vals)
    if band is not None:
        vmin, vmax = band
        inside = finite & (vals >= vmin) & (vals <= vmax)
    else:
        inside = finite
        if finite.any():
            vmin = vals[finite].min() if vmin is None else vmin
            vmax = vals[finite].max() if vmax is None else vmax

    if (~inside).any():
        ax.scatter(*pts[~inside].T, s=s, c="lightgray", alpha=0.25, linewidths=0)
    if inside.any():
        sc = ax.scatter(*pts[inside].T, s=s, c=vals[inside], cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
        if colorbar:
            ax.figure.colorbar(sc, ax=ax, shrink=0.7)
    if source is not None:
        ax.scatter(*pts[source:source + 1].T, s=60, c="limegreen", edgecolors="k", zorder=10)

    if dim == 2:
        ax.set_aspect("equal")
    else:
        ax.set_box_aspect(np.maximum(np.ptp(pts, axis=0), 1e-9))
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return ax


def plot_distance_field(points, distances, source=None, **kwargs):
    """Distance to `source`, with the source marked as a green dot."""
    return plot_field(points, distances, source=source, **kwargs)


def plot_distance_isolevels(points, distances, source, levels, width, ncols=4, **kwargs):
    """One panel per isolevel: shows the points with |d - level| <= width."""
    import matplotlib.pyplot as plt

    dim = np.asarray(points).shape[1]
    nrows = -(-len(levels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False,
                             subplot_kw={"projection": "3d"} if dim == 3 else {})
    for ax, lvl in zip(axes.ravel(), levels):
        plot_field(points, distances, ax=ax, source=source, band=(lvl - width, lvl + width),
                   title=f"d = {lvl:.3g}", colorbar=False, **kwargs)
    for ax in axes.ravel()[len(levels):]:
        ax.set_visible(False)
    return fig

def plot_path(points, path, source=None, target=None, **kwargs):
    """Scatter plot of a path on a 2D or 3D point cloud.

    `path` is a sequence of indices into `points`.
    All points that are not part of the path are drawn in light gray.
    All points in the path are colored by their index in the path to the source (or target if source is None).
    """
    import matplotlib.pyplot as plt

    pts = np.asarray(points, dtype=float)
    path = np.asarray(path, dtype=int)
    dim = pts.shape[1]
    if dim not in (2, 3):
        raise ValueError("plot_path supports 2D or 3D points: project first")
    ax = plt.figure(figsize=(5, 5)).add_subplot(projection="3d" if dim == 3 else None)

    all_idx = np.arange(len(pts))
    outside = ~np.isin(all_idx, path)
    if outside.any():
        ax.scatter(*pts[outside].T, s=4.0, c="lightgray", alpha=0.25, linewidths=0)
    if len(path) > 0:
        sc = ax.scatter(*pts[path].T, s=20.0, c=np.arange(len(path)), cmap="plasma", linewidths=0)
        ax.plot(*pts[path].T, c="k", alpha=0.5, lw=1.0)
        ax.figure.colorbar(sc, ax=ax, shrink=0.7)

    if source is not None:
        ax.scatter(*pts[source:source + 1].T, s=60, c="limegreen", edgecolors="k", zorder=10)
    if target is not None:
        ax.scatter(*pts[target:target + 1].T, s=60, c="orangered", edgecolors="k", zorder=10)

    if dim == 2:
        ax.set_aspect("equal")
    else:
        ax.set_box_aspect(np.maximum(np.ptp(pts, axis=0), 1e-9))
    ax.set_axis_off()
    return ax
    

def show_polyscope(points, **scalars):
    """Interactive view: show_polyscope(X, distance=d, det=gm.determinants). Requires polyscope."""
    import polyscope as ps

    ps.init()
    cloud = ps.register_point_cloud("points", np.asarray(points, dtype=float))
    for name, v in scalars.items():
        v = np.asarray(v, dtype=float)
        v = np.where(np.isfinite(v), v, v[np.isfinite(v)].max())   # polyscope dislikes inf
        cloud.add_scalar_quantity(name, v)
    ps.show()