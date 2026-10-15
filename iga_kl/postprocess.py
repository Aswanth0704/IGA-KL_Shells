"""Sampling and matplotlib plotting of NURBS shells and convergence curves."""

import numpy as np


def sample_surface(surface, u=None, scale=1.0, n=(31, 31)):
    """Grid of surface points X (and X + scale * u if u is given): arrays (n_xi, n_eta, 3)."""
    xis = np.linspace(0, 1, n[0])
    etas = np.linspace(0, 1, n[1])
    X = np.zeros((n[0], n[1], 3))
    for i, xi in enumerate(xis):
        for j, eta in enumerate(etas):
            X[i, j] = surface.point(xi, eta)
            if u is not None:
                X[i, j] += scale * surface.displacement_at(u, xi, eta)
    return X


def mirror(X, axes, sign=None):
    """Mirror a sampled grid through coordinate planes (for assembling symmetric models).

    Returns a list of grids: the original and every mirrored copy.
    """
    grids = [X]
    for ax in axes:
        new = []
        for g in grids:
            m = g.copy()
            m[..., ax] *= -1.0
            new.append(m)
        grids += new
    return grids


def plot_shell(ax, surface, u=None, scale=1.0, n=(31, 31), color="tab:green", alpha=1.0, wire=True, mirror_axes=()):
    X = sample_surface(surface, u, scale, n)
    for g in mirror(X, mirror_axes):
        ax.plot_surface(g[..., 0], g[..., 1], g[..., 2], color=color, alpha=alpha, linewidth=0, antialiased=True, shade=True)
        if wire:
            step = max(1, n[0] // 10)
            for k in range(0, n[0], step):
                ax.plot(g[k, :, 0], g[k, :, 1], g[k, :, 2], color="k", linewidth=0.3)
            for k in range(0, n[1], step):
                ax.plot(g[:, k, 0], g[:, k, 1], g[:, k, 2], color="k", linewidth=0.3)
    return X


def plot_control_net(ax, surface, color="tab:red"):
    P = surface.control_points
    for i in range(surface.n):
        ax.plot(P[i, :, 0], P[i, :, 1], P[i, :, 2], "--", color=color, linewidth=0.6)
    for j in range(surface.m):
        ax.plot(P[:, j, 0], P[:, j, 1], P[:, j, 2], "--", color=color, linewidth=0.6)
    ax.scatter(P[..., 0], P[..., 1], P[..., 2], color=color, s=6)


def set_axes_equal(ax):
    lims = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centre = lims.mean(axis=1)
    r = 0.5 * (lims[:, 1] - lims[:, 0]).max()
    ax.set_xlim3d(centre[0] - r, centre[0] + r)
    ax.set_ylim3d(centre[1] - r, centre[1] + r)
    ax.set_zlim3d(centre[2] - r, centre[2] + r)


def plot_convergence(ax, results, reference, title, ylabel="u"):
    """results: {p: [(n_cp_side, value), ...]}; reference: float."""
    markers = {2: "D", 3: "s", 4: "^", 5: "x"}
    for p in sorted(results):
        pts = sorted(results[p])
        ax.plot([a for a, _ in pts], [b for _, b in pts], marker=markers.get(p, "o"), markersize=4, label=f"p={p}")
    ax.axhline(reference, color="purple", linewidth=1, label="reference")
    ax.set_xlabel("number of control points per side")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0, 40)
    ax.legend()
    ax.grid(alpha=0.3)
