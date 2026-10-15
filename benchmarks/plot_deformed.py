"""Deformed-shape plots of the three obstacle course problems (cf. paper Figs. 14-15).

Usage: python benchmarks/plot_deformed.py [--p 3] [--n_el 16] [--out results]
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from iga_kl.benchmarks import BUILDERS, TITLES  # noqa: E402
from iga_kl.postprocess import plot_shell, set_axes_equal  # noqa: E402

# (mirror axes to assemble the full structure, displacement scale factor)
SETTINGS = {
    "scordelis_lo": ((0, 2), 10.0),
    "pinched_cylinder": ((0, 1, 2), 3.0e6),
    "hemisphere": ((0, 1), 20.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--n_el", type=int, default=16)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name, (axes, scale) in SETTINGS.items():
        bm = BUILDERS[name](args.p, args.n_el)
        val = bm.solve()
        surf = bm.model.surface
        # the symmetric models are mirrored about the symmetry planes; for the cylinders the
        # z-mirror must be about z = L/2, so shift before mirroring
        fig = plt.figure(figsize=(11, 5))
        for k, (u, ttl) in enumerate([(None, "initial configuration"), (bm.u, f"deformed (scale {scale:g})")]):
            ax = fig.add_subplot(1, 2, k + 1, projection="3d")
            if name in ("scordelis_lo", "pinched_cylinder"):
                _plot_cylinder_full(ax, surf, u, scale, axes)
            else:
                plot_shell(ax, surf, u, scale, mirror_axes=axes)
            set_axes_equal(ax)
            ax.set_title(ttl)
        fig.suptitle(f"{TITLES[name]}: p={args.p}, {args.n_el}x{args.n_el} elements, value = {val:.5g}")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"{name}_deformed.png"), dpi=150)
        plt.close(fig)
        print(f"{name}: {val:.6g}")


def _plot_cylinder_full(ax, surf, u, scale, axes):
    import numpy as np

    from iga_kl.postprocess import mirror, sample_surface

    X = sample_surface(surf, u, scale)
    zmax = surf.control_points[..., 2].max()
    X[..., 2] -= zmax  # symmetry plane z = L/2 -> z = 0
    for g in mirror(X, axes):
        ax.plot_surface(g[..., 0], g[..., 1], g[..., 2], color="tab:green", linewidth=0, antialiased=True)
        for k in range(0, g.shape[0], 3):
            ax.plot(g[k, :, 0], g[k, :, 1], g[k, :, 2], color="k", linewidth=0.3)
        for k in range(0, g.shape[1], 3):
            ax.plot(g[:, k, 0], g[:, k, 1], g[:, k, 2], color="k", linewidth=0.3)


if __name__ == "__main__":
    main()
