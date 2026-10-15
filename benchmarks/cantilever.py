"""Cantilever examples: paper Sec. 4.1 (one cubic element = exact beam theory) and a
geometrically nonlinear tip-load case compared with Mattiasson's elliptic-integral solution.

Usage: python benchmarks/cantilever.py [--out results]
"""

import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from iga_kl import geometry  # noqa: E402
from iga_kl.constraints import clamp  # noqa: E402
from iga_kl.material import Material  # noqa: E402
from iga_kl.model import ShellModel  # noqa: E402
from iga_kl.postprocess import plot_control_net, plot_shell, set_axes_equal  # noqa: E402

# Mattiasson (1981), cantilever with tip point load: PL^2/EI -> (w/L, u/L)
MATTIASSON = {1.0: (0.30172, 0.05643), 2.0: (0.49346, 0.16064), 5.0: (0.71379, 0.38763), 10.0: (0.81061, 0.55500)}


def _displaced_surface(surf, u):
    s = surf.copy()
    for i in range(surf.n):
        for j in range(surf.m):
            cp = surf.cp_index(i, j)
            s.Pw[i, j, :3] += u[3 * cp : 3 * cp + 3] * s.Pw[i, j, 3]
    return s


def linear_fig4(out):
    L, b, t, E, P = 20.0, 1.0, 0.1, 1.0e6, 1.0
    surf = geometry.plate(L, b).k_refine(3, 1)  # one cubic element (paper Fig. 4)
    model = ShellModel(surf, Material(E, 0.0, t))
    clamp(model.constraints, surf, "xi0", 2)
    model.add_edge_load("xi1", [0.0, 0.0, -P / b])
    u = model.solve_linear()
    w = -model.displacement_at(u, 1.0, 0.5)[2]
    w_beam = P * L**3 / (3 * E * b * t**3 / 12)
    print(f"Sec. 4.1 cantilever, one cubic element: w_tip = {w:.8g}, beam theory PL^3/3EI = {w_beam:.8g}")
    fig = plt.figure(figsize=(10, 4))
    for k, (uu, ttl) in enumerate([(None, "original geometry"), (u, "deformed geometry")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        plot_shell(ax, surf, uu, 1.0, n=(41, 5), wire=False)
        plot_control_net(ax, surf if uu is None else _displaced_surface(surf, uu))
        set_axes_equal(ax)
        ax.set_title(ttl)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "cantilever_linear.png"), dpi=150)
    plt.close(fig)
    return w, w_beam


def nonlinear_tip_load(out):
    L, b, t, E = 10.0, 1.0, 0.1, 1.2e4
    I = b * t**3 / 12
    P = 10.0 * E * I / L**2
    surf = geometry.plate(L, b).k_refine(4, 8, 1)
    print("\nLarge-deflection cantilever, tip load (p=4, 8 elements), Newton with 10 load steps:")
    print(f"{'PL^2/EI':>8} {'w/L':>9} {'u/L':>9} {'Mattiasson w/L':>15} {'u/L':>9} {'iters':>6}")
    states, u = {}, None
    for k in range(1, 11):
        model = ShellModel(surf, Material(E, 0.0, t))
        clamp(model.constraints, surf, "xi0", 2)
        model.add_edge_load("xi1", [0.0, 0.0, -k * P / 10 / b])
        u, h = model.solve_nonlinear(n_steps=1, tol=1e-9, u0=u)
        states[k] = u
        tip = model.displacement_at(u, 1.0, 0.5)
        ref = MATTIASSON.get(float(k), (float("nan"), float("nan")))
        print(f"{k:8d} {-tip[2] / L:9.5f} {-tip[0] / L:9.5f} {ref[0]:15.5f} {ref[1]:9.5f} {len(h[0]) - 1:6d}")
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    plot_shell(ax, surf, None, n=(41, 5), wire=False, color="lightgray")
    for k in [2, 5, 10]:
        plot_shell(ax, surf, states[k], 1.0, n=(41, 5), wire=False)
    set_axes_equal(ax)
    ax.view_init(elev=15, azim=-80)
    ax.set_title("cantilever under tip load, PL^2/EI = 0, 2, 5, 10")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "cantilever_nonlinear.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    linear_fig4(args.out)
    nonlinear_tip_load(args.out)


if __name__ == "__main__":
    main()
