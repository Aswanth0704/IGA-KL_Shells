"""Run the shell obstacle course for p = 2..5 and 2..32 elements per edge (paper Figs. 11-13).

Usage:  python benchmarks/run_obstacle_course.py [--problems scordelis_lo,pinched_cylinder,hemisphere]
                                                  [--degrees 2,3,4,5] [--elements 2,4,8,16,32] [--out results]
"""

import argparse
import csv
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from iga_kl.benchmarks import BUILDERS, REFERENCE, TITLES  # noqa: E402
from iga_kl.postprocess import plot_convergence  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="scordelis_lo,pinched_cylinder,hemisphere")
    ap.add_argument("--degrees", default="2,3,4,5")
    ap.add_argument("--elements", default="2,4,8,16,32")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    degrees = [int(s) for s in args.degrees.split(",")]
    elements = [int(s) for s in args.elements.split(",")]
    rows = []
    for name in args.problems.split(","):
        results = {}
        print(f"\n{TITLES[name]}  (paper converged {REFERENCE[name]['paper']:.5g}, reference {REFERENCE[name]['belytschko']:.5g})")
        print(f"{'p':>3} {'n_el':>5} {'n_cp':>5} {'n_dof':>6} {'value':>14} {'ratio':>8} {'time [s]':>9}")
        for p in degrees:
            results[p] = []
            for n_el in elements:
                t0 = time.time()
                bm = BUILDERS[name](p, n_el)
                val = bm.solve()
                dt = time.time() - t0
                results[p].append((bm.n_cp_side, val))
                rows.append([name, p, n_el, bm.n_cp_side, bm.n_dof, val, dt])
                print(f"{p:3d} {n_el:5d} {bm.n_cp_side:5d} {bm.n_dof:6d} {val:14.6e} {val / REFERENCE[name]['paper']:8.4f} {dt:9.2f}")
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_convergence(ax, results, REFERENCE[name]["paper"], TITLES[name])
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"{name}_convergence.png"), dpi=150)
        plt.close(fig)
    with open(os.path.join(args.out, "obstacle_course.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "p", "n_el", "n_cp_side", "n_dof", "value", "time_s"])
        w.writerows(rows)
    print(f"\nwrote {args.out}/obstacle_course.csv and convergence plots")


if __name__ == "__main__":
    main()
