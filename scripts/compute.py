"""Script to compute numeric and analytic bifurcation diagrams."""

from __future__ import annotations

import time
import argparse
import multiprocessing

import pandas as pd

from poincare_map import ode, paths, poincare_map


def compute_numeric_bif_diagram(dt: float=0.5) -> pd.DataFrame:
    """Compute bifurcation diagram numerically for n=1,2,3,4,5."""
    mlml = ode.MLML("odes/mlml.ode")
    nsteps = 100

    gmin = 0.01
    ns = [1, 2, 3, 4, 5]
    g0s = [0.35, 0.4, 0.5, 0.52, 0.56]
    gsteps = [0.002, 0.002, 0.002, 0.001, 0.001]
    totals = [5000, 8000, 10000, 15000, 15000]

    curves = []
    for n, g0, step, total in zip(ns, g0s, gsteps, totals):
        print(f"Starting branch at g={g0}")
        left = mlml.continuate(
            n=n, step=-step, nsteps=nsteps, g=g0, total=total, gmin=gmin, dt=dt
        )
        right = mlml.continuate(
            n=n, step=step, nsteps=nsteps, g=g0, total=total, gmin=gmin, dt=dt
        )
        if (left is not None) and (right is not None):
            curve = pd.concat([left.iloc[::-1], right]).reset_index(
                drop=True
            )
            curves.append(curve)
    return pd.concat(curves, axis=0)  # type: ignore


def compute_analytic_bif_diagram() -> pd.DataFrame:
    """Compute bifurcation diagram analytically."""
    ns = list(range(10))[1:]
    with multiprocessing.get_context("fork").Pool() as pool:
        futures = []
        for n in ns:
            poincare = poincare_map.PoincareMap(n)
            future = pool.apply_async(poincare.compute_stable_branch)
            futures.append(future)
        branches = [future.get() for future in futures]
        return pd.concat(branches)  # type: ignore




parser = argparse.ArgumentParser()
parser.add_argument(
    "--recompute", help="Whether to recompute diagrams", action="store_true"
)

if __name__ == "__main__":

    args = parser.parse_args()

    t0 = time.time()
    if (not paths.numeric_bif_diagram.exists()) or args.recompute:
        print(
            "Computing numeric bifurcation diagram. This might take a while ..."
        )
        numeric_diagram = compute_numeric_bif_diagram()
        numeric_diagram.to_pickle(paths.numeric_bif_diagram)

    # if (not paths.analytic_bif_diagram.exists()) or args.recompute:
    #     print("Computing analytic bifurcation diagram.")
    #     analytic_diagram = compute_analytic_bif_diagram()
    #     analytic_diagram.to_pickle(paths.analytic_bif_diagram)

    print("Done!")
    print(f"Computation took {time.time() - t0} seconds.")
