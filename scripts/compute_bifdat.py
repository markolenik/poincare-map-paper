"""Script to numerically compute bifurcation diagram."""

import argparse
import pickle

import pandas as pd

from poincare_map import ode


def compute_bif_diagram(xppfile: str, echo: bool = True) -> pd.DataFrame:
    nsteps = 100
    p0s = [0.3, 0.4, 0.5, 0.52, 0.56]
    parsteps = [0.01, 0.01, 0.005, 0.001, 0.001]
    totals = [5000, 8000, 10000, 15000, 15000]
    curves = []
    for p0, step, total in zip(p0s, parsteps, totals):
        if echo:
            print(f"Initialising parameter at {p0}")
        left = ode.cont(
            xppfile, g=p0, parstep=-step, nsteps=nsteps, total=total
        )
        right = ode.cont(
            xppfile, g=p0, parstep=step, nsteps=nsteps, total=total
        )
        if (left is not None) and (right is not None):
            curve = (
                left.iloc[::-1].append(right.iloc[1:]).reset_index(drop=True)
            )
            curves.append(curve)
        else:
            curves.append(None)
    return pd.concat(curves)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Output file", type=str)
    parser.add_argument(
        "--xppfile", help="XPP ODE file", default="mlml.ode", type=str
    )
    args = parser.parse_args()

    dat = compute_bif_diagram(args.xppfile)
    pickle.dump(dat, open(args.output, "wb"))
