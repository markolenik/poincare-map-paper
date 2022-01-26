"""Utilities for numerical integration of XPPAUT ODEs."""

import pandas as pd
from typing import List, Optional, Tuple, Dict
import multiprocessing
from functools import partial
import numpy as np
import pyxpp.pyxpp as xpp
from poincare_map import helpers


def run(xppfile: str, **kwargs) -> pd.DataFrame:
    """Run XPP ODEs and wrap result in a Pandas dataframe."""

    dat = xpp.run(xppfile, **kwargs)
    allvars = np.append("t", xpp.read_vars(xppfile))
    return pd.DataFrame(dat, columns=allvars)


def par_run(xppfile: str, nrand: int = 5, **kwargs) -> List[pd.DataFrame]:
    """Randomise ICs and integrate system multiple times."""

    with multiprocessing.get_context("fork").Pool() as pool:
        futures = []
        for idx in range(nrand):
            ics = randics(seed=idx)
            runfun = partial(run, uid=idx, ics=ics, **kwargs)
            future = pool.apply_async(runfun, [xppfile])
            futures.append(future)
        lcs = [future.get() for future in futures]
    return lcs


def nullclines(xppfile: str, gtot=0, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Compute nullclines."""

    # NOTE You can't modify the plot parameters (xhi, xlo etc).
    # That must be written directly into the xpp file.
    vnull, wnull = xpp.nullclines(xppfile, g=0.0, gtot=gtot, **kwargs)
    return vnull, wnull


def randics(seed=0):
    """Randomise initial conditions."""

    np.random.seed(seed)
    # FIXME: The dimensions etc should reflect the odes, fixex for now.
    # You have to initialise auxilliary variables as well (xpp).
    ics = np.concatenate(
        (np.random.uniform(-50.0, 40.0, 2), np.random.uniform(0.0, 1.0, 8))
    )
    return ics


# def spikes(sol: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
#     """Find spike indices of both cells using depression variable."""

#     id1 = helpers.events(sol.d1)
#     id2 = helpers.events(sol.d2)
#     return (id1, id2)

def spikes(sol: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Find spike indices of both cells using roots."""

    id1 = helpers.spikes(sol.v1)
    id2 = helpers.spikes(sol.v2)

    return id1, id2


def norm_lc(lc: pd.DataFrame) -> pd.DataFrame:
    """Normalise LC s.t. one cell fires its first spike at t=0."""

    period = lc.t.iloc[-1]
    s1, _ = spikes(lc)  # Spike indices.
    t1 = lc.t.iloc[s1].values  # Spike times.
    i1 = np.append(np.diff(t1), period - t1[-1])  # ISI.

    # Case 1-1.
    if len(t1) == 1 and t1[0] == 0:
        return lc
    else:
        long_isi_id = np.argmax(i1)
        if long_isi_id == (len(i1) - 1):  # Already normalised
            return lc
        else:
            # First spike should come after long ISI.
            first_spike_id = s1[(long_isi_id + 1)]
            from_first_spike = lc.iloc[first_spike_id:]
            to_first_spike = lc.iloc[:first_spike_id]
            new_lc = from_first_spike.append(to_first_spike).reset_index(
                drop=True
            )
            new_lc.t = np.linspace(0, max(lc.t), len(lc))
            return new_lc


def find_lc(sol: pd.DataFrame, norm: bool = True) -> Optional[pd.DataFrame]:
    """Find limit cycle using first return of depression variable."""

    ids = helpers.find_lc(sol.d1)
    if ids is not None:
        start = ids[0]
        end = ids[1]
    else:
        return None
    lc = sol.iloc[start:end].reset_index(drop=True)
    lc.t = lc.t - lc.t.iloc[0]
    if norm:
        return norm_lc(lc)
    else:
        return lc


def lc_info(lc) -> Dict:
    """Extract info from LC."""
    period = lc.t.iloc[-1]
    s1, _ = spikes(lc)
    n = len(s1)
    dmax = lc.d1.max()
    info = {
        "sol": lc,
        "n": n,
        "period": period,
        "dmax": dmax,
    }
    return info


def cont(
    xppfile: str,
    parstep: float = 0.01,
    parmin: float = 0,
    parmax: float = 1,
    nsteps: int = 100,
    nrand: int = 5,
    echo: bool = True,
    **kwargs,
) -> Optional[pd.DataFrame]:
    """Run pseudo continuation along g in one direction
    for period-n limit cycle solution.

    You can pass parameters, initial conditions or numerical
    options as kwargs.

    """
    par = "g"
    result = []
    # Use provided ICs to find the right solution branch.
    ics = kwargs.pop("ics", xpp.read_ics(xppfile))
    # Find initial solution.
    sol = run(xppfile, ics=ics, **kwargs)
    lc = find_lc(sol)

    if lc is not None:
        info = {**lc_info(lc), par: kwargs[par]}
        result.append(info)
        if echo:
            print(
                (
                    f"n: {info['n']}, par: {par} {round(kwargs[par], 4)},"
                    f" period: {info['period']}"
                )
            )
    else:
        return None

    # Perform a number of simulations in parallel.
    # Choose any LC with appropriate number of spikes.
    for _ in range(nsteps):
        # Update ICs and parameter.
        newpar = kwargs[par] + parstep
        if parmin < newpar < parmax:
            ics = lc.iloc[0, 1:].values
            kwargs[par] = newpar
            sols = par_run(xppfile, nrand=nrand, **kwargs)
            lcs = [find_lc(sol) for sol in sols]
            infos = [
                {**lc_info(lc), par: kwargs[par]}
                for lc in lcs
                if lc is not None
            ]
            # Pick any n-n limit cycle.
            ninfos = list(filter(lambda x: x["n"] == info["n"], infos))

            # Simulations yield correct type of LC.
            if ninfos:
                info = ninfos[0]
                result.append(info)
                print(
                    (
                        f"n: {info['n']}, par: {par} {round(kwargs[par], 4)}"
                        f", period: {info['period']}"
                    )
                )

            # Either a bifurcation or not enough randomisations.
            else:
                return pd.DataFrame(result)

        # Either a bifurcation or not enough randomisations.
        else:
            return pd.DataFrame(result)
    # Done with all steps.
    else:
        return pd.DataFrame(result)
