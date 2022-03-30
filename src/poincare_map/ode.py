"""Utilities for numerical integration of XPPAUT ODEs."""
from __future__ import annotations

import multiprocessing
import typing as tp
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pyxpp.pyxpp as xpp
from scipy import optimize

from poincare_map import helpers


class ODE(ABC):
    ode_file: str
    # State variables.
    state_vars: tp.List[str]
    # Model parameters.
    pars: tp.Dict[str, float]
    # Initial conditions.
    ics: pd.DataFrame
    # Numerical options.
    numerics: tp.Dict[str, float]

    def __init__(self, ode_file: str | Path) -> None:
        """Initialise class from ode file."""
        self.ode_file = str(ode_file)
        self.state_vars = list(xpp.read_vars(ode_file))
        self.pars = dict(xpp.read_pars(ode_file))
        self.ics = pd.DataFrame(
            [xpp.read_ics(ode_file)], columns=self.state_vars
        )
        self.numerics = dict(xpp.read_opts(ode_file))

    def run(
        self,
        ics: pd.DataFrame | tp.Iterable | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Run XPP ODEs and wrap result in a Pandas dataframe."""
        # Convert ics to vector format.
        if ics is not None:
            ics_vector = np.array(ics)
        else:
            ics_vector = ics
        sol = xpp.run(self.ode_file, ics=ics_vector, **kwargs)
        index = pd.Index(data=sol[:, 0], name="time")
        return pd.DataFrame(sol[:, 1:], columns=self.state_vars, index=index)

    def parallel_rand_run(
        self, nrand: int = 5, **kwargs
    ) -> tp.List[pd.DataFrame]:
        """Randomise ICs and integrate system multiple times."""

        with multiprocessing.get_context("fork").Pool() as pool:
            futures = []
            # First result starts with default ics.
            runfun = partial(self.run, ics=self.ics, uid=0, **kwargs)
            future = pool.apply_async(runfun)
            futures.append(future)
            for idx in range(nrand)[1:]:
                ics = self.rand_ics(seed=idx)
                runfun = partial(self.run, ics=ics, uid=idx, **kwargs)
                future = pool.apply_async(runfun)
                futures.append(future)
            sols = [future.get() for future in futures]
        return sols

    @abstractmethod
    def rand_ics(self, seed: int | None = None) -> pd.DataFrame:
        """Numerically compute fixed point of nullclines."""
        pass

    @abstractmethod
    def find_limit_cycle(self, **kwargs) -> pd.DataFrame:
        """Integrate and return either limit cycle solution or empty dataframe."""
        pass


class ML(ODE):
    def rand_ics(self, seed: int | None = None) -> pd.DataFrame:
        """Randomise initial conditions."""

        np.random.seed(seed)
        # You have to initialise auxilliary variables as well (xpp).
        ics = np.concatenate(
            (np.random.uniform(-50.0, 40.0, 2), np.random.uniform(0.0, 1.0, 8))
        )
        return pd.DataFrame([ics], columns=self.state_vars)

    def find_limit_cycle(self, thresh: float = 2, **kwargs) -> pd.DataFrame:
        """Integrate and return either limit cycle or stable fixed point."""
        solution = self.run(**kwargs)
        if not solution.empty:
            spikes = helpers.spike_times(solution["v"])
            if spikes.empty:
                # If no limit can be found return empty DataFrame.
                return pd.DataFrame(columns=solution.columns)
            else:
                lc_index = helpers.limit_cycle_index(
                    solution["v"], spikes, thresh=thresh
                )
                lc = solution.loc[lc_index]
                # Make lc cyclic
                lc = pd.DataFrame(
                    np.vstack([lc.values, lc.iloc[0].values]),
                    columns=lc.columns,
                )
                # Reset time index
                period = lc.index[-1] - lc.index[0]
                time = np.linspace(0, period, len(lc))
                return lc.set_index(pd.Index(time, name=lc.index.name))
        # If no limit can be found return empty DataFrame.
        return pd.DataFrame(columns=solution.columns)

    def minf(self, v: float) -> float:
        """Calcium asymptotic function."""
        return helpers.sigmoid(v, self.pars["va"], self.pars["vb"])

    def winf(self, v: float) -> float:
        """Potassium nullcline function."""
        return helpers.sigmoid(v, self.pars["vc"], self.pars["vd"])

    def vinf(self, v: float, gtot: float = 0) -> float:
        """V-nullcline."""
        I_leak = self.pars["gl"] * (v - self.pars["vl"])
        I_ca = self.pars["gca"] * self.minf(v) * (v - self.pars["vca"])
        I_syn = gtot * (v - self.pars["vs"])
        I_k = self.pars["gk"] * (v - self.pars["vk"])
        return (-I_leak - I_ca - I_syn + self.pars["iapp"]) / I_k

    def fixed_point(self, gtot: float = 0.0, x0: float = -69) -> float | None:
        """Numerically compute fixed point of nullclines."""
        root_fun = lambda v: self.vinf(v, gtot) - self.winf(v)
        root, info = optimize.newton(root_fun, x0=x0, full_output=True)
        if info.converged:
            return root

    def find_HB(
        self,
        g0: float = 0.007,
        g_step: float = -1e-4,
        max_steps: int = 1000,
        echo_progress: bool = False,
        **kwargs,
    ) -> float | None:
        """Find value of g at Andronov-Hopf bifurcation using brute-force.
        This assumes that we start the procedure at a stable fixed point."""
        g = g0
        for _ in range(max_steps):
            if echo_progress:
                print(f"g={round(g, 4)}")
            equilibrium = self.find_limit_cycle(g=g, **kwargs)
            if type(equilibrium) is pd.DataFrame:  # limit cycle
                return g
            else:
                g = g + g_step
        else:
            return None

    def find_left_knee(self, gtot: float) -> float | None:
        """Find left knee of v-nullcline."""
        result = optimize.minimize(
            partial(self.vinf, gtot=gtot), method="Nelder-Mead", x0=-60
        )
        if result.success:
            return result.x[0]


class MLML(ODE):
    """Two-cell Morris-Lecar model."""

    def rand_ics(self, seed: int | None = None) -> pd.DataFrame:
        """Randomise initial conditions."""

        np.random.seed(seed)
        # You have to initialise auxilliary variables as well (xpp).
        ics = np.concatenate(
            (np.random.uniform(-50.0, 40.0, 2), np.random.uniform(0.0, 1.0, 8))
        )
        return pd.DataFrame([ics], columns=self.state_vars)

    def find_limit_cycle(
        self, n: int, nrand: int = 5, thresh: float = 1e-3, **kwargs
    ) -> pd.DataFrame:
        """Randomise ICs to guess a n-period limit cycle."""
        # Compute a bunch of solutions.
        def normalise_limit_cycle(lc: pd.DataFrame) -> pd.DataFrame:
            """Normalise limit cycle s.t. cell 1 spikes its first spike at t=0."""
            t0 = lc.index[lc["d1"].argmax()]
            # Append remaining trajectory to end.
            lc_before = lc.loc[lc.index < t0]
            lc_after = lc.loc[lc.index >= t0]
            lc_normed = pd.concat([lc_after, lc_before])
            # Recompute index.
            period = lc.index[-1] - lc.index[0]
            time = np.linspace(0, period, len(lc))
            return lc_normed.set_index(pd.Index(time, name=lc.index.name))

        solutions = self.parallel_rand_run(nrand=nrand, **kwargs)
        for sol in solutions:
            spikes = helpers.spike_times(sol["v1"])
            lc_index = helpers.limit_cycle_index(
                sol["d1"], spikes, thresh=thresh
            )
            lc = sol.loc[lc_index]
            if not lc.empty:
                lc_normed = normalise_limit_cycle(lc)
                info = helpers.limit_cycle_info(lc_normed)
                if info["n"] == n:
                    return lc_normed
        # If no limit can be found return empty DataFrame.
        return pd.DataFrame(columns=solutions[0].columns)

    def continuate(
        self,
        n: int,
        step: float = 0.01,
        gmin: float = 0.01,
        gmax: float = 0.6,
        nsteps: int = 100,
        nrand: int = 5,
        echo_progress: bool = True,
        thresh: float = 1e-3,
        **kwargs,
    ) -> pd.DataFrame:
        """Run pseudo continuation along g in one direction along a stable n-branch.

        You can pass parameters, initial conditions or numerical
        options as kwargs.

        """

        def print_continuation_step_info(info) -> None:
            print_string = (
                f"n: {info['n']}, g={round(g, 4)}," f" period: {info['period']}"
            )
            print(print_string)

        # Parameter value is either read from file or given as kwarg.
        g = kwargs.pop("g", self.pars["g"])
        results = []

        # Run continuation in some direction
        for _ in range(nsteps):
            # import pdb; pdb.set_trace()
            if gmin < g < gmax:
                # Try finding a n-limit cycle
                lc = self.find_limit_cycle(
                    n=n, nrand=nrand, thresh=thresh, g=g, **kwargs
                )
                if not lc.empty:
                    lc_info = helpers.limit_cycle_info(lc)
                    if lc_info["n"] == n:
                        results.append(
                            {
                                **lc_info,
                                **{"g": g, "lc": lc},
                            }
                        )
                        if echo_progress:
                            print_continuation_step_info(lc_info)
                    else:
                        return pd.DataFrame(results)
                # Either a bifurcation or not enough randomisations.
                else:
                    return pd.DataFrame(results)
            else:
                return pd.DataFrame(results)
            # Update ICs and parameter.
            g = g + step
        # Done with all steps.
        else:
            return pd.DataFrame(results)
