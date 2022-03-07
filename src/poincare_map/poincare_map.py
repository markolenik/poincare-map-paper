"""Poincare map definition with methods.

Module containing the most important model parameters and derived
quantities and functions.
"""
from __future__ import annotations

from typing import NamedTuple, Tuple

import numpy as np
import pandas as pd
from pynverse import inversefunc
from scipy import optimize

# Ignore divide-by-zero errors that can occur for minisation for extreme values.
# np.seterr(divide='ignore')


class PoincareMap(NamedTuple):
    """Model for map computation."""

    n: int = 2

    # These parameters should match the ODE file.
    tauk: float = 100
    taua: float = 1000
    taub: float = 100

    # These are derived.
    gstar: float = 0.0068
    T: float = 376  # Spiking period.
    Tact: float = 49
    Tinact: float = 327

    @property
    def Lambda(self):
        """Depression speed."""
        return np.exp(-self.Tact / self.taub)

    @property
    def Rho(self):
        """Depression recovery speed."""
        return np.exp(-self.Tinact / self.taua)

    @property
    def dsup(self):
        """Value of d for suppressed solution."""
        return (1 - self.Rho) / (1 - self.Lambda * self.Rho)

    @property
    def gsup(self):
        """Minimum value of g for suppressed solution."""
        return (self.gstar / self.Lambda) * (
            np.exp(self.Tinact / self.tauk) / self.dsup
        )

    @property
    def tau(self):
        """Overall time constant."""
        return 2 * self.tauk / self.taua

    @property
    def alpha(self) -> float:
        return pow(self.Lambda * self.Rho, self.n - 1)

    @property
    def beta(self) -> float:
        return (1 - self.Rho) * sum(
            [pow(self.Lambda * self.Rho, k) for k in range(int(self.n - 1))]
        )

    def delta(self, d: float) -> float:
        """Solution to d-map after n spikes, which is the value of depression
        variable at nth spike time."""
        return self.alpha * d + self.beta

    def d_asymptote(self) -> float:
        """Leftmost bound of P-map domain."""
        return -self.beta / self.alpha

    def F_map(self, d: float, g: float) -> float:
        """Free phase map F."""
        return self.tauk * np.log(
            (g / self.gstar) * self.Lambda * self.delta(d)
        )

    def Q_map(self, delta_t: float, g: float) -> float:
        """Quiet phase map Q."""
        delta_n = (
            (1 / (g * self.Lambda)) * self.gstar * np.exp(delta_t / self.tauk)
        )
        exponent = (
            -((self.n - 1) * self.T + self.Tact + 2 * delta_t) / self.taua
        )
        return 1 - (1 - self.Lambda * delta_n) * np.exp(exponent)

    def P_map(self, d: float, g: float) -> float:
        """Explicit burstmap."""
        return self.Q_map(self.F_map(d, g), g)

    # # NOTE: This is correct!
    # def P_map_direct(self, d, n, g):
    #     delta = self.delta(d, n)
    #     A = 1 - delta * self.Lambda
    #     B = pow((self.g / self.gstar) * self.Lambda * delta, -self.tau)
    #     C = np.exp(-((n - 1) * self.T + self.Tact) / self.taua)
    #     return 1 - A * B * C

    def G(self, d: float) -> float:
        """Fixed point function maps d to the corresponding fixed point value gbar."""
        delta = self.delta(d)
        A = (1 - delta * self.Lambda) / (1 - d)
        B = np.exp(-((self.n - 1) * self.T + self.Tact) / self.taua)
        return (self.gstar / (delta * self.Lambda)) * pow(A * B, 1 / self.tau)

    def period(self, g: float) -> float | None:
        """Compute n-n period (eq. 45)."""
        df = self.phi_stable(g)
        if df is not None:
            delta_t = self.F_map(df, g)
            return 2 * ((self.n - 1) * self.T + self.Tact + delta_t)

    def critical_fp(self, x0: float = 0.5) -> Tuple:
        """Compute critical fixed point (db, gb) at fold bifurcation
        (equations 41 and 42).
        """
        fun = lambda d: self.G(d)
        # res = optimize.minimize(fun, x0=x0, method="Nelder-Mead")
        res = optimize.minimize(fun, x0=x0, method="TNC")
        d_min = res.x[0]
        g_min = res.fun
        return d_min, g_min

    def phi_stable(self, g: float) -> float | None:
        """Find stable fixed point dstar."""
        f = lambda d: self.G(d)
        dmin, gmin = self.critical_fp()
        if g > gmin:
            return float(
                inversefunc(f, g, domain=[dmin, 1], open_domain=True)  # type: ignore
            )

    def phi_unstable(self, g: float) -> float | None:
        """Find unstable fixed point dstar."""
        f = lambda d: self.G(d)
        dmin, gmin = self.critical_fp()
        da = self.d_asymptote()
        if g > gmin:
            return float(
                inversefunc(f, g, domain=[da, dmin], open_domain=True)  # type: ignore
            )

    def L_fun(self, g: float) -> float | None:
        """Compute L function (eq. 58)."""
        df = self.phi_stable(g)
        if df is not None:
            # Value of d at (n-1)T + Tact
            dlast = self._replace(n=self.n - 1).delta(df) * self.Lambda
            return g * dlast * np.exp(-self.Tinact / self.tauk) - self.gstar

    def R_fun(self, g: float) -> float | None:
        """Compute R function (eq. 56)."""
        df = self.phi_stable(g)
        if df is not None:
            return self.F_map(df, g) - self.Tinact

    def left_branch_border(self, x0: float = 1.0) -> float | None:
        """Compute left bifurcation border of n-branch."""
        f = lambda g: self.L_fun(g)
        root, info = optimize.newton(f, x0, full_output=True)
        if info.converged:
            return root

    def right_branch_border(self, x0: float = 1.0) -> float | None:
        """Compute right bifurcation border of n-branch."""
        f = lambda g: self.R_fun(g)
        root, info = optimize.newton(f, x0, full_output=True)
        if info.converged:
            return root

    def compute_stable_branch(self, x0: float = 1.0) -> pd.DataFrame:
        """Compute stable branch of bifurcation diagram for map Pi."""
        # Compute branch borders.
        if self.n > 1:
            left_border = self.left_branch_border(x0)
        else:
            left_border = 0
        right_border = self.right_branch_border()

        if (left_border is not None) and (right_border is not None):
            # Compute cycle period.
            gs = np.linspace(left_border, right_border, 1000)
            # Find stable fixed points with corresponding cycle period.
            dfs = []
            periods = []
            for g in gs:
                dfs.append(self.phi_stable(g))
                periods.append(self.period(g))

            return pd.DataFrame(
                {
                    "g": gs,
                    "period": periods,
                    "dstar": dfs,
                    "n": self.n,
                }
            )
        else:
            return pd.DataFrame()
