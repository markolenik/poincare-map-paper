"""Poincare map definition with methods.

Module containing the most important model parameters and derived
quantities and functions.
"""

from typing import NamedTuple, Sequence, Optional, Tuple

import numpy as np
from scipy import optimize
from pandas import DataFrame
from pynverse import inversefunc


class Pars(NamedTuple):
    """Map parameters."""

    g: float = 0
    gstar: float = 0.0036
    period: float = 375  # Spiking period.
    tauk: float = 100


    @property
    def rho(self):
        return np.exp(-self.period / self.taud)

    @property
    def d_sup(self):
        return (1 - self.rho) / (1 - self.Lambda * self.rho)

    @property
    def g_sup(self):
        return self.gstar * np.exp(self.period / self.taus) / self.d_sup

    @property
    def tau(self):
        return 2 * self.taus / self.taud


## Convenience functions


def alpha(pars: Pars, n: int) -> float:
    return pow(pars.Lambda * pars.rho, n - 1)


def beta(pars: Pars, n: int) -> float:
    return (1 - pars.rho) * sum(
        [pow(pars.Lambda * pars.rho, k) for k in range(int(n - 1))]
    )


def L2delt(pars: Pars, L: float) -> float:
    """Get delta-t from burst length L."""
    if L < 0:
        return L
    else:
        return L % pars.period


def L2n(pars: Pars, L: float) -> int:
    """Get number of spikes in burst of length L."""
    if L < 0:
        return 0
    else:
        return int(L // pars.period + 1)


def d2n(pars: Pars, d: float) -> int:
    """Just a convenience function to find n from d."""
    L = F_map(pars, d)
    return L2n(pars, L)


def d_n(pars: Pars, delt: float) -> float:
    """Value of d at last (nth) spike time."""
    return pars.gstar / pars.g * np.exp(delt / pars.taus)


def d_a(pars: Pars, n: int) -> float:
    """Leftmost bound of Pn-map domain."""
    return -beta(pars, n) / alpha(pars, n)


# def root_info(pars: Pars, root: float) -> dict:
#     """Try to extract useful information from root."""
#     if root is not None:
#         period =

# def calc_period(pars: Pars, dstar: float, n=None):
#     """Compute solution period associated with fixed point of map."""
#     if n is not None:       # explicit map
#         delt = self.F_map(dstar, n)
#     else:                   # recursive map
#         L = self.F_map_rec(dstar)
#         n = self.L2n(L)
#         delt = self.L2delt(L)

#     return 2*((n-1)*self.T+delt)


## Explicit maps and functions


def s_map(pars: Pars, d: float) -> float:
    """Solution to ds/dt."""
    return pars.g * d * np.exp(-pars.period / pars.taus)


def d_map(pars: Pars, d: float) -> float:
    """Solution to dd/dt."""
    return 1 - (1 - pars.Lambda * d) * pars.rho


# TODO: Rename to delta
def d_map_sol(pars: Pars, d: float, n: int) -> float:
    """Solution to d_map after n spikes."""
    return alpha(pars, n) * d + beta(pars, n)


def Fn_map(pars: Pars, d: float, n: int) -> float:
    """Free phase map Fn."""
    return pars.taus * np.log((pars.g / pars.gstar) * d_map_sol(pars, d, n))


def Qn_map(pars: Pars, delt: float, n: int) -> float:
    """Quiet phase map Qn."""
    exponent = -((n - 1) * pars.period + 2 * delt) / pars.taud
    return 1 - (1 - pars.Lambda * d_n(pars, delt)) * np.exp(exponent)


def Pn_map(pars: Pars, d: float, n: int) -> float:
    """Explicit burstmap."""
    return Qn_map(pars, Fn_map(pars, d, n), n)


def Pn_map_diff(pars: Pars, d: float, n: int) -> float:
    """Derivative of Pn for root finder."""

    delta = d_map_sol(pars, d, n)
    gr = pars.g / pars.gstar
    tau = 2 * pars.taus / pars.taud
    b = (pars.taud - 2 * pars.taus) / (pars.taud * pars.taus)
    p1 = pars.taus * alpha(pars, n) * pow(pars.rho, n - 1) * pow(gr, -tau)
    p2 = 2 / pars.taud * pow(delta, -(tau + 1)) + b * pars.Lambda * pow(
        delta, -tau
    )
    return p1 * p2


def Pn_map_fps(pars: Pars, n: int) -> Tuple[Optional[float], Optional[float]]:
    """Compute fixed points of Pn-map using Newton."""

    P = lambda d: Pn_map(pars, d, n) - d
    DP = lambda d: Pn_map_diff(pars, d, n) - 1

    # Unstable fixed point
    EPSILON = 0.0001
    x0 = d_a(pars, n) + EPSILON
    try:
        fp0 = optimize.newton(P, x0, fprime=DP)
        # fp0 = sp.optimize.brentq(P, x0, 1)
    except:
        fp0 = None

    # Stable fixed point
    x1 = 1
    try:
        fp1 = optimize.newton(P, x1, fprime=DP)
    except:
        fp1 = None

    # There are either two or no fixed points.
    if fp0 is None and fp1 is None:
        return (None, None)
    else:
        return (fp0, fp1)


def Phi(pars, d, n):
    return Pn_map(pars, d, n) - d


def phi(pars, n, x0=1.0):
    """Stable fixed point."""
    P = lambda d: Pn_map(pars, d, n) - d

    try:
        fp = sp.optimize.newton(P, x0)
        if np.isreal(fp):
            return np.real(fp)
        else:
            return np.nan
    except:
        return np.nan


def fixed_points(pars, n, eps=0.000000001):
    """Stable fixed point."""
    try:
        # First find stable fixed point using Newton
        x1 = sp.optimize.newton(lambda d: Phi(pars, d, n), x0=1)
        # Then find unstable fixed point using bisection
        x0 = sp.optimize.bisect(
            lambda d: Phi(pars, d, n), d_a(pars, n) + eps, x1 - eps
        )
        return (x0, x1)
    except:
        return (np.nan, np.nan)


def Pn_map_period(pars: Pars, dstar: float, n: int) -> float:
    """Estimate burst period from a fixed point of Pn-map."""
    delt = Fn_map(pars, dstar, n)
    return 2 * ((n - 1) * pars.period + delt)


def Pn_map_bif(pars: Pars, n: int, gs: Sequence) -> np.ndarray:
    """Compute bifurcation diagram of Pn-map along g."""
    # Table rows storing information.
    rows = []
    for g in gs:
        # print('g=%s' % g)
        _pars = pars._replace(g=g)
        fps = Pn_map_fps(_pars, n)

        # Extract information from fixed points.
        for idx, fp in enumerate(fps):
            if fp is not None:
                period = Pn_map_period(_pars, fp, n)
                row = {
                    "g": g,
                    "dstar": fp,
                    "n": n,
                    "period": period,
                    "stable": False if idx == 0 else True,
                }
                rows.append(row)

    return DataFrame(rows)


def Pn_map_bif_T(pars, n, Ts):
    """Compute bifurcation diagram of Pn-map along T."""
    # Table rows storing information.
    rows = []
    for T in Ts:
        # print('T=%s' % T)
        _pars = pars._replace(period=T)
        da = d_a(_pars, n)
        fp_unstable, fp_stable = fixed_points(_pars, n)
        # fp_stable = phi(_pars, n, x0=1)
        # fp_unstable = phi(_pars, n, x0=da+0.01)

        period_stable = Pn_map_period(_pars, fp_stable, n)
        period_unstable = Pn_map_period(_pars, fp_unstable, n)
        row = {
            "T": T,
            "dstar": fp_stable,
            "n": n,
            "period": period_stable,
            "stable": True,
        }
        rows.append(row)
        row = {
            "T": T,
            "dstar": fp_unstable,
            "n": n,
            "period": period_unstable,
            "stable": False,
        }
        rows.append(row)

    return DataFrame(rows)


# R and L are the bif border functions
def R(pars, n):
    fp = phi(pars, n, x0=0.99)
    if fp:
        return Fn_map(pars, fp, n) - pars.period
    else:
        return None


def L(pars, n):
    fp = phi(pars, n, x0=0.99)
    if fp:
        gtot = (
            pars.g
            * d_map_sol(pars, fp, n - 1)
            * np.exp(-pars.period / pars.taus)
        )
        return gtot - pars.gstar
    else:
        return None


def rborder(pars, n, x0):
    f = lambda T: R(pars._replace(period=T), n)
    fp0 = sp.optimize.newton(
        f,
        x0,
    )
    return fp0


def lborder(pars, n, x0):
    f = lambda T: L(pars._replace(period=T), n)
    fp0 = np.optimize.newton(f, x0)
    return fp0


# TODO:
# Remove function later
def fp2g(pars: Pars, dstar: float, n: int) -> float:
    """Compute synaptic strength as function of fixed point."""
    # That's G_n in the equations.
    delta = d_map_sol(pars, dstar, n)
    tau = 2 * pars.taus / pars.taud
    prod = delta ** (-tau) * pars.rho ** (n - 1) * (1 - pars.Lambda * delta)
    return pars.gstar * (prod / (1 - dstar)) ** (1 / tau)


# NOTE: Gn is the same as fp2g, I'm just refactoring and want to keep the
# naming closer to the paper.
def Gn(pars, dstar, n):
    delta = d_map_sol(pars, dstar, n)
    tau = pars.tau
    prod = delta ** (-tau) * pars.rho ** (n - 1) * (1 - pars.Lambda * delta)
    return pars.gstar * (prod / (1 - dstar)) ** (1 / tau)


def critical_fp(pars, n):
    """Compute critical fixed point (db, gb) at fold bifurcation."""
    fun = lambda d: Gn(pars, d, n)
    bounds = [d_a(pars, n), 1]
    res = scipy.optimize.minimize(fun, x0=0, method="Nelder-Mead")
    d_min = res.x[0]
    g_min = res.fun
    return d_min, g_min


def phi_unstable(pars, g, n):
    """Find unstable fixed point"""
    f = lambda d: Gn(pars._replace(g=g), d, n)
    dmin, gmin = critical_fp(pars, n)
    return inversefunc(f, domain=[d_a(pars, n), dmin])(g)


def phi_stable(pars, g, n):
    """Find unstable fixed point"""
    f = lambda d: Gn(pars._replace(g=g), d, n)
    dmin, gmin = critical_fp(pars, n)
    return inversefunc(f, domain=[dmin, 1])(g)


def left_fun(pars, g, n):
    df = phi_stable(pars, g, n)
    gtot = g * d_map_sol(pars, df, n - 1) * np.exp(-pars.period / pars.taus)
    return gtot - pars.gstar


def right_fun(pars, g, n):
    df = phi_stable(pars, g, n)
    return Fn_map(pars._replace(g=g), df, n) - pars.period


def left_border(pars, n, x0=1.0):
    f = lambda g: left_fun(pars, g, n)
    fp0 = optimize.newton(f, x0)
    return fp0


def right_border(pars, n, x0=1.0):
    f = lambda g: right_fun(pars, g, n)
    fp0 = optimize.newton(f, x0)
    return fp0


## Recursive maps and functions


def F_map(pars: Pars, d: float) -> float:
    """Recursive free phase map F."""
    if s_map(pars, d) <= pars.gstar:
        return pars.taus * np.log(pars.g * d / pars.gstar)
    else:
        dnew = d_map(pars, d)
        return pars.period + F_map(pars, dnew)


def Q_map(pars: Pars, L: float) -> float:
    """Recursive quiet phase map Q."""
    delt = L2delt(pars, L)
    n = L2n(pars, L)
    return Qn_map(pars, delt, n)


def P_map(pars: Pars, d: float) -> float:
    """Recursive burst map."""
    return Q_map(pars, F_map(pars, d))


def P_map_fps(
    pars: Pars, nsteps: int = 10000, con_thresh: float = 0.005
) -> [float, float]:
    """
    Compute fixed points of recursive P-map using brute force.

    Since the input is 1D and the function only partially
    continuous brute force is appropriate.

    """

    # Define grid.
    xs = np.linspace(0, 1, nsteps)
    ys = np.array([P_map(pars, x) - x for x in xs])

    # Find continuous intervals.
    contints = np.hstack(([0], abs(np.diff(ys)) < con_thresh))
    # Find zero crossing by sign change.
    zerocros = np.hstack(([0], np.diff((ys > 0) * 1)))

    # Roots
    y0idxs = np.where(np.multiply(contints, zerocros))
    roots = xs[y0idxs]

    return roots


def P_map_period(pars: Pars, dstar: float) -> float:
    """Estimate burst period from a fixed point of recursive P-map."""
    L = F_map(pars, dstar)
    n = L2n(pars, L)
    delt = L2delt(pars, L)
    return 2 * ((n - 1) * pars.period + delt)


def P_map_bif(
    pars: Pars, gs: Sequence, nsteps: int = 10000, con_thresh: float = 0.005
) -> np.ndarray:
    """Compute bifurcation diagram of recursive P-map along g."""
    # Table rows storing information.
    rows = []
    for g in gs:
        print("g=%s" % g)
        _pars = pars._replace(g=g)
        fps = P_map_fps(_pars, nsteps, con_thresh)

        # Extract information from fixed points.
        for idx, fp in enumerate(fps):
            if fp is not None:
                period = P_map_period(_pars, fp)
                row = {
                    "g": g,
                    "dstar": fp,
                    "n": L2n(_pars, period / 2),
                    "period": period,
                    "stable": False if idx == 0 else True,
                }
            else:
                row = {
                    "g": g,
                    "dstar": None,
                    "n": None,
                    "period": None,
                    "stable": None,
                }
            biftab = rows.append(row)

    return DataFrame(rows)
