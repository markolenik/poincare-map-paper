from __future__ import annotations
from logging import debug

import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from typing import Dict, List, Optional, Tuple
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


def roots(x: np.ndarray, up: int = 0) -> np.ndarray:
    """Find roots using sign changes."""
    signs = np.sign(x)
    # root_mask = (signs + np.roll(signs, -1)) == 0
    # NOTE: Careful here, not sure if this introduces others bugs...
    root_mask = np.append(np.diff(signs) != 0, False)
    # Consider direction of sign changes.
    if up == 1:  # Return positive change roots.
        return np.where(root_mask & (signs < 0))[0]
    elif up == -1:  # Return negative change roots.
        return np.where(root_mask & (signs > 0))[0]
    else:  # Return all roots.
        return np.where(root_mask)[0]


def spike_times(sol: pd.Series) -> pd.Index:
    """Find spike times of a voltage trace using upward zero crossings."""
    ids = roots(sol.to_numpy(), up=1)
    return sol.index[ids]


# def events(x: np.ndarray, down: bool = True) -> np.ndarray:
#     """Find indices of events where the slope of array changes its sign."""
#     grad = np.gradient(x)
#     signs = np.sign(grad)
#     direction = signs > 0 if down else signs < 0
#     sign_changes = np.where((signs + np.roll(signs, -1) == 0) & direction)[0]
#     return sign_changes


# def find_limit_cycle(
#     sol: pd.DataFrame,
#     thresh: float = 1e-3,
#     norm: bool = True,
# ) -> pd.DataFrame:
#     """Try to find limit cycle in solution using a voltage section."""
#     tks = spike_times(sol["v1"])
#     # Case 1: No spiking - return fixed point.
#     if tks.empty:
#         return sol.iloc[-1]
#     # Case 2: Spiking - return limit cycle.
#     else:
#         abs_errors = abs(sol.loc[tks]["v"] - sol.loc[tks[-1]]["v"])
#         return_mask = np.array(abs_errors < thresh)
#         # Do we have a return that meets the threshold criteria at all?
#         if np.any(return_mask[:-1]):
#             first_return_time = tks[abs_errors < thresh][-2]
#             lc = sol.loc[(first_return_time <= sol.index) & (sol.index < tks[-1])]
#             if norm:
#                 t0 = lc.index[lc["v"].argmax()]
#                 # Append remaining trajectory to end.
#                 lc_before = lc.loc[lc.index < t0]
#                 lc_after = lc.loc[lc.index >= t0]
#                 lc_normed = pd.concat([lc_after, lc_before])
#                 # Recompute index.
#                 period = lc.index[-1] - lc.index[0]
#                 time = np.linspace(0, period, len(lc))
#                 return lc_normed.set_index(pd.Index(time, name=lc.index.name))
#             else:
#                 return lc
#     # No limit cycle could be found, return empty dataframe.
#     else:
#         return pd.DataFrame(columns=sol.columns)
#     if spike_times.empty:
#         return sol.iloc[-1]


def limit_cycle_index(
    series: pd.Series, event_times: pd.Index, thresh: float = 1e-3
) -> pd.Index:
    """Return limit limit cycle indices in variable using events."""
    # Compute errors in series to last event.
    abs_errors = abs(series.loc[event_times] - series.loc[event_times[-1]])
    return_mask = np.array(abs_errors < thresh)
    # Do we have a return that meets the threshold criteria at all?
    if np.any(return_mask[:-1]):
        first_return_time = event_times[abs_errors < thresh][-2]
        return series.index[
            (first_return_time <= series.index)
            & (series.index < event_times[-1])
        ]
    # No limit cycle could be found, return empty series.
    else:
        return pd.Index([], name=series.index.name)


def find_depression_limit_cycle(
    sol: pd.DataFrame,
    thresh: float = 1e-3,
    norm: bool = True,
) -> pd.DataFrame:
    """Try to fine limit cycle in 2-cell solution. Return empty dataframe on fail."""

    tks = spike_times(sol["v1"])
    # Compute errors in d1 to last spike
    abs_errors = abs(sol.loc[tks]["d1"] - sol.loc[tks[-1]]["d1"])
    return_mask = np.array(abs_errors < thresh)
    # Do we have a return that meets the threshold criteria at all?
    if np.any(return_mask[:-1]):
        first_return_time = tks[abs_errors < thresh][-2]
        lc = sol.loc[(first_return_time <= sol.index) & (sol.index < tks[-1])]
        # Normalise limit cycle s.t. d1 is maximum at t=0.
        if norm:
            t0 = lc.index[lc["d1"].argmax()]
            # Append remaining trajectory to end.
            lc_before = lc.loc[lc.index < t0]
            lc_after = lc.loc[lc.index >= t0]
            lc_normed = pd.concat([lc_after, lc_before])
            # Recompute index.
            period = lc.index[-1] - lc.index[0]
            time = np.linspace(0, period, len(lc))
            return lc_normed.set_index(pd.Index(time, name=lc.index.name))
        else:
            return lc
    # No limit cycle could be found, return empty dataframe.
    else:
        return pd.DataFrame(columns=sol.columns)


def limit_cycle_info(lc: pd.DataFrame) -> Dict:
    """Extract info from LC."""
    period = lc.index[-1] - lc.index[0]
    tks = spike_times(lc["v1"])
    n = len(tks)
    dmax = lc["d1"].max()
    dmin = lc["d1"].min()
    info = {"n": n, "period": period, "dmax": dmax, "dmin": dmin}
    return info


def sigmoid(x: float, A, B) -> float:
    """Sigmoid function."""
    return 0.5 * (1 + np.tanh((x - A) / B))


# https://stackoverflow.com/a/27637925/9904918
def add_arrow_to_line2D(
    axes,
    line,
    arrow_locs=[0.2, 0.4, 0.6, 0.8],
    arrowstyle="-|>",
    arrowsize=1,
    transform=None,
):
    """Add arrows to a matplotlib.lines.Line2D at selected locations.

    Parameters:
    -----------
    axes:
    line: Line2D object as returned by plot command
    arrow_locs: list of locations where to insert arrows, % of total length
    arrowstyle: style of the arrow
    arrowsize: size of the arrow
    transform: a matplotlib transform instance, default to data coordinates

    Returns:
    --------
    arrows: list of arrows
    """
    if not isinstance(line, mlines.Line2D):
        raise ValueError("expected a matplotlib.lines.Line2D object")
    x, y = line.get_xdata(), line.get_ydata()

    arrow_kw = {
        "arrowstyle": arrowstyle,
        "mutation_scale": 10 * arrowsize,
    }

    color = line.get_color()
    use_multicolor_lines = isinstance(color, np.ndarray)
    if use_multicolor_lines:
        raise NotImplementedError("multicolor lines not supported")
    else:
        arrow_kw["color"] = color

    linewidth = line.get_linewidth()
    if isinstance(linewidth, np.ndarray):
        raise NotImplementedError("multiwidth lines not supported")
    else:
        arrow_kw["linewidth"] = linewidth

    if transform is None:
        transform = axes.transData

    arrows = []
    for loc in arrow_locs:
        s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        n = np.searchsorted(s, s[-1] * loc)
        arrow_tail = (x[n], y[n])
        arrow_head = (np.mean(x[n : n + 2]), np.mean(y[n : n + 2]))
        p = mpatches.FancyArrowPatch(
            arrow_tail, arrow_head, transform=transform, **arrow_kw
        )
        axes.add_patch(p)
        arrows.append(p)
    return arrows
