import numpy as np
from typing import List, Optional, Tuple
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


def roots(x: np.ndarray) -> np.ndarray:
    """Find roots using sign changes."""
    signs = np.sign(x)
    roots = np.where(signs + np.roll(signs, -1) == 0)[0]
    return roots


def spikes(x: np.ndarray) -> np.ndarray:
    """Find spike indices using roots with positive slope."""

    id1 = roots(x)
    if len(id1) > 0:
        grad1 = np.gradient(x)
        return id1[grad1[id1]>0]
    else:
        return np.array([])


def events(x: np.ndarray, down: bool = True) -> np.ndarray:
    """Find indices of events where the slope of array changes its sign."""
    grad = np.gradient(x)
    signs = np.sign(grad)
    direction = signs > 0 if down else signs < 0
    sign_changes = np.where((signs + np.roll(signs, -1) == 0) & direction)[0]
    return sign_changes


def find_lc(x: np.ndarray, thresh: float = 1e-3) -> Optional[Tuple[int, int]]:
    """Return start and end index of limit cycle solution."""
    idxs = events(x)

    def first_return(idxs, thresh=thresh):
        """Compute indices of limit cycle via first periodic return."""

        # Go backwards because of transients.
        back_cross = idxs[::-1]
        # Find crossing where norm function is small enough.
        for _, idx in enumerate(back_cross[1:]):
            err = abs(x[idx] - x[idxs[-1]])
            if err < thresh:
                return idx
        # No section return found.
        return None

    start = first_return(idxs, thresh=thresh)
    end = idxs[-1]
    if start is not None:
        return (start, end)
    else:
        return None


def shift_sol(sol, shift):
    """Shift solution by some time."""
    P = sol[-1, 0]
    dt = sol[1, 0] - sol[0, 0]
    roll_by = int(shift / float(dt))
    shifted_sol = np.roll(sol, roll_by, axis=0)
    # Recompute time.
    shifted_sol[:, 0] = np.linspace(0.0, P, len(shifted_sol))
    return shifted_sol


def cobwebplot(f, xn, ax=None, **plot_args):
    """Plot a sequence as cobweb."""
    if ax is None:
        ax = plt.gca()
    for x in xn:
        ax.plot([x, f(x)], [f(x), f(x)], **plot_args)
        ax.plot([f(x), f(x)], [f(x), f(f(x))], **plot_args)
    return ax


def iterate(f, x0, N):
    """Iterate function with given initial condition for some steps."""
    ns = range(N)
    xs = np.zeros(N)
    xs[0] = x0
    for idx, x in enumerate(xs[:-1]):
        xprev = xs[idx]
        xs[idx + 1] = f(xprev)
    return xs


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
