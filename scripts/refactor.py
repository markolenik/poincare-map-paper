## Copy of recursive maps and functions (in case I need that)


def L2delt(self, L: float) -> float:
    """Get delta-t from burst length L."""
    if L < 0:
        return L
    else:
        return L % self.T



def L2n(self, L: float) -> int:
    """Get number of spikes in burst of length L."""
    if L < 0:
        return 0
    else:
        return int(L // self.T + 1)


def d2n(self, d: float) -> int:
    """Just a convenience function to find n from d."""
    L = self.F_map(d)
    return self.L2n(L)

def F_map(self: self, d: float) -> float:
    """Recursive free phase map F."""
    if s_sol(self, d) <= self.gstar:
        return self.taus * np.log(self.g * d / self.gstar)
    else:
        dnew = d_sol(self, d)
        return self.T + F_map(self, dnew)


def Q_map(self: self, L: float) -> float:
    """Recursive quiet phase map Q."""
    delt = L2delt(self, L)
    n = L2n(self, L)
    return Qn_map(self, delt, n)


def P_map(self: self, d: float) -> float:
    """Recursive burst map."""
    return Q_map(self, F_map(self, d))


def P_map_fps(
    self: self, nsteps: int = 10000, con_thresh: float = 0.005
) -> [float, float]:
    """
    Compute fixed points of recursive P-map using brute force.

    Since the input is 1D and the function only partially
    continuous brute force is appropriate.

    """

    # Define grid.
    xs = np.linspace(0, 1, nsteps)
    ys = np.array([P_map(self, x) - x for x in xs])

    # Find continuous intervals.
    contints = np.hstack(([0], abs(np.diff(ys)) < con_thresh))
    # Find zero crossing by sign change.
    zerocros = np.hstack(([0], np.diff((ys > 0) * 1)))

    # Roots
    y0idxs = np.where(np.multiply(contints, zerocros))
    roots = xs[y0idxs]

    return roots


def P_map_period(self: self, dstar: float) -> float:
    """Estimate burst period from a fixed point of recursive P-map."""
    L = F_map(self, dstar)
    n = L2n(self, L)
    delt = L2delt(self, L)
    return 2 * ((n - 1) * self.T + delt)


def P_map_bif(
    self: self, gs: Sequence, nsteps: int = 10000, con_thresh: float = 0.005
) -> np.ndarray:
    """Compute bifurcation diagram of recursive P-map along g."""
    # Table rows storing information.
    rows = []
    for g in gs:
        print("g=%s" % g)
        _self = self._replace(g=g)
        fps = P_map_fps(_self, nsteps, con_thresh)

        # Extract information from fixed points.
        for idx, fp in enumerate(fps):
            if fp is not None:
                period = P_map_period(_self, fp)
                row = {
                    "g": g,
                    "dstar": fp,
                    "n": L2n(_self, period / 2),
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
