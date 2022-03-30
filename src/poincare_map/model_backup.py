
    # ----------------------------------------------------------
    # TODO: Sort this out, what's still needed and what can be discarded

    # def Pn_map_diff(self, d: float, n: int) -> float:
    #     """Derivative of Pn for root finder."""

    #     _delta = self.delta(self, d, n)
    #     gr = self.g / self.gstar
    #     tau = 2 * self.taus / self.taud
    #     b = (self.taud - 2 * self.taus) / (self.taud * self.taus)
    #     p1 = (
    #         self.taus
    #         * self.alpha(self, n)
    #         * pow(self.Rho, n - 1)
    #         * pow(gr, -tau)
    #     )
    #     p2 = 2 / self.taud * pow(_delta, -(tau + 1)) + b * self.Lambda * pow(
    #         _delta, -tau
    #     )
    #     return p1 * p2

    # def Pn_map_fps(self, n: int) -> Tuple[Optional[float], Optional[float]]:
    #     """Compute fixed points of Pn-map using Newton."""

    #     P = lambda d: self.P_map(d, n) - d
    #     DP = lambda d: self.Pn_map_diff(d, n) - 1

    #     # Unstable fixed point
    #     EPSILON = 0.0001
    #     x0 = self.d_a(n) + EPSILON
    #     try:
    #         fp0 = optimize.newton(P, x0, fprime=DP)
    #         # fp0 = sp.optimize.brentq(P, x0, 1)
    #     except:
    #         fp0 = None

    #     # Stable fixed point
    #     x1 = 1
    #     try:
    #         fp1 = optimize.newton(P, x1, fprime=DP)
    #     except:
    #         fp1 = None

    #     # There are either two or no fixed points.
    #     if fp0 is None and fp1 is None:
    #         return (None, None)
    #     else:
    #         return (fp0, fp1)

    # def Phi(self, d, n):
    #     """Fixed point equation."""
    #     return self.P_map(d, n) - d

    # def phi(self, n, x0=1.0):
    #     """Numerically find stable fixed point."""
    #     try:
    #         fp = optimize.newton(partial(self.Phi, n=n), x0)
    #         if np.isreal(fp):
    #             return np.real(fp)
    #         else:
    #             return np.nan
    #     except:
    #         return np.nan

    # def fixed_points(self, n, eps=0.000000001):
    #     """Numerically find unstable fixed point."""
    #     try:
    #         # First find stable fixed point using Newton
    #         x1 = optimize.newton(lambda d: self.Phi(d, n), x0=1)
    #         # Then find unstable fixed point using bisection
    #         x0 = optimize.bisect(
    #             lambda d: self.Phi(d, n), self.d_a(n) + eps, x1 - eps
    #         )
    #         return (x0, x1)
    #     except:
    #         return (np.nan, np.nan)

    # def Pn_map_period(self, dstar: float, n: int) -> float:
    #     """Estimate burst period from a fixed point of Pn-map."""
    #     delt = self.F_map(dstar, n)
    #     return 2 * ((n - 1) * self.T + delt)

    # def Pn_map_bif(self, n: int, gs: Sequence) -> np.ndarray:
    #     """Compute bifurcation diagram of Pn-map along g."""
    #     # Table rows storing information.
    #     rows = []
    #     for g in gs:
    #         # print('g=%s' % g)
    #         _self = self._replace(g=g)
    #         fps = self.Pn_map_fps(_self, n)

    #         # Extract information from fixed points.
    #         for idx, fp in enumerate(fps):
    #             if fp is not None:
    #                 period = Pn_map_period(_self, fp, n)
    #                 row = {
    #                     "g": g,
    #                     "dstar": fp,
    #                     "n": n,
    #                     "period": period,
    #                     "stable": False if idx == 0 else True,
    #                 }
    #                 rows.append(row)

    #     return DataFrame(rows)

    # def Pn_map_bif_T(self, n, Ts):
    #     """Compute bifurcation diagram of Pn-map along T."""
    #     # Table rows storing information.
    #     rows = []
    #     for T in Ts:
    #         # print('T=%s' % T)
    #         _self = self._replace(period=T)
    #         da = self.d_a(_self, n)
    #         fp_unstable, fp_stable = self.fixed_points(n)
    #         # fp_stable = phi(_self, n, x0=1)
    #         # fp_unstable = phi(_self, n, x0=da+0.01)

    #         period_stable = Pn_map_period(_self, fp_stable, n)
    #         period_unstable = Pn_map_period(_self, fp_unstable, n)
    #         row = {
    #             "T": T,
    #             "dstar": fp_stable,
    #             "n": n,
    #             "period": period_stable,
    #             "stable": True,
    #         }
    #         rows.append(row)
    #         row = {
    #             "T": T,
    #             "dstar": fp_unstable,
    #             "n": n,
    #             "period": period_unstable,
    #             "stable": False,
    #         }
    #         rows.append(row)

    #     return DataFrame(rows)

    # # R and L are the bif border functions
    # def R(self, n):
    #     fp = self.phi(n, x0=0.99)
    #     if fp:
    #         return self.F_map(fp, n) - self.period
    #     else:
    #         return None

    # def L(self, n):
    #     fp = self.phi(n, x0=0.99)
    #     if fp:
    #         gtot = (
    #             self.g
    #             * self.delta(fp, n - 1)
    #             * np.exp(-self.period / self.taus)
    #         )
    #         return gtot - self.gstar
    #     else:
    #         return None

    # def rborder(self, n, x0):
    #     f = lambda T: R(self._replace(period=T), n)
    #     fp0 = optimize.newton(f, x0,)
    #     return fp0

    # def lborder(self, n, x0):
    #     f = lambda T: L(self._replace(period=T), n)
    #     fp0 = np.optimize.newton(f, x0)
    #     return fp0
