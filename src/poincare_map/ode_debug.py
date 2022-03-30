import numpy as np
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
import typing as tp

plt.ioff()

from poincare_map import ode
from poincare_map.poincare_map import PoincareMap
from poincare_map import helpers
from scipy import optimize

ml = ode.ML("odes/ml.ode")
mlml = ode.MLML("odes/mlml.ode")

#%% Find HB.

gstar = 0.0068
g_hb = ml.find_HB(g0=0.007, total=3000)

#%% Investigate how close cell trajectory is to fixed point trajectory.
vs = np.linspace(-60, 60, 500)

# Initialise system at fixed point.
# gtots = np.linspace(0.3, tp.cast(float, g_hb), 100)
gtot0 = 0.3
vfp0 = tp.cast(float, ml.fixed_point(gtot0))

# Integrate for single gtot for now
sol = ml.run(ics=[vfp0, ml.winf(vfp0), gtot0])
vfp = np.array([ml.fixed_point(gtot) for gtot in sol["gtot"]])
wfp = [ml.winf(v) for v in vfp]
unstable_mask = sol["gtot"]<g_hb
v_LK = [
    tp.cast(float, ml.find_left_knee(gtot)) for gtot in sol["gtot"]
]
w_LK = [ml.winf(v) for v in v_LK]


#%% Plot nullclines
fig, ax = plt.subplots()
# Plot nullclines without inhibition
ax.plot(vs, [ml.winf(v) for v in vs])
ax.plot(vs, [ml.vinf(v, 0) for v in vs])
ax.plot([vfp0], [ml.winf(vfp0)], "o", color="black")
ax.plot(sol["v"], sol["w"], color="black")
# ax.plot(vfp_trajectory, [ml.winf(v) for v in vfp_trajectory], color="red")
ax.plot(
    v_LK,
    [ml.winf(v) for v in v_LK],
    color="red",
    linestyle="dotted",
)
ax.set(xlabel="v", ylabel="w", ylim=(-0.05, 0.4), xlim=(-62, 10))
fig.savefig("tmp/LK/nullclines.pdf")

fig, ax = plt.subplots()
ax.plot(sol.index, sol["gtot"])
ax.set(xlabel="time", ylabel="gtot")
fig.savefig("tmp/LK/gtot.pdf")


#%% At what point does cell spike, where's LK?

fig, axs = plt.subplots(nrows=2, sharex=True, tight_layout=True, figsize=(5, 10))
axs[0].plot(sol.index, sol["v"], label=r"$v$")
# LK
axs[0].plot(
    sol.index,
    v_LK,
    label=r"$v_{LK}$",
    color="black",
    linestyle="dashed",
)
# fp unstable
axs[0].plot(
    sol.index,
    vfp,
    label=r"$v_{fp}$",
    color="C5",
    linestyle="dotted",
)
# fp stable
axs[0].plot(
    sol.index,
    vfp,
    label=r"$v_{fp}$",
    color="C5",
    linestyle="dotted",
)

axs[0].legend()
axs[0].set(ylabel=r"$v$")
axs[1].plot(sol.index, sol["w"], label=f"$w$")
# LK
axs[1].plot(
    sol.index,
    w_LK,
    label=r"$w_{LK}$",
    color="black",
    linestyle="dashed",
)
# fp
axs[1].plot(
    sol.index,
    wfp,
    label=r"$w_{fp}$",
    color="grey",
    linestyle="dotted",
)
axs[1].axvline(370, ymax=200, clip_on=False, linestyle='-.', color='C3')
axs[1].set(xlabel="time", ylabel="w")
axs[1].legend()
fig.savefig("tmp/LK/sol.pdf")

#%% plot delta
ds = np.linspace(0, 1, 100)
poincare = PoincareMap(4)
# delta_n = lambda n: poincare._replace(n=n).delta()

dmap = lambda d: poincare.Lambda*poincare.Rho*d + (1-poincare.Rho)

fig, ax = plt.subplots()
ax.plot(ds, ds, color='black', linestyle='dashed')
# ax.plot(ds, [PoincareMap(n).delta(d) for d in ds], label=f"n={n}")
ax.plot(ds, [dmap(d) for d in ds])
ax.axvline(poincare.dsup, linestyle='dashed', color='black')

ax.set(xlabel=r"$d_k$", ylabel=r"$d_{k+1}$")
fig.savefig("tmp/delta.pdf")
