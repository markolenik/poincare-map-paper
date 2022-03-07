import os
from functools import partial
import pickle
import pandas as pd

import matplotlib

# NOTE: Maybe plot directly as pgf and import?
matplotlib.use("pgf")
import matplotlib
from matplotlib import patches
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import string
import numpy as np

from poincare_map import ode
from poincare_map import paths

from poincare_map.poincare_map import PoincareMap
from poincare_map import ode, paths, helpers


plt.ioff()
rcp = matplotlib.rcParamsDefault
rcp["figure.max_open_warning"] = 0
rcp["axes.spines.top"] = False
rcp["axes.spines.right"] = False
rcp["axes.unicode_minus"] = False
rcp["text.usetex"] = True
rcp["pgf.rcfonts"] = False
rcp["pgf.texsystem"] = "pdflatex"
rcp["xtick.labelsize"] = "small"
rcp["ytick.labelsize"] = "small"
rcp["axes.labelsize"] = "x-large"
rcp["axes.titlesize"] = "xx-large"
rcp["figure.titlesize"] = "large"
rcp["font.size"] = 9
# Use latex preamble.
rcp["pgf.preamble"] = r"\usepackage{amsmath} \usepackage{siunitx}"

matplotlib.rcParams.update(rcp)

np.seterr(divide='ignore')
np.seterr(all='ignore')

FIGWIDTH = 3.85
FIGSIZE = (FIGWIDTH, FIGWIDTH)
FIG2by1 = (FIGWIDTH * 1.4, (FIGWIDTH * 1.4) / 2)

ml = ode.ML(paths.ml_file)
mlml = ode.ML(paths.mlml_file)
pmap = PoincareMap()

df = pd.read_pickle(paths.numeric_bif_diagram)

#%% Nullclines uncoupled
vs = np.linspace(-65, 60, 100)
vnull = [ml.vinf(v, gtot=0) for v in vs]
wnull = [ml.winf(v) for v in vs]
lc = ml.find_equilibrium(total=3000)

# loop = lc.append(lc.iloc[0])    # Close the loop.
loop = lc

#%%
# Plot.
fig, axs = plt.subplots(ncols=2, tight_layout=True, figsize=FIG2by1)

# Nullclines.
ax = axs[0]
ax.plot(vs, vnull, color="C0")
ax.plot(vs, wnull, color="C1")
(line,) = ax.plot(loop["v"], loop["w"], color="black")
fp_stable = [-5.53, 0.219]
ax.plot(fp_stable[0], fp_stable[1], c="C1", marker="o", markeredgecolor="k")
ax.text(fp_stable[0] + 4, fp_stable[1] - 0.01, r"$p_f$")

helpers.add_arrow_to_line2D(
    ax, line, arrow_locs=[0.05, 0.35, 0.6, 0.8], arrowstyle="->"
)

ax.text(-45, 0.442, "Jump Down")
ax.text(-4.5, 0.02, "Jump Up")
ax.text(-60, 0.08, "Silent\nState")
ax.text(30, 0.3, "Active\nState")

# ax.text(-50, 0.6, r'$v_\infty$')
# ax.text(40, 0.9, r'$w_\infty$')

ax.set(
    ylim=(0, 0.65),
    xlabel=r"$v\; (\si{mV})$",
    ylabel=r"$w$",
    xticks=np.arange(-60, 80, 40),
    yticks=np.arange(0, 0.6, 0.2),
)

# ax.autoscale(tight=True, axis="both")

# # Trace.
# ax = axs[1]
# ax.plot(lc.t, lc.v1, color='C0')

# ax.text(4, 5, 'Jump\nUp')
# ax.text(60, -20, 'Jump\nDown')
# ax.text(155, -47, 'Silent\nState')
# ax.text(37, 25, 'Active\nState')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.set_xlabel('time ' r'$(\si{ms})$')
# ax.set_ylabel(r'$v\;(\si{mV})$')
# ax.set_xticks(sp.arange(0, 220, 50))
# ax.set_ylim(top=70)
# ax.autoscale(tight=True, axis='x')

# # Add Tact and Tinact
# ax.axhline(0, linestyle='--', color='k', linewidth=0.5)
# ax.axvline(50, linestyle='--', color='k', linewidth=0.5)
# ax.axvline(217, linestyle='--', color='k', linewidth=0.5)
# ax.annotate("", xy=(0, 55), xytext=(50, 55), arrowprops=dict(arrowstyle="<->"),
#             xycoords='data')
# ax.annotate("", xy=(50, 55), xytext=(217, 55), arrowprops=dict(arrowstyle="<->"),
#             xycoords='data')
# ax.text(15, 60, r'$T_{act}$')
# ax.text(110, 60, r'$T_{inact}$')

# for idx, ax in enumerate(axs):
#     ax.set_title(string.ascii_uppercase[idx], loc='left')
fig.savefig(paths.figures / "nullclines-uncoupled.pdf")


#%% Free-quiet
# TODO: Replace gbar*s with ggot?

n = 3
mlml = ode.MLML(paths.mlml_file)
poincare = PoincareMap(n=n)
# Compute 3-3 solution.
g = 0.5
lc = mlml.find_limit_cycle(n=n, total=10000)
period = lc.index[-1]
# Spike times of cell 1.
spike_times = helpers.spike_times(lc["v1"])

fig, axs = plt.subplots(
    nrows=3, tight_layout=True, sharex=True, figsize=(FIGWIDTH, FIGWIDTH * 1.7)
)

# A: Plot voltage trace.
ax = axs[0]
ax.plot(lc.index, lc["v1"])
ax.plot(lc.index, lc["v2"])
ax.set(ylim=(-80, 140), ylabel=r"$v_{1,2}$")

# Add delta-t patches.
delt = period / 2 - poincare.T * (n - 1) - poincare.Tact
# Patch start time
delt_t0s = [spike_times[-1] + poincare.Tact, period - delt]
min_v = min(lc["v1"])
max_v = max(lc["v1"])
height = max_v - min_v
for delt_t0 in delt_t0s:
    rect = patches.Rectangle(
        (delt_t0, min_v),
        delt,
        2 * height,
        alpha=0.2,
        color="grey",
        transform=ax.transData,
    )
    ax.add_patch(rect)
    ax.text(
        delt_t0 + delt / 2,
        1.1 * height,
        r"$\Delta t$",
        horizontalalignment="center",
        transform=ax.transData,
    )

# Add arrows between spikes.
OFFSET = 16
v0 = max(lc["v1"])
arrow_startpoints = spike_times.to_list()
arrow_endpoints = arrow_startpoints[1:] + [period]
for arrow_startpoint, arrow_endpoint in zip(arrow_startpoints, arrow_endpoints):
    midpoint = arrow_startpoint + (arrow_endpoint - arrow_startpoint) / 2.0
    p = patches.FancyArrowPatch(
        (arrow_startpoint, v0),
        (arrow_endpoint, v0),
        connectionstyle="arc3, rad=-0.3",
        arrowstyle="->",
        mutation_scale=8,
    )
    ax.add_patch(p)

    # Put text on top of patch.
    if arrow_endpoint == period:
        text_y = v0 + 2.5 * OFFSET
        text = r"$(n-1)T + 2(\Delta t + T_{act})$"
    else:
        text_y = v0 + OFFSET
        text = r"$T$"

    ax.text(midpoint, text_y, text, horizontalalignment="center")


# B: Plot synaptic variables.
ax = axs[1]
ax.plot(lc.index, lc["gtot1"], c="C0")
ax.plot(lc.index, poincare.gstar * np.ones(len(lc)), c="k", linestyle="--")
ax.set(ylabel=r"$\bar g s_1$", xticks=[0, 2 * poincare.T, period])
ax.autoscale(axis="x", tight=True)
ax.text(1500, 0.03, r"$g^{\star}$")

# Add first delta-t patch here as well.
rect = patches.Rectangle(
    (2 * poincare.T + poincare.Tact, 0),
    delt,
    1.0,
    alpha=0.2,
    color="grey",
    transform=ax.transData,
)

ax.add_patch(rect)
ax.text(
    (spike_times[-1] + delt / 2) / period,
    0.9,
    r"$\Delta t$",
    horizontalalignment="center",
    transform=ax.transAxes,
)


# C: Plot d.
ax = axs[2]
ax.plot(lc.index, lc["d1"], c="C0", lw=2)
ax.set(
    xlabel="time",
    ylabel=r"$d_1$",
    xticks=[0, 2 * poincare.T + poincare.Tact, period],
    xticklabels=[
        r"$0$",
        r"$(n-1)T + T_{act}$",
        r"$P_n=2((n-1)T + \Delta t + T_{act})$",
    ],
)
ax.autoscale(axis="x", tight=True)

# Free-quiet patches
rect_free = patches.Rectangle(
    (0, 0),
    2 * poincare.T + poincare.Tact,
    1.0,
    alpha=0.1,
    color="C0",
    transform=ax.transData,
)
rect_quiet = patches.Rectangle(
    (2 * poincare.T + poincare.Tact, 0),
    period - 2 * poincare.T,
    1.0,
    alpha=0.1,
    color="C1",
    transform=ax.transData,
)
ax.add_patch(rect_free)
ax.add_patch(rect_quiet)
text_y = 0.85
ax.text(
    0.2,
    text_y,
    r"free: $\bar gs>g^{\star}$",
    horizontalalignment="center",
    transform=ax.transAxes,
)
ax.text(
    0.7, text_y, "quiet", horizontalalignment="center", transform=ax.transAxes
)

# Add subplot titles.
for idx, ax in enumerate(axs):
    # ax.set(title=string.ascii_uppercase[idx], loc="left")
    ax.set_title(string.ascii_uppercase[idx], loc="left")
    ax.set_yticks([])

fig.savefig("figures/free-quiet.pdf")


#%% Release delay
# NOTE: We don't need this one anymore

## Compute T diagram.
pmap = PoincareMap()


def get_delta_epsilon(sol, offset=10):
    """Compute the release time, that is when gs<g*."""
    spike_times2 = helpers.spike_times(sol["v2"])
    # Need offset since first couple of ticks RC might be satisfied.
    _sol = sol.loc[sol.index > offset]
    release_condition_satisfied = _sol.loc[_sol["gtot1"] < pmap.gstar].index[0]
    return spike_times2[0] - release_condition_satisfied


delta_epsilons = []
for row in df.itertuples():
    sol = getattr(row, "lc")
    delta_epsilons.append(get_delta_epsilon(sol))
df["delta_epsilon"] = delta_epsilons


fig, ax = plt.subplots(tight_layout=True)
ax.plot(df["g"].iloc[1:], df["delta_epsilon"].iloc[1:], ".")
ax.set_xlim(left=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlabel(r"$\bar g$ " + r"$(\si{mS/cm^{2}})$")
ax.set_ylabel("Release delay (ms)")


fig.savefig("figures/release-delay.pdf")

#%% Delta t

fig, ax = plt.subplots(figsize=FIGSIZE, tight_layout=True)

# Plot delta-t
for n in df["n"].unique():
    dfn = df.loc[df["n"] == n]
    delta_ts = dfn["lc"].apply(partial(helpers.calc_delta_t, Tact=pmap.Tact))
    ax.plot(dfn["g"], delta_ts, color="C0")

# Place Tinact on top horizontal line
ax.text(
    0.49,
    pmap.Tinact + 0.5,
    r"$T_{inact}$",
    horizontalalignment="center",
    verticalalignment="bottom",
)

# Place ns
ax.text(0.3, 293, r"$1$")
ax.text(0.37, 305, r"$2$")
ax.text(0.45, 315, r"$3$")
ax.text(0.5, 320, r"$4$")
ax.text(0.55, 323, r"$5$")

# Add vertical line indicating T.
ax.axhline(y=pmap.Tinact, c="grey", linestyle="--")

ax.set(
    # yticks=np.arange(80, 230, 20),
    xlim=(0.28, 0.6),
    ylim=(290, 330),
    xlabel=r"$\bar g$ " + r"$(\si{mS/cm^{2}})$",
    ylabel=r"$\Delta t$ " + r"$(\si{ms})$",
)

fig.savefig(paths.figures / "delta-t.pdf")

#%% FQ maps
g = 0.5
ds = np.linspace(-1, 1, 10000)
ns = np.arange(1, 5, 1)

fig, axs = plt.subplots(figsize=FIG2by1, tight_layout=True, ncols=2)

ax = axs[0]
for n in ns:
    pmap_n = PoincareMap(n)
    ys = np.array([pmap_n.F_map(d, g=g) for d in ds])
    # Filter out complex values of F.
    _ys = ys[np.isreal(ys)].real
    _ds = ds[np.isreal(ys)]
    ax.plot(_ds, _ys, c="C0")

    # Add asymptotes where possible.
    d_a = pmap_n.d_asymptote()
    ax.axvline(d_a, c="k", ls=":")
    if n < 3:
        ax.text(d_a + 0.1, 0, r"$d_a(%s)$" % n)

# Add n numbers
ax.text(1.01, 375, r"$1$")
ax.text(1.01, 347, r"$2$")
ax.text(1.01, 325, r"$3$")
ax.text(1.01, 305, r"$4$")

ax.axvline(pmap.dsup, c="grey", ls="--")
ax.text(pmap.dsup+0.04, 0, r'$d_s$')

ax.set(
    xlabel=r'$d^\star$',
    ylabel=r'$\Delta t=F_n(d^\star)$',
    xlim=(-1, 1),
    ylim=(-20, 400)
)

ax = axs[1]
delts = np.linspace(-200, 500, 10000)

for n in ns:
    pmap_n = PoincareMap(n)
    ys = np.array([pmap_n.Q_map(delt, g=g) for delt in delts])
    ax.plot(delts, ys, c='C0')

# Add n numbers
ax.text(500, 1.38, r"$1$")
ax.text(500, 1.25, r"$2$")
ax.text(500, 1.14, r"$3$")
ax.text(500, 1.05, r"$4$")


intersection = np.log(g/(pmap.gstar))*pmap.tauk
ax.axvline(intersection, c='grey', ls='--')
ax.text(intersection+10, 0,
        r'$\tau_s\ln{\left(\frac{\bar g}{g^{\star}}\right)}$')


ax.set(xlabel=r'$\Delta t$',
       ylabel=r'$d^\star=Q_n(\Delta t)$',
       xlim=(-30, 500))

for idx, ax in enumerate(axs):
    ax.set_title(string.ascii_uppercase[idx], loc='left')

fig.savefig(paths.figures / "FQ.pdf")
