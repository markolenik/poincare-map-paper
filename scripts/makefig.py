#%%
import os
import tqdm
from functools import partial
import pickle
import pandas as pd

import matplotlib

matplotlib.use("pgf")
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
rcp["xtick.labelsize"] = "small"
rcp["ytick.labelsize"] = "small"
rcp["axes.labelsize"] = "x-large"
rcp["axes.titlesize"] = "xx-large"
rcp["figure.titlesize"] = "large"
rcp["font.size"] = 9
# Use latex preamble.
rcp["pgf.rcfonts"] = False
rcp["pgf.texsystem"] = "pdflatex"
rcp["pgf.preamble"]= "\n".join([  # plots will use this preamble
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[detect-all,locale=UK]{siunitx}",
        r"\usepackage{amssymb}",
    ])

matplotlib.rcParams.update(rcp)

np.seterr(divide="ignore")
np.seterr(all="ignore")

FIGWIDTH = 3.85
FIGSIZE = (FIGWIDTH, FIGWIDTH)
FIG2by1 = (FIGWIDTH * 1.4, (FIGWIDTH * 1.4) / 2)

ml = ode.ML(paths.ml_file)
mlml = ode.ML(paths.mlml_file)
pmap = PoincareMap()

df = pd.read_pickle(paths.numeric_bif_diagram)
df_ana = pd.read_pickle(paths.analytic_bif_diagram)


#%% Nullclines

# Compute spiking solution.
lc = ml.find_limit_cycle()

# Plot.
fig, axs = plt.subplots(ncols=2, tight_layout=True, figsize=FIG2by1)

# A: Nullclines.
vs = np.linspace(-69, 60, 100)
vinf = np.vectorize(ml.vinf)
winf = np.vectorize(ml.winf)

ax = axs[0]
ax.plot(vs, vinf(vs), color="C0")
ax.plot(vs, winf(vs), color="C1")
ax.axvline(ml.pars["vtheta"], c="k", ls="--", lw=0.7)
(line,) = ax.plot(lc["v"], lc["w"], color="black")
vfp = ml.fixed_point()
wfp = ml.winf(vfp)  # type: ignore
if vfp:
    ax.plot(vfp, wfp, c="C1", marker="o", markeredgecolor="k")
    ax.text(vfp - 6, wfp + 0.03, r"$p_f$")

helpers.add_arrow_to_line2D(
    ax, line, arrow_locs=[0.05, 0.35, 0.6, 0.8], arrowstyle="->"
)
ax.text(ml.pars["vtheta"] - 10, 0.48, r"$v_\theta$")

# Add inhibited vnullcline

ax.text(-53, 0.53, r"$v_\infty$")
ax.text(8, 0.53, r"$w_\infty$")

ax.set(
    ylim=(-0.02, 0.6),
    xlabel=r"$v$ (\si{mV})",
    ylabel=r"$w$",
    xticks=np.arange(-60, 80, 40),
    yticks=np.arange(0, 0.6, 0.2),
)

## B: Trace.
ax = axs[1]
ax.plot(lc.index, lc["v"], color="C0")
ax.axhline(ml.pars["vtheta"], c="k", ls="--", lw=0.7)

ax.set(
    xlabel="time " + r"$(\si{ms})$",
    ylabel=r"$v$ (\si{mV})",
    xticks=np.arange(0, pmap.T, 100),
    ylim=(None, 70),
)

# Add Tact and Tinact
ax.axhline(0, linestyle="--", color="k", linewidth=0.5)
ax.text(300, ml.pars["vtheta"] - 5, r"$v_\theta$")
ax.axvline(0, linestyle="--", color="k", linewidth=0.5)
ax.axvline(pmap.Ta, linestyle="--", color="k", linewidth=0.5)
ax.axvline(pmap.T, linestyle="--", color="k", linewidth=0.5)
ax.annotate(
    "",
    xy=(-3, 55),
    xytext=(pmap.Ta + 3, 55),
    arrowprops=dict(arrowstyle="<->"),
    xycoords="data",
)
ax.annotate(
    "",
    xy=(pmap.Ta - 3, 55),
    xytext=(pmap.T + 3, 55),
    arrowprops=dict(arrowstyle="<->"),
    xycoords="data",
)
ax.text(8, 60, r"$T_{a}$")
ax.text(200, 60, r"$T_{s}$")

for idx, ax in enumerate(axs):
    ax.set_title(string.ascii_uppercase[idx], loc="left")
fig.savefig(paths.figures / "nullclines.pdf")


#%% All burst solutions
## Compute solutions
gs = [0.35, 0.45, 0.5, 0.53, 0.6]
sols = []
for g in gs:
    sol = mlml.run(g=g, transient=2000, total=6500)
    sol.index = sol.index - sol.index[0]
    sols.append(sol)

## Plot
fig, axs = plt.subplots(
    nrows=len(gs), tight_layout=True, figsize=(FIGWIDTH, 6), sharex=True
)

for idx, ax in enumerate(axs[:-1]):
    sol = sols[idx]
    ax.plot(sol.index, sol["v1"])
    ax.plot(sol.index, sol["v2"])

    # ax.autoscale(axis='x', tight=True)

# Suppressed
sol = sols[-1]
ax = axs[-1]
ax.plot(sol.index, sol["v1"])
ax.plot(sol.index, sol["v2"])
ax.set(xlabel="time " + r"$(\si{ms})$", xticks=np.arange(0, 4502, 1000))

for idx, ax in enumerate(axs):
    if idx == len(axs) - 1:
        title = (
            r"suppressed solution $(\bar g = %s \, \si{mS/cm^{2}})$" % gs[idx]
        )
    else:
        title = r"%s:%s solution $(\bar g = %s \, \si{mS/cm^{2}})$" % (
            idx + 1,
            idx + 1,
            gs[idx],
        )
    ax.set(ylabel=r"$v_1,v_2$", ylim=(-70, 50))
    ax.set_title(title, fontsize=9, loc="center")
    ax.set_title(string.ascii_uppercase[idx], loc="left")
fig.savefig(paths.figures / "burst-sols.pdf")


#%% Numeric bifurcation diagram
ns = df['n'].unique()

fig, ax = plt.subplots(tight_layout=True)
# Plot numeric diagram
for n in ns:
    dfn = df[df["n"] == n]
    ax.plot(dfn["g"], dfn["period"], c="C0", lw=2.5)


# Highlight bi-stability regions.
for n1, n2 in zip(ns, ns[1:]):
    n1_right = df.loc[df['n']==n1]['g'].max()
    n2_left = df.loc[df['n']==n2]['g'].min()
    ax.axvspan(n1_right, n2_left, ymin=0, ymax=1, color="C0", alpha=0.2)

# Add dotted lines with higher-order solutions.
ax.axvline(df["g"].max(), c="grey", ls="--")
ax.axvline(pmap.gsup, c="grey", ls="--")

# Add suppressed period
sup_gs = np.linspace(pmap.gsup, 0.62, 10)
ax.plot(sup_gs, pmap.T * np.ones(len(sup_gs)), ls="--", c="C0")

# Period T note
ax.annotate(r"$T$", (1.1, pmap.T), (1.1 + 0.05, pmap.T - 100))

ax.set(
    xlim=(0.3, sup_gs[-1]),
    xlabel=r"$\bar g$ " + r"$(\si{mS/cm^{2}})$",
    ylabel="period " + r"(\si{ms})$",
)

ax.text(0.35, 890, r"$1$")
ax.text(0.42, 1560, r"$2$")
ax.text(0.5, 2350, r"$3$")
ax.text(0.53, 3100, r"$4$")
ax.text(0.56, 3850, r"$5$")
ax.text(0.56, 3850, r"$5$")
ax.text(sup_gs[0] + 0.003, pmap.T + 50, "suppressed", fontsize=10)
ax.text(sup_gs[-1] + 0.002, pmap.T - 100, r"$T$", fontsize=13)

fig.savefig(paths.figures / "bif-diagram.pdf")


#%% Free-quiet

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

# A: Plot voltage lc.
ax = axs[0]
ax.plot(lc.index, lc["v1"])
ax.plot(lc.index, lc["v2"])
ax.set(ylim=(-80, 140), ylabel=r"$v_{1,2}$")

# Add delta-t patches.
delt = period / 2 - poincare.T * (n - 1) - poincare.Ta
# Patch start time
delt_t0s = [spike_times[-1] + poincare.Ta, period - delt]
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
arrow_startpoints[-1] += pmap.Ta
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
        text = r"$(n-1)T + T_{a} + 2\Delta t$"
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
    (2 * poincare.T + poincare.Ta, 0),
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
    xticks=[0, 2 * poincare.T + poincare.Ta, period],
    xticklabels=[
        r"$0$",
        r"$(n-1)T + T_{a}$",
        r"$P_n=2((n-1)T + \Delta t + T_{a})$",
    ],
)
ax.autoscale(axis="x", tight=True)

# Free-quiet patches
rect_free = patches.Rectangle(
    (0, 0),
    2 * poincare.T + poincare.Ta,
    1.0,
    alpha=0.1,
    color="C0",
    transform=ax.transData,
)
rect_quiet = patches.Rectangle(
    (2 * poincare.T + poincare.Ta, 0),
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
    0.21,
    text_y,
    r"active: $\bar gs>g^{\star}$",
    horizontalalignment="center",
    transform=ax.transAxes,
)
ax.text(
    0.7,
    text_y,
    "silent",
    horizontalalignment="center",
    transform=ax.transAxes,
)

# Add subplot titles.
for idx, ax in enumerate(axs):
    # ax.set(title=string.ascii_uppercase[idx], loc="left")
    ax.set_title(string.ascii_uppercase[idx], loc="left")
    ax.set_yticks([])

fig.savefig("figures/free-quiet.pdf")

#%% Delta t

fig, ax = plt.subplots(figsize=FIGSIZE, tight_layout=True)

# Plot delta-t
for n in df["n"].unique():
    dfn = df.loc[df["n"] == n]
    delta_ts = dfn["lc"].apply(partial(helpers.calc_delta_t, Tact=pmap.Ta))
    ax.plot(dfn["g"], delta_ts, color="C0")

# Place Tinact on top horizontal line
ax.text(
    0.49,
    pmap.Ts + 1.0,
    r"$T_{s}$",
    horizontalalignment="center",
    verticalalignment="bottom",
)

# Place ns
ax.text(0.3, 296.5, r"$1$")
ax.text(0.37, 304, r"$2$")
ax.text(0.45, 315, r"$3$")
ax.text(0.5, 320, r"$4$")
ax.text(0.55, 323.5, r"$5$")

# Add vertical line indicating T (correct for a time step).
ax.axhline(y=pmap.Ts + 0.5, c="grey", linestyle="--", lw=1.5)

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

# A: F-map
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
ax.text(pmap.dsup + 0.04, 0, r"$d_s$")

ax.set(
    xlabel=r"$d^\star$",
    ylabel=r"$\Delta t=F_n(d^\star)$",
    xlim=(-1, 1),
    ylim=(-20, 400),
)

# B: Q-map
ax = axs[1]
delts = np.linspace(-200, 500, 10000)

for n in ns:
    pmap_n = PoincareMap(n)
    ys = np.array([pmap_n.Q_map(delt, g=g) for delt in delts])
    ax.plot(delts, ys, c="C0")

# Add n numbers
ax.text(500, 1.38, r"$1$")
ax.text(500, 1.25, r"$2$")
ax.text(500, 1.14, r"$3$")
ax.text(500, 1.05, r"$4$")


intersection = np.log(g / (pmap.gstar)) * pmap.tauk
ax.axvline(intersection, c="grey", ls="--")
ax.text(
    intersection + 10,
    0,
    r"$\tau_\kappa\ln{\left(\frac{\bar g}{g^{\star}}\right)}$",
)


ax.set(
    xlabel=r"$\Delta t$", ylabel=r"$d^\star=Q_n(\Delta t)$", xlim=(-30, 500)
)

for idx, ax in enumerate(axs):
    ax.set_title(string.ascii_uppercase[idx], loc="left")

fig.savefig(paths.figures / "FQ-maps.pdf")

#%% Pi map

g = 0.5
ds = np.linspace(-1, 1, 10000)
ns = np.arange(1, 5, 1)

fig, axs = plt.subplots(ncols=2, tight_layout=True, figsize=FIG2by1)


# A: Pi with all curves.
ax = axs[0]

for n in ns:
    pmap_n = PoincareMap(n)
    ys = np.array([pmap_n.P_map(d, g=g) for d in ds])
    realidx = np.isreal(ys)
    ax.plot(ds[realidx], ys[realidx], c="C0")


# Add first 2 asymptotes.
for n in ns[:2]:
    pmap_n = PoincareMap(n)
    d_a = pmap_n.d_asymptote()
    ax.axvline(d_a, c="k", ls=":")
    ax.text(d_a - 0.33, 0, r"$d_a(%s)$" % n)

# # Add n numbers.
n_locs = [(0.03, -0.7), (-0.60, -0.7), (-0.8, 0.67), (-0.8, 0.95)]
for loc, n in zip(n_locs, ns):
    ax.text(
        loc[0],
        loc[1],
        r"$%s$" % n,
        verticalalignment="center",
        horizontalalignment="left",
    )

ax.set(ylabel=r"$\Pi_n(d^\star)$")

# B: Pi with n=2 and varying g.
gs = [0.0005, 0.0015, 0.005]
ax = axs[1]
for g in gs:
    pmap_n = PoincareMap(2)
    ys = np.array([pmap_n.P_map(d, g) for d in ds])
    realidx = np.isreal(ys)
    ax.plot(
        ds[realidx], ys[realidx], label=r"$\bar g\approx%s$" % round(g, 4)
    )
ax.legend()
ax.set(ylabel=r"$\Pi_2(d^\star)$")

for idx, ax in enumerate(axs):
    # Diagonal
    ax.plot(ds, ds, c="k", ls="--")
    ax.set(xlabel=r"$d^\star$", xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    ax.locator_params(axis="x", tight=True, nbins=4)
    ax.locator_params(axis="y", tight=True, nbins=4)
    ax.set_title(string.ascii_uppercase[idx], loc="left")

fig.savefig(paths.figures / "Pn-map.pdf")

#%% Folds

ns = [1, 2, 3, 4, 5]
gs = np.geomspace(0.000001, 0.6, 100)
ds = np.linspace(-1.5, 1, 5000)
gmax = gs[-1]

fig, axs = plt.subplots(figsize=FIG2by1, tight_layout=True, ncols=2)

# A: Compute analytic bifurcation curves of d.
curves = []
for n in ns:
    pmap_n = PoincareMap(n)
    if n > 1:
        d_bif, g_bif = pmap_n.critical_fp(0.1)
    else:
        d_bif, g_bif = pmap_n.critical_fp(0.5)
    ds_stable = ds[ds > d_bif]
    ds_unstable = ds[ds < d_bif]
    curve_stable = [pmap_n.G(d) for d in ds_stable]
    curve_unstable = [pmap_n.G(d) for d in ds_unstable]
    axs[0].plot(curve_stable, ds_stable, color="C0")
    axs[0].plot(curve_unstable, ds_unstable, color="C0", linestyle="dashed")


axs[0].set(
    xlim=(-0.05, gmax),
    ylim=(-1.5, 1.3),
    xlabel=r"$\bar g$ " + r"$(\si{mS/cm^{2}})$",
    ylabel=r"$d^{\star}_{f}$",
)

axs[0].locator_params(axis="x", tight=False, nbins=6)
axs[0].text(gmax + 0.01, 0.0, r"$1$")
axs[0].text(gmax + 0.01, -0.70, r"$2$")

patch_stable = Line2D([], [], linestyle="-", linewidth=1, label="stable")
patch_unstable = Line2D([], [], linestyle=":", linewidth=1, label="unstable")
axs[0].legend(handles=[patch_stable, patch_unstable], loc=(0.5, 0.9))

# B: Compute period curves.
ax = axs[1]
for n in ns:
    pmap_n = PoincareMap(n)
    dfn = df[df["n"] == n]
    # analytic
    ax.plot(gs, [pmap_n.period(g) for g in gs], color="C0")
    # numeric
    ax.plot(dfn["g"], dfn["period"], color="C1")

ax.set(
    ylim=(500, 4300),
    xlabel=r"$\bar g$ " + r"$(\si{mS/cm^{2}})$",
    ylabel="period " + r"$(\si{ms})$",
)

# Add labels
for n in ns:
    axs[1].text(gmax + 0.01, PoincareMap(n).period(gmax) - 50, r"$%s$" % n)

for idx, ax in enumerate(axs):
    ax.set_title(string.ascii_uppercase[idx], loc="left")

patch_ana = Line2D([], [], color="C0", linewidth=1, label="analytic")
patch_num = Line2D([], [], color="C1", linewidth=1, label="numeric")
axs[1].legend(handles=[patch_ana, patch_num], loc=(0.5, 0.9))

fig.savefig(paths.figures / "folds.pdf")

#%% Final bifurcation diagram

ns = np.arange(1, 10)

fig, ax = plt.subplots(tight_layout=True)

# Plot analytic diagram on top
for n in df_ana["n"].unique():
    dfn_ana = df_ana[df_ana["n"] == n]
    ax.plot(dfn_ana["g"], dfn_ana["period"], c="C0", lw=2.5)

# Plot numeric diagram
for n in df["n"].unique():
    dfn = df[df["n"] == n]
    ax.plot(dfn["g"], dfn["period"], c="C1", lw=1.3)

ax.set(
    xlim=(0.3, pmap.gsup),
    xlabel=r"$\bar g$ " + r"$(\si{mS/cm^{2}})$",
    ylabel="period " + r"(\si{ms})$",
)

# Add legend
patch_ana = Line2D([], [], color="C0", linewidth=1, label="analytic")
patch_num = Line2D([], [], color="C1", linewidth=1.5, label="numeric")

fig.legend(handles=[patch_ana, patch_num], loc=(0.2, 0.8), prop={"size": 12})


ax.text(0.35, 890, r"$1$")
ax.text(0.42, 1560, r"$2$")
ax.text(0.5, 2350, r"$3$")
ax.text(0.53, 3100, r"$4$")
ax.text(0.56, 3850, r"$5$")
ax.text(0.57, 4560, r"$6$")
ax.text(0.5760, 5380, r"$7$")
ax.text(0.58, 6100, r"$8$")
ax.text(0.5815, 6850, r"$9$")

fig.savefig(paths.figures / "final-bif")

#%% Gstar numerical diagram and release delay


def gstar(df_row):
    """Find gstar at index of first spike of cell 2 from LC solution."""
    lc = df_row["lc"]
    spike_times = helpers.spike_times(lc["v2"])
    first_spike_state = lc.loc[spike_times[0]]
    return df_row["g"] * first_spike_state["s1"]


def release_delay(df_row):
    """Compute the release time, that is when gs=g*."""
    sol = df_row["lc"]
    tks2 = helpers.spike_times(sol["v2"])
    # Get rid of transient
    _sol = sol.loc[sol.index > 5]
    # Release time as defined by the release condition
    release_condition_satisfied = _sol.loc[_sol["gtot1"] < pmap.gstar].index[0]
    return release_condition_satisfied - tks2[0]


gstars = df.apply(gstar, axis=1)
release_delays = df.apply(release_delay, axis=1)
df = df.assign(gstar=gstars, release_delay=release_delays)


# Compute gstar for suppressed solution
sol = mlml.run(g=pmap.gsup, total=5000)
gstar_numeric = sol.loc[sol.index > 3000]["gtot2"].min()

fig, axs = plt.subplots(
    tight_layout=True, nrows=2, sharex=True, figsize=(FIGWIDTH, 5)
)
for n in df.n.unique():
    dfn = df[df.n == n]
    # A: Plot gstar.
    axs[0].plot(dfn.g, dfn.gstar, ".", label=f"${n}:{n}$", ms=2)
    # B: Plot release delay.
    axs[1].plot(dfn["g"], dfn["release_delay"], ".", ms=2)

axs[0].axhline(gstar_numeric, linestyle="--", color="k")
axs[0].set(ylim=(0.0066, 0.00705))
axs[0].set_ylabel("Release conductance " + r"$(\si{mS/cm^{2}})$", fontsize=10)
axs[0].text(
    0.57,
    gstar_numeric - 0.00004,
    r"$g^\star$",
    fontsize=11,
)
axs[0].legend(loc=2)

# axs[1].axhline(0, ls="--", c="k")
axs[1].set(
    xlabel=r"$\bar g$ " + r"$(\si{mS/cm^{2}})$",
    xlim=(0.3, 0.6),
    ylim=(-4, 3),
)
axs[1].set_ylabel("Release delay " + r"$(\si{ms})$", fontsize=10)

for idx, ax in enumerate(axs):
    ax.set_title(string.ascii_uppercase[idx], loc="left")

fig.savefig(paths.figures / "gstar-diag.pdf")

# #%% ISI stats
# def calc_ISI(sol):
#     """Return list of ISIs."""
#     isis1 = np.diff(helpers.spike_times(sol["v1"]).values)
#     isis2 = np.diff(helpers.spike_times(sol["v2"]).values)
#     return list(np.concatenate([isis1, isis2]))


# all_isis = list(np.concatenate(df["lc"].apply(calc_ISI).values))
# mean = np.mean(all_isis)
# std = np.std(all_isis)
# interval = [min(all_isis), max(all_isis)]
# print(f"mean: {mean}")
# print(f"std: {std}")
# print(f"interval: [{interval[0]}, {interval[1]}]")

# fig, ax = plt.subplots()
# ax.hist(all_isis, bins="auto")
# fig.savefig("tmp/ISI-hist.pdf")


#%% Exploring tauk
def calc_ISIs(sol: pd.DataFrame) -> np.ndarray:
    """Compute nth ISI."""
    spike_times = helpers.spike_times(sol["v"])
    ISIs = np.diff(spike_times)
    return ISIs


vfp = ml.fixed_point(gtot=pmap.gstar)
wfp = ml.winf(vfp)  # type: ignore

sols_tauk = []
sols_delay = []
tauks = np.linspace(10, 800, 30)
for tauk in tqdm.tqdm(tauks):
    # tauk
    gtot0 = pmap.gstar
    v0 = ml.pars["vtheta"] - 1  # Subtract 1 to have a root at t=0.
    w0 = wfp
    ics = np.array([v0, w0, gtot0])
    sol = ml.run(total=2000, ics=ics, tauk=tauk, dt=0.5)
    sols_tauk.append(sol)

    # delay
    gtot0 = 0.45
    v0 = ml.fixed_point(gtot=gtot0)
    w0 = ml.winf(v0)  # type: ignore
    ics = np.array([v0, w0, gtot0])
    sol = ml.run(total=2000, ics=ics, tauk=tauk, dt=0.1)
    sols_delay.append(sol)

ISI_list = [calc_ISIs(sol) for sol in sols_tauk]

fig, ax = plt.subplots()
ax.plot(tauks, [ISIs[0] for ISIs in ISI_list], label=r"$ISI_1$")
ax.plot(tauks, [ISIs[1] for ISIs in ISI_list], label=r"$ISI_2$")
ax.plot(tauks, [ISIs[2] for ISIs in ISI_list], label=r"$ISI_3$")
ax.plot(tauks, np.ones(len(tauks)) * pmap.T, "k--")
ax.set(xlabel=r"$\tau_k$ $(\si{ms})$", ylabel=r"$ISI$ $(\si{ms})$", xlim=(0, 800), ylim=(350, 525))
ax.legend()

# Add T
ax.text(750, pmap.T - 8, r"$T$", fontsize=12)

fig.savefig(paths.figures / "tauk-vs-ISI.pdf")


#%% Response stuff
# ================================================================================================

# Irregular solution
g = 0.581

sol = mlml.run(g=g, transient=480000, total=500000)
sol.index = sol.index - min(sol.index)

fig, ax = plt.subplots()
ax.plot(sol.index, sol["v1"], label=r"$v_1$")
ax.plot(sol.index, sol["v2"], label=r"$v_2$")
ax.set(xlabel="time " + r"$(\si{ms})$", ylabel=r"$v_i$ " + r"$(\si{mV})$")
ax.legend()
fig.savefig(paths.figures / "irregular.pdf")

# Comparison of Boses model to our update model (nullclines)

# Plot.
fig, axs = plt.subplots(ncols=2, tight_layout=True, figsize=FIG2by1)
ml2 = ode.ML(paths.ml_file)
ml2.pars["iapp"] = 6.0

# A: Bose
vs = np.linspace(-69, 60, 100)
vinf = np.vectorize(ml.vinf)
winf = np.vectorize(ml.winf)

ax = axs[0]
ax.plot(vs, vinf(vs), color="C0")
ax.plot(vs, vinf(vs, gtot=0.0068), color="C0", linestyle="--")
ax.plot(vs, winf(vs), color="C1")
vfp = ml.fixed_point()
wfp = ml.winf(vfp)  # type: ignore


ax.set(
    ylim=(-0.02, 0.6),
    xlabel=r"$v$",
    ylabel=r"$w$",
    xticks=np.arange(-60, 80, 40),
    yticks=np.arange(0, 0.6, 0.2),
)
ax.set_title(r"$I=3.8$ " + r"$\si{\mu A/cm^2}$", fontsize=10)

# B: Changed model
vs = np.linspace(-69, 60, 100)
vinf = np.vectorize(ml2.vinf)
winf = np.vectorize(ml2.winf)

ax = axs[1]
ax.plot(vs, vinf(vs), color="C0")
ax.plot(vs, vinf(vs, gtot=0.03804), color="C0", linestyle="--")
ax.plot(vs, winf(vs), color="C1")
vfp = ml2.fixed_point()
wfp = ml2.winf(vfp)  # type: ignore

# Add inhibited vnullcline

ax.set(
    ylim=(-0.02, 0.6),
    xlabel=r"$v$",
    ylabel=r"$w$",
    xticks=np.arange(-60, 80, 40),
    yticks=np.arange(0, 0.6, 0.2),
)
ax.set_title(r"$I=6$ " + r"$\si{\mu A/cm^2}$", fontsize=10)

fig.savefig(paths.figures / "model-changes.pdf")
