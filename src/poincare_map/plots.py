"""Functions to build figures."""

#%% Initialise
import os
import pickle
import pandas as pd

import matplotlib
# NOTE: Maybe plot directly as pgf and import?
# matplotlib.use('pgf')
from matplotlib import patches
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import string
import numpy as np

import pyxpp.pyxpp as xpp
from poincare_map.poincare_map import PoincareMap
from poincare_map import ode
from poincare_map import paths

# plt.ioff()

xppfile = "odes/mlml.ode"
model = PoincareMap.from_ode(xppfile)


FIGWIDTH = 3.85
FIGSIZE = (FIGWIDTH, FIGWIDTH)
FIG2by1 = (FIGWIDTH*1.4, (FIGWIDTH*1.4)/2)

#%% Plotting

# TODO: Add patches
def free_quiet() -> Figure:
    """Illustration of free and quiet phases  with voltage and synpase 
    variables."""
    # Compute 3-3 solution.
    TRANSIENT = 3000
    TOTAL = 12000
    G = 0.5
    N = 2
    sol = ode.run(xppfile, transient=3000, total=10000, g=G)
    lc = ode.find_lc(sol, norm=True)

    period = lc.t.iloc[-1]
    idxs, _ = ode.spikes(lc)
    # Append last point to first to make it look nice.
    idxs = np.concatenate((idxs, [-1]))

    fig, axs = plt.subplots(nrows=3, tight_layout=True, sharex=True,
                            figsize=(FIGWIDTH, FIGWIDTH*1.7))

    # A: Plot voltage trace.
    ax = axs[0]
    ax.plot(lc.t, lc.v1)
    ax.plot(lc.t, lc.v2)
    ax.set_ylim((-80, 140))
    ax.set_ylabel(r'$v_{1,2}$')

    # # Add arrows between ds.
    # OFFSET = 16
    # v0 = max(lc.v1)
    # for idx0, idx1 in zip(idxs, idxs[1:]):
    #     t0 = lc.t.iloc[idx0]
    #     t1 = lc.t.iloc[idx1]
    #     midpoint = t0 + (t1-t0)/2.
    #     p = patches.FancyArrowPatch((t0, v0), (t1, v0),
    #                                 connectionstyle='arc3, rad=-0.3',
    #                                 arrowstyle='->', mutation_scale=8)
    #     ax.add_patch(p)

    #     # Put text on top of patch.
    #     if idx1 == -1:
    #         text_y = v0+3.0*OFFSET
    #         text = r'$(n-1)T + 2\Delta t$'
    #     else:
    #         text_y = v0+OFFSET
    #         text = r'$T$'

    #     ax.text(midpoint, text_y, text,
    #             horizontalalignment='center')

    # # Add delta-t patches.
    # delt = period/2. - pars.period*N
    # delt_t0s = [lc.t.iloc[idxs[-2]], lc.t.iloc[idxs[-1]] - delt]
    # min_v = min(lc['v1'])
    # max_v = max(lc['v1'])
    # height = max_v - min_v
    # for delt_t0 in delt_t0s:
    #     rect = patches.Rectangle((delt_t0, min_v), delt, 2*height,
    #                              alpha=0.2, color='grey',
    #                              transform=ax.transData)
    #     ax.add_patch(rect)
    #     ax.text(delt_t0 + delt/2, 1.1*height, r'$\Delta t$',
    #             horizontalalignment='center', transform=ax.transData)

    # B: Plot synaptic variables.
    ax = axs[1]
    ax.plot(lc.t, lc.gtot1, c='C0')
    ax.plot(lc.t, model.gstar*np.ones(len(lc.t)), c='k', linestyle='--')
    ax.set_ylabel(r'$\bar g s_1$')
    ax.set_xticks([0, 2*model.T, lc.t.iloc[-1]])
    ax.autoscale(axis='x', tight=True)
    ax.text(840, 0.07, r'$g^{\star}$')

    # # Add first delta-t patch here as well.
    # delt = period/2. - pars.period*N
    # delt_t0 = lc.t.iloc[idxs[-2]]
    # rect = patches.Rectangle((2*pars.period, 0), delt, 1.,
    #                          alpha=0.2, color='grey',
    #                          transform=ax.transData)

    # ax.add_patch(rect)
    # ax.text((delt_t0 + delt/2)/period, 0.9, r'$\Delta t$',
    #         horizontalalignment='center', transform=ax.transAxes)


    # C: Plot d.
    ax = axs[2]
    ax.plot(lc.t, lc.d1, c='C0', lw=2)
    ax.set_xlabel('time')
    ax.set_ylabel(r'$d_1$')
    ax.set_xticks([0, 2*model.T, lc.t.iloc[-1]])
    ax.set_xticklabels([r'$0$', r'$(n-1)T$', r'$P_n=2\Delta t + 2(n-1)T$'])
    ax.autoscale(axis='x', tight=True)

    # # Free-quiet patches
    # rect_free = patches.Rectangle((0, 0), 2*pars.period, 1.,
    #                               alpha=0.1, color='C0',
    #                               transform=ax.transData)
    # rect_quiet = patches.Rectangle((2*pars.period, 0), period-2*pars.period, 1.,
    #                                alpha=0.1, color='C1',
    #                                transform=ax.transData)
    # ax.add_patch(rect_free)
    # ax.add_patch(rect_quiet)
    # text_y = 0.85
    # ax.text(0.2, text_y, r'free: $\bar gs>g^{\star}$',
    #         horizontalalignment='center', transform=ax.transAxes)
    # ax.text(0.7, text_y, 'quiet', horizontalalignment='center',
    #         transform=ax.transAxes)

    # Figure titles
    for idx, ax in enumerate(axs):
        ax.set_title(string.ascii_uppercase[idx], loc='left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticks([])

    return fig


def full_bifurcation_diagram(num_bif_diag: pd.DataFrame, model: PoincareMap) -> Figure:
    """Plot both numerical and analytic bifurcation diagrams."""

    ns = np.arange(1, 10)

    # Compute numeric
    branches = []
    # Plot numeric fixed points.
    fig, ax = plt.subplots(tight_layout=True)
    for idx, n in enumerate(ns):
        # Compute borders
        if n>1:
            L = model.left_branch_border(pars, n)
        else:
            L = 0
        R = model.right_branch_border(pars, n)
        # Compute period
        db, gb = model.critical_fp(pars, n)
        gs = np.linspace(gb+0.0001, pars.g_sup, 1000)
        dfs = [model.phi_stable(pars._replace(g=g), g, n) for g in gs]
        periods = [gtoP(pars, g, n) for g in gs]
        df = pd.DataFrame({'period': periods,
                        'g': gs})
        if n>1:
            df_b = df[(df['g']<R) & (df['g']>L)]
        else:
            df_b = df[df['g']<R]

        branches.append(df_b)
        pickle.dump(branches, open(branches_dat, 'wb'))
    else:
        branches = pickle.load(open(branches_dat, 'rb'))

    for branch in branches:
            ax.plot(branch['g'], branch['period'], c='C0', lw=2.5)

    # # plot period from ode integration
    # for n in dat['n'].unique():
    #     odebranch = dat[dat['n']==n]
    #     # ax.plot(odebranch['g'], odebranch['period'], c='c1', lw=1.3)

    ax.set_xlim(0.45, pars.g_sup)
    ax.set_ylim(bottom=200)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(r'$\bar g$ ' + r'$(\si{mS/cm^{2}})$')
    ax.set_ylabel('period (\si{ms})')

    # Add legend
    patch_ana = Line2D([], [], color='C0', linewidth=1, label='analytic')
    patch_num = Line2D([], [], color='C1', linewidth=1.5, label='numeric')

    fig.legend(handles=[patch_ana, patch_num], loc=(0.2, 0.8),
               prop={'size': 12})


    ax.text(0.55, 500, r'$1$')
    ax.text(0.65, 900, r'$2$')
    ax.text(0.8, 1350, r'$3$')
    ax.text(0.9, 1800, r'$4$')
    ax.text(0.97, 2250, r'$5$')
    ax.text(0.99, 2650, r'$6$')
    ax.text(1.005, 3100, r'$7$')
    ax.text(1.013, 3520, r'$8$')
    ax.text(1.015, 3950, r'$9$')

    return fig



fig = free_quiet()
fig.savefig(paths.figures / "free-quiet.pdf")

fig = full_bifurcation_diagram()
fig.savefig(paths.figures / "full-bifurcation-diagram.pdf")

#%% Try plotting analytic bif

ns = np.arange(1, 10)
branches = []
# Plot numeric fixed points.
for idx, n in enumerate(ns):
    # Compute borders
    if n>1:
        L = model.left_branch_border(pars, n)
    else:
        L = 0
    R = model.right_branch_border(pars, n)
    # Compute period
    db, gb = model.critical_fp(pars, n)
    gs = np.linspace(gb+0.0001, pars.g_sup, 1000)
    dfs = [model.phi_stable(pars._replace(g=g), g, n) for g in gs]
    periods = [gtoP(pars, g, n) for g in gs]
    df = pd.DataFrame({'period': periods,
                    'g': gs})
    if n>1:
        df_b = df[(df['g']<R) & (df['g']>L)]
    else:
        df_b = df[df['g']<R]

    branches.append(df_b)

#%% Compute analytic bif
from scripts import compute_bif

bifdat = compute_bif.compute_analytic_bif_diagram([1, 2, 3, 4, 5, 6, 7])
