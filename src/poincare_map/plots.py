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

from poincare_map.poincare_map import PoincareMap
from poincare_map import ode
from poincare_map import paths


xppfile = "odes/mlml.ode"
model = PoincareMap(xppfile)


FIGWIDTH = 3.85
FIGSIZE = (FIGWIDTH, FIGWIDTH)
FIG2by1 = (FIGWIDTH*1.4, (FIGWIDTH*1.4)/2)

#%% Plotting

# TODO: Add patches

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
