# %% Initialise
import itertools
import multiprocessing
import os
import pickle
import string
import warnings
from os import path

import matplotlib
matplotlib.use('pgf')

import pandas as pd
import pyxpp.pyxpp as xpp
import scipy as sp
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from pandas import DataFrame
from scipy.misc import derivative
from pathlib import Path

from poincare_map import helpers
from poincare_map import model
from poincare_map import ode
from poincare_map import paths

import numpy as np

plt.ioff()

warnings.filterwarnings("ignore", category=DeprecationWarning)
# Ignore devide by zero
np.seterr(divide='ignore')
np.seterr(all='ignore')


xppfile = str(paths.odes / "mlml.ode")
CENTER_TITLE_SIZE = 11
FIGWIDTH = 3.85
FIGSIZE = (FIGWIDTH, FIGWIDTH)
FIG2by1 = (FIGWIDTH*1.4, (FIGWIDTH*1.4)/2)

EQSIZE = 15                     # Font size of equaitons.

rcp = matplotlib.rcParamsDefault
rcp['figure.max_open_warning'] = 0
rcp['axes.unicode_minus'] = False
rcp['text.usetex'] = True
rcp['pgf.rcfonts'] = False
rcp['pgf.texsystem'] = 'pdflatex'
rcp['xtick.labelsize'] = 'small'
rcp['ytick.labelsize'] = 'small'
rcp['axes.labelsize'] = 'x-large'
rcp['axes.titlesize'] = 'xx-large'
rcp['figure.titlesize'] = 'large'
rcp['font.size'] = 9
# Use latex preamble.
rcp['pgf.preamble'] = r'\usepackage{amsmath} \usepackage{siunitx}'

matplotlib.rcParams.update(rcp)

p = xpp.read_pars(xppfile)

# pars = model.Pars(p['g'], p['gstar'], p['taud'], p['taus'],
#                   p['period'], p['lambda'])


#%% Fig: nullclines-uncoupled
LABEL = 'nullclines-uncoupled'

vnull, wnull = ode.nullclines(XPPFILE, gtot=0)
sol = ode.run(XPPFILE, g=0)
lc = ode.find_lc(sol)
loop = lc.append(lc.iloc[0])    # Close the loop.

# Plot.
fig, axs = plt.subplots(ncols=2, tight_layout=True,
                        figsize=FIG2by1)

# Nullclines.
ax = axs[0]
ax.plot(vnull[:, 0], vnull[:, 1], color='C0')
ax.plot(wnull[:, 0], wnull[:, 1], color='C1')
line, = ax.plot(loop.v1, loop.w1, color='k')
fp_stable = [-5.53, 0.219]
ax.plot(fp_stable[0], fp_stable[1], c='C1', marker='o', markeredgecolor='k')
ax.text(fp_stable[0]+4, fp_stable[1]-0.01, r'$p_f$')

helpers.add_arrow_to_line2D(ax, line, arrow_locs=[0.05, 0.35, 0.6, 0.8],
                            arrowstyle='->')

ax.text(-45, 0.442, 'Jump Down')
ax.text(-4.5, 0.02, 'Jump Up')
ax.text(-60, 0.08, 'Silent\nState')
ax.text(30, 0.3, 'Active\nState')

ax.text(-50, 0.6, r'$v_\infty$')
ax.text(40, 0.9, r'$w_\infty$')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel(r'$v\; (\si{mV})$')
ax.set_ylabel(r'$w$')
ax.set_xticks(sp.arange(-60, 80, 40))
ax.set_yticks(sp.arange(0, 1, 0.3))
ax.autoscale(tight=True, axis='both')

# Trace.
ax = axs[1]
ax.plot(lc.t, lc.v1, color='C0')

ax.text(4, 5, 'Jump\nUp')
ax.text(60, -20, 'Jump\nDown')
ax.text(155, -47, 'Silent\nState')
ax.text(37, 25, 'Active\nState')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('time ' r'$(\si{ms})$')
ax.set_ylabel(r'$v\;(\si{mV})$')
ax.set_xticks(sp.arange(0, 220, 50))
ax.set_ylim(top=70)
ax.autoscale(tight=True, axis='x')

# Add Tact and Tinact
ax.axhline(0, linestyle='--', color='k', linewidth=0.5)
ax.axvline(50, linestyle='--', color='k', linewidth=0.5)
ax.axvline(217, linestyle='--', color='k', linewidth=0.5)
ax.annotate("", xy=(0, 55), xytext=(50, 55), arrowprops=dict(arrowstyle="<->"),
            xycoords='data')
ax.annotate("", xy=(50, 55), xytext=(217, 55), arrowprops=dict(arrowstyle="<->"),
            xycoords='data')
ax.text(15, 60, r'$T_{act}$')
ax.text(110, 60, r'$T_{inact}$')

for idx, ax in enumerate(axs):
    ax.set_title(string.ascii_uppercase[idx], loc='left')

fig.savefig(f'{FIGDIR}{LABEL}.pdf')
fig.savefig(f'{FIGDIR}{LABEL}.pgf')


#%% Compute T diagram.
LABEL = 'bif-diagram'
datfile = os.path.join(DATDIR, LABEL + '.dat')
NSTEPS = 100
TOTAL = 10000


if not path.isfile(datfile):
    print('COMPUTING BIFDIAGRAM')
    n0s = [1, 2, 3, 4, 5]
    p0s = [.4, .7, .8, .9, .96]
    parsteps = [0.01, 0.01, 0.01, 0.005, 0.001]
    curves = []
    for n0, p0, step in zip(n0s, p0s, parsteps):
        left = ode.cont(XPPFILE, 'g', g=p0, parstep=-step, nsteps=NSTEPS,
                        total=TOTAL)
        right = ode.cont(XPPFILE, 'g', g=p0, parstep=step, nsteps=NSTEPS,
                         total=TOTAL)
        curve = left.iloc[::-1].append(right.iloc[1:]).reset_index(drop=True)
        curves.append(curve)
    dat = pd.concat(curves)
    pickle.dump(dat, open(datfile, 'wb'))
else:
    dat = pickle.load(open(datfile, 'rb'))


## Compute ISI diagram.

isi_max = []
isi_min = []
for row in dat.itertuples():
    idx = getattr(row, 'Index')
    sol = getattr(row, 'sol')
    period = getattr(row, 'period')
    ids1, _ = ode.spikes(sol)
    tks1 = sol.t.iloc[ids1]
    # Take the big ISI.
    _isis = sp.diff(sp.append(tks1, period))
    isis1 = max(_isis)
    # T-isi (hacky but whatever, there are some outliers in data).
    isi_min.append(_isis[(_isis>pars.period-30)&(_isis<pars.period+50)])
    isi_max.append(isis1)

# Add a column with ISIs.
dat['ISI_max'] = isi_max
# isi_min = [sp.mean(isi) for isi in isi_min]
dat['ISI_T'] = isi_min

## Plot
fig, axs = plt.subplots(ncols=2, tight_layout=True,
                        figsize=FIG2by1)

# A: Period diagram
ax = axs[0]
nmax = max(dat['n'])
for n in range(nmax+1)[1:]:
    ndat = dat.loc[dat['n']==n]
    ax.plot(ndat['g'], ndat['period'], c='C0')
    ax.set_ylabel('period ' + r'$(\si{ms})$')
    ax.set_yticks(sp.arange(500, 2500, 500))

    # place text next to branch
    right_idx = ndat['g'].argmax()
    right_g = ndat.loc[right_idx]['g']
    left_g = ndat['g'].min()
    mid_g = left_g + (right_g - left_g)/2.
    right_P = ndat.loc[right_idx]['period']
    if n > 1:
        ax.text(mid_g-.02, right_P+20, r'$%s$' % n)
    else:
        ax.text(mid_g+0.1, right_P+15, r'$%s$' % n)

# Add suppressed solution.
gsup_range = sp.linspace(pars.g_sup, 1.1, 100)
ax.plot(gsup_range, sp.ones(len(gsup_range))*pars.period, c='C0')
ax.autoscale(axis='x', tight=True)
ax.text(pars.g_sup+0.03, pars.period+40, 'suppressed')
# ax.text(pars.g_sup+0.02, 1000, r'$\bar g_s$')

# Add dotted lines with higher-order solutions.
ax.axvline(right_g, c='grey', ls='--')
ax.axvline(pars.g_sup, c='grey', ls='--')

# Period T note
ax.annotate(r'$T$', (1.1, pars.period), (1.1+0.05, pars.period-100))


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# B: ISI diagram
ax = axs[1]
for n in range(nmax+1)[1:]:
    ndat = dat.loc[dat['n']==n]
    # Hacky: take only large ISIs first.
    ax.plot(ndat['g'], ndat['ISI_max'], c='C0')
    # Place text next to branch.
    right_idx = ndat['g'].argmax()
    right_g = ndat.loc[right_idx]['g']
    left_g = ndat['g'].min()
    mid_g = left_g + (right_g - left_g)/2.
    right_ISI = ndat.loc[right_idx]['ISI_max']
    if n > 1:
        ax.text(mid_g-.02, right_ISI+20, r'$%s$' % n)
    else:
        ax.text(mid_g+0.1, right_ISI-20, r'$%s$' % n)

# Plot also where ISI=T as scatter plot
gs = dat['g']
g_T_pairs = []
for index, row in dat.iterrows():
    pairs = itertools.product([row['g']], row['ISI_T'])
    g_T_pairs.append(pairs)
g_T_pairs = sp.array(list(itertools.chain(*g_T_pairs)))

ax.axhline(y=pars.period, c='grey', linestyle='--', lw=0.5)
ax.scatter(g_T_pairs[:, 0], g_T_pairs[:, 1], s=0.2)

ax.set_ylabel(r'$ISI/IBI$' + r'$(\si{ms})$')
ax.set_ylim((50, 1400))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.autoscale(axis='x', tight=True)

# Place text on top of branches.
ax.text(0.78, 235, r'$\approx T$', horizontalalignment='center',
        verticalalignment='bottom')

for idx, ax in enumerate(axs):
    ax.set_title(string.ascii_uppercase[idx], loc='left')
    ax.set_xlim(left=0.2, right=1.1)
    ax.set_xlabel(r'$\bar g$ ' + r'$(\si{mS/cm^{2}})$')

fig.savefig(f'{FIGDIR}{LABEL}.pdf')
fig.savefig(f'{FIGDIR}{LABEL}.pgf')



#%% Fig: nullclines-coupled
LABEL = 'nullclines-coupled'

gstar = xpp.read_pars(XPPFILE)['gstar']
n1_uncoupled, n2_uncoupled = ode.nullclines(XPPFILE, gtot=0)
n1_sup, n2_sup = ode.nullclines(XPPFILE, gtot=0.08)
n1_star, n2_star = ode.nullclines(XPPFILE, gstar)

## Plot.
fig, ax = plt.subplots(figsize=FIGSIZE, tight_layout=True)
for (n1, n2) in [[n1_uncoupled, n2_uncoupled], [n1_sup, n2_sup],
                 [n1_star, n2_star]]:
    ax.plot(n1[:, 0], n1[:, 1], color='C0')
    ax.plot(n2[:, 0], n2[:, 1], color='C1')

fp_stable = [-5.53, 0.219]
fp_sup = [-33.5, 0.006]
fp_star = [-20.55, 0.036]
ax.plot(fp_stable[0], fp_stable[1], c='C1', marker='o', markeredgecolor='k')
ax.plot(fp_sup[0], fp_sup[1], c='C0', marker='o', markeredgecolor='k')
ax.plot(fp_star[0], fp_star[1], c='C0', marker='o', markeredgecolor='k',
        fillstyle='left')
ax.plot(fp_star[0], fp_star[1], c='C1', marker='o', markeredgecolor='k',
        fillstyle='right')

ax.text(-50, 0.6, r'$v_\infty$')
ax.text(40, 0.93, r'$w_\infty$')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel(r'$v$')
ax.set_ylabel(r'$w$')
ax.set_xticks(sp.arange(-60, 80, 40))
ax.set_yticks(sp.arange(0, 1, 0.3))
ax.set_ylim(bottom=-0.3)

ax.text(77, -0.06, r'$\bar g s < g^{\star}$')
ax.text(77, -0.155, r'$\bar g s = g^{\star}$')
ax.text(77, -0.25, r'$\bar g s > g^{\star}$')

ax.text(fp_stable[0]-12, fp_stable[1], r'$p_f$')

fig.savefig(f'{FIGDIR}{LABEL}.pdf')
fig.savefig(f'{FIGDIR}{LABEL}.pgf')


#%% Fig: depression-traces
LABEL = 'depression-traces'

sol = ode.run(XPPFILE, transient=3100, total=5700, g=.9)
t0 = sol.t.iloc[0]
sol.t -= t0

fig, axs = plt.subplots(nrows=2, tight_layout=True, sharex=True,
                        figsize=FIGSIZE)

# voltage traces
ax = axs[0]
ax.plot(sol.t, sol.v1, label=r'$v_1$')
ax.plot(sol.t, sol.v2, label=r'$v_2$')
ax.set_yticks(sp.arange(-60, 40, 30))
ax.set_ylabel(r'$v_1,v_2$')
ax.legend(loc=1)

# synapse traces
ax = axs[1]
ax.plot(sol.t, sol.d1, c='C0', label=r'$d_1$')
ax.plot(sol.t, sol.s1, c='grey', label=r'$s_1$')
ax.set_yticks(sp.arange(0, 0.8, 0.2))
ax.set_xlabel('time (ms)')
ax.set_ylabel(r'$d_1,s_1$')
ax.legend(loc=1)

for idx, ax in enumerate(axs):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(string.ascii_uppercase[idx], loc='left')

fig.savefig(f'{FIGDIR}{LABEL}.pdf')
fig.savefig(f'{FIGDIR}{LABEL}.pgf')

#%% Fig: burst-sols
LABEL = 'burst-sols'

SOLFILE = path.join(DATDIR, 'soldat.pkl')
gs = [.5, .7, .8, .9, 1.2]
if not path.isfile(SOLFILE):
    print('COMPUTING SOLUTIONS')
    TRANSIENT = 3000
    TOTAL = 6300
    sols = []
    for g in gs:
        sol = ode.run(XPPFILE, transient=TRANSIENT, total=TOTAL, g=g)
        # Make time start from zero.
        sol.t = sol.t - sol.t.iloc[0]
        sols.append(sol)
    pickle.dump(sols, open(SOLFILE, 'wb'))
else:
    sols = pickle.load(open(SOLFILE, 'rb'))

sol_types = [r'$1-1$',
             r'$2-2$',
             r'$3-3$',
             r'$4-4$',
             'suppressed']

## Plot
fig, axs = plt.subplots(nrows=len(gs), tight_layout=True,
                        figsize=(FIGWIDTH, 6), sharex=True)

for idx, (ax, sol, sol_type, g) in enumerate(zip(axs, sols, sol_types, gs)):
    ax.plot(sol.t, sol.v1)
    ax.plot(sol.t, sol.v2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(sp.arange(0, 3300, 1000))
    ax.set_ylabel(r'$v_1,v_2$')
    ax.autoscale(axis='x', tight=True)
    ax.set_title(sol_type+' solution at ' + r'$\bar g='+str(g)+'$ '+
                 r'$\si{mS/cm^{2}}$', fontsize=10, loc='center')
    ax.set_title(string.ascii_uppercase[idx], loc='left')
    ax.tick_params(axis='both', which='major')

axs[-1].set_xlabel('time (ms)')

fig.savefig(f'{FIGDIR}{LABEL}.pdf')
fig.savefig(f'{FIGDIR}{LABEL}.pgf')


#%% Fig: Release condition
LABEL = 'release-condition'
from scipy.optimize import fsolve

_g = 0.8
# Get 3-3 LC
sol = ode.run(XPPFILE, transient=3100, total=5700, g=_g)
lc = ode.find_lc(sol)
gs = lc.s1*_g # total syn conductance
vs = sp.linspace(-65, 50, 100)

minf = lambda v: 0.5*(1+sp.tanh((v-p['va'])/p['vb']))
winf = lambda v: 0.5*(1+sp.tanh((v-p['vc'])/p['vd']))
vinf = lambda v, g: (-p['gca']*minf(v)*(v-p['vca'])-p['gl']*
                (v-p['vl'])+p['iapp']-g*(v-p['vinb']))/(p['gk']*(v-p['vk']))

# Fixed point values
vfs = []
wfs = []
for g in gs:
    vf = fsolve(lambda v: vinf(v, g)-winf(v), -65)
    vfs.append(vf[0])
    wfs.append(winf(vf))


fig, axs = plt.subplots(nrows=3, tight_layout=True, sharex=True,
                       figsize=(FIGWIDTH, FIGWIDTH*1.7))
axs[0].plot(lc.t, gs, label=r'$\bar g s_1$')
axs[0].plot(lc.t, pars.gstar*sp.ones(len(lc.t)), c='k', linestyle='--',
            label=r'$g^\star$')
axs[0].text(800, 0.06, r'$g^\star$')
axs[1].plot(lc.t, lc.v2, label=r'$v_2$')
axs[1].plot(lc.t, vfs, label=r'$v_f$', c='k')
axs[2].plot(lc.t, lc.w2, label=r'$w_2$')
axs[2].plot(lc.t, wfs, label=r'$w_f$', c='k')

for idx, ax in enumerate(axs):
    ax.set_title(string.ascii_uppercase[idx], loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([])
    ax.legend()


axs[0].set_ylabel(r'$\bar gs_1$')
axs[1].set_ylabel(r'$v_2$')
axs[2].set_ylabel(r'$w_2$')
axs[2].set_xlabel('time (ms)')

fig.savefig(f'{FIGDIR}{LABEL}.pdf')


#%% Fig: release delay
LABEL = 'release-delay'
datfile = os.path.join(DATDIR, 'bif-diagram.dat')
NSTEPS = 100
TOTAL = 10000

## Compute T diagram.

if not path.isfile(datfile):
    print('COMPUTING BIFDIAGRAM')
    n0s = [1, 2, 3, 4, 5]
    p0s = [.4, .7, .8, .9, .96]
    parsteps = [0.01, 0.01, 0.01, 0.005, 0.001]
    curves = []
    for n0, p0, step in zip(n0s, p0s, parsteps):
        left = ode.cont(XPPFILE, 'g', g=p0, parstep=-step, nsteps=NSTEPS,
                        total=TOTAL)
        right = ode.cont(XPPFILE, 'g', g=p0, parstep=step, nsteps=NSTEPS,
                         total=TOTAL)
        curve = left.iloc[::-1].append(right).reset_index(drop=True)
        curves.append(curve)
    dat = pd.concat(curves)
    pickle.dump(dat, open(datfile, 'wb'))
else:
    dat = pickle.load(open(datfile, 'rb'))


def get_delta_epsilon(sol):
    ''' Compute the release time, that is when gs<g*. '''
    ids1, ids2 = ode.spikes(sol)
    tks2 = sol.t.iloc[ids2]
    _sol = sol.iloc[ids1[0]+1:]
    release_time = _sol[_sol['gtot1']<pars.gstar].iloc[0]['t']
    delta_epsilon = tks2.iloc[0] - release_time
    return delta_epsilon


delta_epsilons = []
for row in dat.itertuples():
    sol = getattr(row, 'sol')
    delta_epsilons.append(get_delta_epsilon(sol))
dat['delta_epsilon'] = delta_epsilons


sol1 = dat.iloc[(abs(dat['g']-0.3)).argmin()]['sol']
dat2 = dat[dat['n']==2]
sol2 = dat2.iloc[(abs(dat2['g']-0.51)).argmin()]['sol']
sol3 = dat.iloc[(abs(dat['g']-0.8)).argmin()]['sol']

fig, ax = plt.subplots(tight_layout=True)
ax.plot(dat['g'].iloc[1:], dat['delta_epsilon'].iloc[1:], '.')
ax.set_xlim(left=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel(r'$\bar g$ ' + r'$(\si{mS/cm^{2}})$')
ax.set_ylabel('Release delay (ms)')

# Add solutions
inset1 = ax.inset_axes([0.28, 65, 0.3, 40], transform=ax.transData)
inset1.plot(sol1['t'], sol1['gtot1'])
inset1.plot(sol1['t'], sol1['gtot2'])
inset1.plot(sol1['t'], sp.ones(len(sol1['t']))*pars.gstar, '--',
            linewidth=1)
inset1.text(300, 0.05, r'$g^\star$')
inset2 = ax.inset_axes([0.65, 175, 0.3, 40], transform=ax.transData)
inset2.plot(sol2['t'], sol2['gtot1'])
inset2.plot(sol2['t'], sol2['gtot2'])
inset2.plot(sol2['t'], sp.ones(len(sol2['t']))*pars.gstar, '--',
            linewidth=1)
inset2.text(800, 0.06, r'$g^\star$')
inset3 = ax.inset_axes([0.7, 15, 0.3, 40], transform=ax.transData)
inset3.plot(sol3['t'], sol3['gtot1'])
inset3.plot(sol3['t'], sol3['gtot2'])
inset3.plot(sol3['t'], sp.ones(len(sol3['t']))*pars.gstar, '--',
            linewidth=1)
inset3.text(1300, 0.08, r'$g^\star$')
for idx, insetax in enumerate([inset1, inset2, inset3]):
    insetax.spines['top'].set_visible(False)
    insetax.spines['right'].set_visible(False)
    insetax.set_ylabel(r'$\bar g s$')
    insetax.yaxis.set_ticks([])
    insetax.set_title(string.ascii_uppercase[idx], loc='left')
fig.savefig(f'{FIGDIR}{LABEL}.pdf')


#%% Fig: free-quiet
LABEL = 'free-quiet'

# Compute 3-3 solution.
TRANSIENT = 3000
TOTAL = 12000
G = 0.8
N = 2
sol = ode.run(xppfile, transient=3000, total=10000, g=G)
lc = ode.find_lc(sol, norm=True)

period = lc.t.iloc[-1]
idxs, _ = ode.spikes(lc)
# Append last point to first to make it look nice.
idxs = sp.concatenate((idxs, [-1]))

fig, axs = plt.subplots(nrows=3, tight_layout=True, sharex=True,
                        figsize=(FIGWIDTH, FIGWIDTH*1.7))

# A: Plot voltage trace.
ax = axs[0]
ax.plot(lc.t, lc.v1)
ax.plot(lc.t, lc.v2)
ax.set_ylim((-80, 140))
ax.set_ylabel(r'$v_{1,2}$')

# Add arrows between ds.
OFFSET = 16
v0 = max(lc.v1)
for idx0, idx1 in zip(idxs, idxs[1:]):
    t0 = lc.t.iloc[idx0]
    t1 = lc.t.iloc[idx1]
    midpoint = t0 + (t1-t0)/2.
    p = patches.FancyArrowPatch((t0, v0), (t1, v0),
                                connectionstyle='arc3, rad=-0.3',
                                arrowstyle='->', mutation_scale=8)
    ax.add_patch(p)

    # Put text on top of patch.
    if idx1 == -1:
        text_y = v0+3.0*OFFSET
        text = r'$(n-1)T + 2\Delta t$'
    else:
        text_y = v0+OFFSET
        text = r'$T$'

    ax.text(midpoint, text_y, text,
            horizontalalignment='center')

# Add delta-t patches.
delt = period/2. - pars.period*N
delt_t0s = [lc.t.iloc[idxs[-2]], lc.t.iloc[idxs[-1]] - delt]
min_v = min(lc['v1'])
max_v = max(lc['v1'])
height = max_v - min_v
for delt_t0 in delt_t0s:
    rect = patches.Rectangle((delt_t0, min_v), delt, 2*height,
                             alpha=0.2, color='grey',
                             transform=ax.transData)
    ax.add_patch(rect)
    ax.text(delt_t0 + delt/2, 1.1*height, r'$\Delta t$',
            horizontalalignment='center', transform=ax.transData)

# B: Plot gtot.
ax = axs[1]
ax.plot(lc.t, lc.gtot1, c='C0')
ax.plot(lc.t, pars.gstar*sp.ones(len(lc.t)), c='k', linestyle='--')
ax.set_ylabel(r'$\bar g s_1$')
ax.set_xticks([0, 2*pars.period, lc.t.iloc[-1]])
ax.autoscale(axis='x', tight=True)
ax.text(840, 0.07, r'$g^{\star}$')

# Add first delta-t patch here as well.
delt = period/2. - pars.period*N
delt_t0 = lc.t.iloc[idxs[-2]]
rect = patches.Rectangle((2*pars.period, 0), delt, 1.,
                         alpha=0.2, color='grey',
                         transform=ax.transData)

ax.add_patch(rect)
ax.text((delt_t0 + delt/2)/period, 0.9, r'$\Delta t$',
        horizontalalignment='center', transform=ax.transAxes)


# C: Plot d.
ax = axs[2]
ax.plot(lc.t, lc.d1, c='C0', lw=2)
ax.set_xlabel('time')
ax.set_ylabel(r'$d_1$')
ax.set_xticks([0, 2*pars.period, lc.t.iloc[-1]])
ax.set_xticklabels([r'$0$', r'$(n-1)T$', r'$P_n=2\Delta t + 2(n-1)T$'])
ax.autoscale(axis='x', tight=True)

# Free-quiet patches
rect_free = patches.Rectangle((0, 0), 2*pars.period, 1.,
                              alpha=0.1, color='C0',
                              transform=ax.transData)
rect_quiet = patches.Rectangle((2*pars.period, 0), period-2*pars.period, 1.,
                               alpha=0.1, color='C1',
                               transform=ax.transData)
ax.add_patch(rect_free)
ax.add_patch(rect_quiet)
text_y = 0.85
ax.text(0.2, text_y, r'free: $\bar gs>g^{\star}$',
        horizontalalignment='center', transform=ax.transAxes)
ax.text(0.7, text_y, 'quiet', horizontalalignment='center',
        transform=ax.transAxes)

# Figure titles
for idx, ax in enumerate(axs):
    ax.set_title(string.ascii_uppercase[idx], loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([])


fig.savefig(f'{FIGDIR}{LABEL}.pdf')
fig.savefig(f'{FIGDIR}{LABEL}.pgf')


#%% Fig: delta-t
LABEL = 'delta-t'
datfile = os.path.join(DATDIR, 'bif-diagram' + '.dat')
NSTEPS = 100
TOTAL = 10000

if not path.isfile(datfile):
    print('COMPUTING BIFDIAGRAM')
    n0s = [1, 2, 3, 4, 5]
    p0s = [.4, .7, .8, .9, .96]
    parsteps = [0.01, 0.01, 0.01, 0.005, 0.001]
    curves = []
    for n0, p0, step in zip(n0s, p0s, parsteps):
        left = ode.cont(XPPFILE, 'g', g=p0, parstep=-step, nsteps=NSTEPS,
                        total=TOTAL)
        right = ode.cont(XPPFILE, 'g', g=p0, parstep=step, nsteps=NSTEPS,
                         total=TOTAL)
        curve = left.iloc[::-1].append(right).reset_index(drop=True)
        curves.append(curve)
    dat = pd.concat(curves)
    pickle.dump(dat, open(datfile, 'wb'))
else:
    dat = pickle.load(open(datfile, 'rb'))

# Add delt approximation.
delts = []
for row in dat.itertuples():
    period = getattr(row, 'period')
    n = getattr(row, 'n')
    delt = 0.5*period-(n-1)*pars.period
    delts.append(delt)
dat['delt'] = delts

fig, ax = plt.subplots(figsize=FIGSIZE, tight_layout=True)

# Place text on top of branches.
ax.text(0.725, pars.period+1, r'$T$', horizontalalignment='center',
        verticalalignment='bottom')
YOFFSET = 7
nmax = max(dat['n'])
for idx, n in enumerate(range(nmax+1)[1:]):
    ndat = dat[dat['n']==n]
    min_dat = ndat.loc[ndat['g'].argmin()]
    ax.text(min_dat['g'], min_dat['delt']-YOFFSET,
            r'$%s$' % n,
            horizontalalignment='center',
            verticalalignment='bottom')
    ax.plot(ndat['g'], ndat['delt'], c='C0')


# Add vertical line indicating T.
ax.axhline(y=pars.period, c='grey', linestyle='--')

ax.set_yticks(sp.arange(80, 230, 20))
ax.set_ylim(bottom=100, top=220)
ax.set_xlabel(r'$\bar g$ ' + r'$(\si{mS/cm^{2}})$')
ax.set_ylabel(r'$\Delta t$ ' + r'$(\si{ms})$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.savefig(f'{FIGDIR}{LABEL}.pdf')
fig.savefig(f'{FIGDIR}{LABEL}.pgf')

#%% delta-t-slope
from sklearn.linear_model import LinearRegression
regs = []
for n in dat['n'].unique():
    nbranch = dat[dat['n']==n]
    reg = LinearRegression().fit(nbranch['g'].to_numpy().reshape(-1, 1), nbranch['delt'].to_numpy())
    regs.append(reg)

# for 2
nbranch = dat[dat['n']==2]
reg = LinearRegression().fit(nbranch['g'].to_numpy().reshape(-1, 1), nbranch['delt'].to_numpy())
fig, ax = plt.subplots()
ax.plot(nbranch['g'], nbranch['delt'])
ax.plot(nbranch['g'], reg.predict(nbranch['g'].to_numpy().reshape(-1, 1)))
fig.savefig('tmp.pdf')



#%% Fig: psi-map
LABEL = 'psi-map'
ds = sp.linspace(0, 1, 1000)

fig, ax = plt.subplots(figsize=FIGSIZE, tight_layout=True)
ax.plot(ds, [model.d_sol(pars, d) for d in ds])
ax.plot(ds, ds, 'k--')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel(r'$d^\star$')
ax.set_ylabel(r'$\psi(d^\star)$')
ax.autoscale(axis='both', tight=True)

# Add fixed point.
ax.axvline(pars.d_sup, c='grey', ls='--')
ax.text(pars.d_sup+0.01, 0.1, r'$d_s$')

fig.savefig(f'{FIGDIR}{LABEL}.pdf')
fig.savefig(f'{FIGDIR}{LABEL}.pgf')

#%% Fig: delta-map
LABEL = 'delta-map'

ns = sp.arange(1, 5, 1)
ds = sp.linspace(0., 1, 1000)

fig, ax = plt.subplots(figsize=FIGSIZE, tight_layout=True)

for n in ns:
    # Plot curves.
    ys = [model.delta(pars, d, n) for d in ds]
    ax.plot(ds, ys, label=n, c='C0')

    # Add n numbers.
    ax.text(1.01, ys[-1], r'$%s$' % n,
            verticalalignment='center',
            horizontalalignment='left')

# Add fixed point.
ax.axvline(pars.d_sup, c='grey', ls='--')
ax.text(pars.d_sup+0.01, 0.1, r'$d_s$')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel(r'$d^{\star}$')
ax.set_ylabel(r'$\delta_n(d^{\star})$')
ax.autoscale(axis='both', tight=True)

fig.savefig(f'{FIGDIR}{LABEL}.pdf')
fig.savefig(f'{FIGDIR}{LABEL}.pgf')

#%% Fig: Fn-map
LABEL = 'FQ-map'

ds = sp.linspace(-1, 1, 10000)
ns = sp.arange(1, 5, 1)

fig, axs = plt.subplots(figsize=FIG2by1, tight_layout=True, ncols=2)

ax = axs[0]
for n in ns:
    ys = sp.array([model.Fn_map(pars, d, n)  for d in ds])
    # Filter out complex values of F.
    _ys = ys[sp.isreal(ys)].real
    _ds = ds[sp.isreal(ys)]
    ax.plot(_ds, _ys, c='C0')

    # Add n numbers.
    ax.text(1.01, _ys[-1], r'$%s$' % n,
            verticalalignment='center',
            horizontalalignment='left')

    # Add asymptotes where possible.
    d_a = model.d_a(pars, n)
    ax.axvline(d_a, c='k', ls=':')
    if n < 3 :
        ax.text(d_a-0.31, -30, r'$d_a(%s)$'%n)

ax.axvline(pars.d_sup, c='grey', ls='--')
ax.text(pars.d_sup+0.04, -30, r'$d_s$')

ax.set_xlabel(r'$d^\star$')
ax.set_ylabel(r'$\Delta t=F_n(d^\star)$')
ax.autoscale(axis='x', tight=True)
ax.set_xlim(left=-1.0)
ax.set_ylim(bottom=-50, top=300)

ax = axs[1]
delts = sp.linspace(-200, 500, 10000)

for n in ns:
    ys = sp.array([model.Qn_map(pars, delt, n) for delt in delts])
    ax.plot(delts, ys, c='C0')


# intersect = sp.log(pars.g/(pars.Lambda*pars.gstar))*pars.taus
# ax.axvline(intersect, c='grey', ls='--')
# ax.text(intersect+10, 0,
#         r'$\tau_s\ln{\left(\frac{\bar g}{\lambda g^{\star}}\right)}$')

# # Add n numbers.
ypos = [-0.25, -0.03, 0.17, 0.33]
for n, y in enumerate(ypos):
    ax.text(-100, y, r'$%s$' % (n+1),
            verticalalignment='center',
            horizontalalignment='left')

ax.set_xlabel(r'$\Delta t$')
ax.set_ylabel(r'$d^\star=Q_n(\Delta t)$')
ax.autoscale(axis='x', tight=True)
ax.set_ylim(bottom=-0.5, top=1)
ax.set_xlim(right=380)

for idx, ax in enumerate(axs):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(string.ascii_uppercase[idx], loc='left')

fig.savefig(f'{FIGDIR}{LABEL}.pdf')
fig.savefig(f'{FIGDIR}{LABEL}.pgf')

#%% Fig: Qn-map
LABEL = 'Qn-map'

delts = sp.linspace(-200, 500, 10000)
ns = sp.arange(1, 5, 1)

fig, ax = plt.subplots(figsize=FIGSIZE, tight_layout=True)

for n in ns:
    ys = sp.array([model.Qn_map(pars, delt, n) for delt in delts])
    ax.plot(delts, ys, c='C0')

    # Add n numbers.
    ax.text(delts[-1]+10, ys[-1], r'$%s$' % n,
            verticalalignment='center',
            horizontalalignment='left')

intersect = sp.log(pars.g/(pars.Lambda*pars.gstar))*pars.taus
ax.axvline(intersect, c='grey', ls='--')
ax.text(intersect+10, 0,
        r'$\tau_s\ln{\left(\frac{\bar g}{\lambda g^{\star}}\right)}$')

ax.set_xlabel(r'$\Delta t$')
ax.set_ylabel(r'$d^\star=Q_n(\Delta t)$')
ax.autoscale(axis='x', tight=True)
ax.set_ylim(bottom=-1.0)
ax.set_xlim(right=500)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.locator_params(axis='x', tight=True, nbins=4)
ax.locator_params(axis='y', tight=True, nbins=4)

fig.savefig(f'{FIGDIR}{LABEL}.pdf')
fig.savefig(f'{FIGDIR}{LABEL}.pgf')

#%% Fig: Pn-map
LABEL = 'Pn-map'

ds = sp.linspace(-1, 1, 10000)
ns = sp.arange(1, 5, 1)

fig, axs = plt.subplots(ncols=2, tight_layout=True,
                        figsize=FIG2by1)

for idx, ax in enumerate(axs):
    # Diagonal
    ax.plot(ds, ds, c='k', ls='--')
    ax.set_xlabel(r'$d^\star$')
    ax.set_xlim((-1.0, 1.0))
    ax.set_ylim((-1.0, 1.0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.locator_params(axis='x', tight=True, nbins=4)
    ax.locator_params(axis='y', tight=True, nbins=4)
    ax.set_title(string.ascii_uppercase[idx], loc='left')


# A: Pi with all curves.
ax = axs[0]
ax.set_ylabel(r'$\Pi_n(d^\star)$')

for n in ns:
    ys = sp.array([model.Pn_map(pars, d, n) for d in ds])
    realidx = sp.isreal(ys)
    ax.plot(ds[realidx], ys[realidx], c='C0')


# Add first 2 asymptotes.
for n in ns[:2]:
    d_a = model.d_a(pars, n)
    ax.axvline(d_a, c='k', ls=':')
    ax.text(d_a-0.34, 0, r'$d_a(%s)$'%n)

# Add n numbers.
n_locs = [(0.03, -0.7),
          (-0.45, -0.7),
          (-0.8, 0.42),
          (-0.8, 0.74)]
for loc, n in zip(n_locs, ns):
    ax.text(loc[0], loc[1], r'$%s$' % n,
            verticalalignment='center',
            horizontalalignment='left')

# B: Pi with n=2 and varying g.
gs = [0.01, 0.0335, 0.3]
ax = axs[1]
ax.set_ylabel(r'$\Pi_2(d^\star)$')
for g in gs:
    ys = sp.array([model.Pn_map(pars._replace(g=g), d, 2) for d in ds])
    realidx = sp.isreal(ys)
    ax.plot(ds[realidx], ys[realidx],
            label=r'$\bar g\approx%s$' % round(g, 2))
ax.legend()

fig.savefig(f'{FIGDIR}{LABEL}.pdf')
fig.savefig(f'{FIGDIR}{LABEL}.pgf')

#%% Fig: folds
LABEL = 'folds'

ns_A = sp.arange(1, 6, 1)
ns_B = sp.arange(1, 6, 1)
gs = sp.linspace(0, 1.1, 20)
ds = sp.linspace(-1.75, 1, 10000)
gmax = gs[-1]

# A: Compute numeric bifurcation curves of d
curves = []
for n in ns_A:
    curve = model.Pn_map_bif(pars, n, gs)
    curves.append(curve)


# B: Compute numeric bifurcation curves of P
dfs = []
for n in ns_B:
    _gs = [model.fp2g(pars, d, n) for d in ds]
    periods = [model.Pn_map_period(pars._replace(g=g), d, n)
               for (d, g) in zip(ds, _gs)]
    _df = pd.DataFrame({'g': _gs, 'd': ds, 'n': n, 'P': periods})
    dfs.append(_df)
period_df = pd.concat(dfs)

fig, axs= plt.subplots(figsize=FIG2by1, tight_layout=True, ncols=2)

ax = axs[0]
# Plot analytic fixed points.
for n in ns_A:
    db, gb = model.critical_fp(pars, n)
    stable_ds = ds[ds>db]
    unstable_ds = ds[ds<db]
    _gs_stable = [model.fp2g(pars, d, n) for d in stable_ds]
    _gs_unstable = [model.fp2g(pars, d, n) for d in unstable_ds]
    ax.plot(_gs_stable, stable_ds, c='C0', label='stable')
    ax.plot(_gs_unstable, unstable_ds, c='C0', label='unstable',
            linestyle=':')

ax.set_xlim(-0.05, gmax)
ax.set_ylim(-1.8, 1.2)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel(r'$\bar g$ ' + r'$(\si{mS/cm^{2}})$')
ax.set_ylabel(r'$d^{\star}_{f}$')
ax.text(gmax+0.01, 0.0, r'$1$')
ax.text(gmax+0.01, -0.55, r'$2$')
ax.text(gmax+0.01, -1.7, r'$3$')
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.])
ax.locator_params(axis='x', tight=False, nbins=6)

patch_stable = Line2D([], [], linestyle='-', linewidth=1, label='stable')
patch_unstable = Line2D([], [], linestyle=':', linewidth=1, label='unstable')
ax.legend(handles=[patch_stable, patch_unstable], loc=(0.5, 0.9))

ax = axs[1]
for idx, n in enumerate(ns_B):
    db, gb = model.critical_fp(pars, n)
    _nbranch = period_df[period_df['n']==n]
    nbranch = _nbranch[_nbranch['d']>db]
    # nbranches.append({'g': nbranch['g'], 'period': nbranch['P']})
    ax.plot(nbranch.g, nbranch.P, c='C0')
    texty = nbranch[nbranch['g']<gmax].iloc[-1]['P']
    ax.text(gmax+0.01, texty, r'$%s$' % n,
            verticalalignment='center')
    # Plot period from ODE integration
    if n in dat['n'].unique():
        odebranch = dat[dat['n']==n]
        ax.plot(odebranch['g'], odebranch['period'], c='C1')


ax.set_xlim(-0.05, gmax)
ax.set_ylim(200, 2700)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel(r'$\bar g$ ' + r'$(\si{mS/cm^{2}})$')
ax.set_ylabel('period (\si{ms})')
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.])
for idx, ax in enumerate(axs):
    ax.set_title(string.ascii_uppercase[idx], loc='left')


patch_ana = Line2D([], [], color='C0', linewidth=1, label='analytic')
patch_num = Line2D([], [], color='C1', linewidth=1, label='numeric')
ax.legend(handles=[patch_ana, patch_num], loc=(0.5, 0.9))

fig.savefig(f'{FIGDIR}{LABEL}.pdf')
fig.savefig(f'{FIGDIR}{LABEL}.png')
fig.savefig(f'{FIGDIR}{LABEL}.pgf')

#%% Fig: final-bif
LABEL = 'final-bif'

ns = sp.arange(1, 10)

def gtoP(pars, g, n):
    df = model.phi_stable(pars._replace(g=g), g, n)
    return model.Pn_map_period(pars._replace(g=g), df, n)

branches_dat = 'final-bif.dat'
if not path.isfile(branches_dat):
    branches = []
    # Plot numeric fixed points.
    for idx, n in enumerate(ns):
        # Compute borders
        if n>1:
            L = model.left_border(pars, n)
        else:
            L = 0
        R = model.right_border(pars, n)
        # Compute period
        db, gb = model.critical_fp(pars, n)
        gs = sp.linspace(gb+0.0001, pars.g_sup, 1000)
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

fig, ax = plt.subplots(tight_layout=True)
for branch in branches:
        ax.plot(branch['g'], branch['period'], c='C0', lw=2.5)

# Plot period from ODE integration
for n in dat['n'].unique():
    odebranch = dat[dat['n']==n]
    ax.plot(odebranch['g'], odebranch['period'], c='C1', lw=1.3)

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

fig.savefig(f'{FIGDIR}{LABEL}.pdf')
fig.savefig(f'{FIGDIR}{LABEL}.pgf')

#region phin?
#%% Fig: phin(n)
LABEL = 'phin'

ns = sp.arange(1, 15)
gs = [0.5, 0.6, 0.8, 1.0]
ds_curves = []
for g in gs:
    ds = []
    for n in ns:
        fp_stable = model.Pn_map_fps(pars._replace(g=g), n)[-1]
        ds.append(fp_stable)
    ds_curves.append(ds)

fig, ax = plt.subplots(tight_layout=True, figsize=FIGSIZE)

for idx, g in enumerate(gs):
    ax.plot(ds_curves[idx], label=r'$\bar g={}$'.format(str(g)))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel(r'$n$')
ax.set_ylabel(r'$d_f^\star=\phi_n(\bar g)')
ax.legend()

fig.savefig(f'{FIGDIR}{LABEL}.pdf')
fig.savefig(f'{FIGDIR}{LABEL}.png')
fig.savefig(f'{FIGDIR}{LABEL}.pgf')
#endregion

#region LR
#%% Fig: LR
LABEL = 'LR'

# gs = sp.linspace(0.001, pars.g_sup, 100)
ns = sp.arange(2, 10)

Ls = [model.left_border(pars, n) for n in ns]
Rs = [model.right_border(pars, n) for n in ns]

fig, ax = plt.subplots()
ax.plot(ns, Ls, '.', label=r'$L_n$')
ax.plot(ns, Rs, '.', label=r'$R_n$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel(r'$n$')
ax.set_ylabel(r'$\bar g$')
ax.legend()
fig.savefig(f'{FIGDIR}{LABEL}.pdf')
fig.savefig(f'{FIGDIR}{LABEL}.pgf')

#endregion

#region T-plots???
#%% T-plots
LABEL = 'T-plots'
ns = [2, 3, 4, 5]
from scipy import interpolate
Ts = sp.geomspace(1, 280, 1000)
_g = 0.6


# def R(pars, n):
#     fp = model.phi(pars, n, x0=0.99)
#     if fp:
#         return model.Fn_map(pars, fp, n) - pars.period
#     else:
#         return None


# def L(pars, n):
#     fp = model.phi(pars, n, x0=0.99)
#     if fp:
#         gtot = pars.g*model.d_map_sol(pars, fp, n-1)*sp.exp(-pars.period/pars.taus)
#         return gtot - pars.gstar
#     else:
#         return None

# def rborder(pars, n, x0):
#     f = lambda T: R(pars._replace(period=T), n)
#     fp0 = sp.optimize.newton(f, x0,)
#     return fp0


# def lborder(pars, n, x0):
#     f = lambda T: L(pars._replace(period=T), n)
#     fp0 = sp.optimize.newton(f, x0)
#     return fp0


curves = []
for n in ns:
    curve = model.Pn_map_bif_T(pars._replace(g=_g), n, Ts)
    curves.append(curve)


# Right root
Rs = []
Ls = []
for n in ns:
    try:
        _R = model.rborder(pars._replace(g=_g), n, 150)
        _L = model.lborder(pars._replace(g=_g), n, 150)
    except:
        _R = None
        _L = None
    Rs.append(_R)
    Ls.append(_L)

# Left root
Rs2 = []
Ls2 = []
init = 20
for n in ns:
    try:
        _R = model.rborder(pars._replace(g=_g), n, 20)
        _L = model.lborder(pars._replace(g=_g), n, 20)
    except:
        _R = None
        _L = None
    Rs2.append(_R)
    Ls2.append(_L)
print(Rs2)

fig, axs= plt.subplots(figsize=FIG2by1, tight_layout=True, ncols=2)

# Plot dstar folds
ax = axs[0]
for idx, n in enumerate(ns):
    curve = curves[idx]
    stable = curve[curve['stable']]
    # Hacky but have to do it since unstable fp hard to find there
    if idx==0:
        unstable = curve[~curve['stable']]
    else:
        unstable = curve[(~curve['stable']) & (curve['dstar']<0.1)]
    _cc = stable.iloc[::-1].append(unstable)
    cc = _cc[(sp.isreal(_cc['dstar'])) & (_cc['dstar'].notnull())]
    ax.plot(cc['T'], cc['dstar'])

ax.set_ylim(bottom=-0.7, top=1)
ax.set_ylabel(r'$d^{\star}_{f}$')

# Plot stable period
ax = axs[1]

for idx, n in enumerate(ns):
    curve = curves[idx]
    stable = curve[curve['stable']]
    x = stable[(stable['T']<Ls[idx]) & (stable['T']>Rs[idx])]
    ax.plot(x['T'], x['period'], label=r'$'+str(n)+'$-$'+str(n)+'$')

ax.set_ylabel('period ' + r'$(\si{ms})$')
ax.legend()

for idx, ax in enumerate(axs):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_xaxis()
    ax.set_xlim(left=215)
    ax.set_xlabel(r'$T$ ' + r'$(\si{ms})$')
    ax.set_title(string.ascii_uppercase[idx], loc='left')

fig.savefig(f'{FIGDIR}{LABEL}.pdf')
fig.savefig(f'{FIGDIR}{LABEL}.pgf')

#endregion
