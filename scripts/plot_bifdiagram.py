import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyxpp.pyxpp as xpp
from poincare_map import model

rcp = matplotlib.rcParamsDefault
matplotlib.rcParams.update(rcp)
plt.ioff()
matplotlib.use('pgf')


EQSIZE = 15                     # Font size of equations.
CENTER_TITLE_SIZE = 11
FIGWIDTH = 3.85
FIGSIZE = (FIGWIDTH, FIGWIDTH)
FIG2by1 = (FIGWIDTH*1.4, (FIGWIDTH*1.4)/2)

rcp['figure.max_open_warning'] = 0
rcp['axes.unicode_minus'] = False
rcp['text.usetex'] = True
rcp['pgf.rcfonts'] = False
rcp['pgf.texsystem'] = 'lualatex'
rcp['xtick.labelsize'] = 'small'
rcp['ytick.labelsize'] = 'small'
rcp['axes.labelsize'] = 'x-large'
rcp['axes.titlesize'] = 'xx-large'
rcp['figure.titlesize'] = 'large'
rcp['font.size'] = 9
# Use latex preamble.
rcp['pgf.preamble'] = r'\usepackage{amsmath} \usepackage{siunitx}'

matplotlib.rcParams.update(rcp)

XPPFILE = 'mlml.ode'
p = xpp.read_pars(XPPFILE)


pars = model.Pars(p['g'], p['gstar'], p['lambda'])

from poincare_map import ode
# Find initial solution.
sol = ode.run(XPPFILE, g=0., total=5000)
lc = ode.find_lc(sol, norm=True)

# lc = ode.find_lc(sol, norm=False)
lc = ode.find_lc(sol, norm=True)
fig, axs = plt.subplots(nrows=2)
axs[0].plot(lc["t"], lc["v1"])
axs[1].plot(lc["t"], lc["d1"])
fig.savefig("tmp.pdf")


#*

# Load data
df = pickle.load(open('data/bif-diagram.pkl', 'rb'))


## Plot
fig, axs = plt.subplots(ncols=2, tight_layout=True,
                        figsize=FIG2by1)

# A: Period diagram
ax = axs[0]
nmax = max(df['n'])
for n in range(nmax+1)[1:]:
    ndat = df.loc[df['n']==n]
    ax.plot(ndat['g'], ndat['period'], c='C0')
    ax.set_ylabel('period ' + r'$(\si{ms})$')
    # ax.set_yticks(np.arange(500, 2500, 500))

    # place text next to branch
    right_idx = ndat['g'].argmax()
    right_g = ndat.loc[right_idx]['g']
    left_g = ndat['g'].min()
    mid_g = left_g + (right_g - left_g)/2.
    right_P = ndat.loc[right_idx]['period']
    # if n > 1:
    #     ax.text(mid_g-.02, right_P+20, r'$%s$' % n)
    # else:
    #     ax.text(mid_g+0.1, right_P+15, r'$%s$' % n)

# # Add suppressed solution.
# gsup_range = np.linspace(pars.g_sup, 1.1, 100)
# ax.plot(gsup_range, np.ones(len(gsup_range))*pars.period, c='C0')
# ax.autoscale(axis='x', tight=True)
# ax.text(pars.g_sup+0.03, pars.period+40, 'suppressed')
# # ax.text(pars.g_sup+0.02, 1000, r'$\bar g_s$')

# # Add dotted lines with higher-order solutions.
# ax.axvline(right_g, c='grey', ls='--')
# ax.axvline(pars.g_sup, c='grey', ls='--')

# Period T note
# ax.annotate(r'$T$', (1.1, pars.period), (1.1+0.05, pars.period-100))


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.savefig('bif.pdf')
