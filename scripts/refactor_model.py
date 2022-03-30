"""Script to test refactoring of maps in model.py"""
import matplotlib.pyplot as plt
import numpy as np
from poincare_map.model import PoincareMap
import pickle

plt.ioff()

xppfile = "odes/mlml.ode"
model = PoincareMap.from_ode("odes/mlml.ode")

#%%

ds = np.linspace(-1, 1, 100)

n = 3
fig, ax = plt.subplots()
Ps = [model.P_map(d, n) for d in ds]
ax.plot(ds, Ps, label="comp")
Ps = [model.P_map_direct(d, n) for d in ds]
ax.plot(ds, Ps, ".", label="direct")
ax.legend()
fig.savefig("tmp/pimap.pdf")


#%% Vary g
gs = [0.1, 0.3, 0.6]

fig, ax = plt.subplots()
for g in gs:
    Ps = [model._replace(g=g).P_map(d, n) for d in ds]
    ax.plot(ds, Ps, label=f"g={str(g)}")
ax.legend()
fig.savefig("tmp/pimap_g.pdf")

#%% inverse

g = 0.55
n = 4
f = lambda d: model._replace(g=g).G(d, n)
dmin, gmin = model.critical_fp(n)
kk = inversefunc(f, g, domain=[dmin, 1])

#%% Left border
from scipy import optimize

f = lambda g: model.L_fun(g, n=3)
fp0 = optimize.newton(f, x0=1.0)

#%% Test period

# Load numericly computed diagram
df = pickle.load(open("data/bif-diagram.pkl", "rb"))
# ns = df.n.unique()
ns = [2, 3, 4, 5]

fig, ax = plt.subplots()
for idx, n in enumerate(ns):
    dfn = df[(df.n == n) & (df.g > 0.1)]
    ax.plot(dfn.g, dfn.period, c=f"C{idx}")
    # Compute analytic period
    # Ps = [model.Period(g, n) for g in dfn.g]
    # ax.plot(dfn.g, Ps, c='C1', linewidth=1)

gg = 0.006788
# Plot left/right borders
for idx, n in enumerate(ns):
    # NOTE: We're cheating here since T=376.3
    left_border = model._replace(gstar=gg).left_branch_border(n)
    right_border = model._replace(gstar=gg).right_branch_border(n)
    ax.axvline(left_border, color=f"C{idx}")
    ax.axvline(right_border, color=f"C{idx}")

fig.savefig("tmp/bif.pdf")

#%% How well does our fixed point d* match?

# Load numericly computed diagram
df = pickle.load(open("data/bif-diagram.pkl", "rb"))
# ns = df.n.unique()
ns = [2, 3, 4, 5]

fig, ax = plt.subplots()
for idx, n in enumerate(ns):
    dfn = df[(df.n == n) & (df.g > 0.1)]
    ax.plot(dfn.g, dfn.dmax, c=f"C{idx}")
    # Compute analytic period
    dmax_ana = [model.phi_stable(g, n) for g in dfn.g]
    # Ps = [model.Period(g, n) for g in dfn.g]
    # ax.plot(dfn.g, dmax_ana, c='C1', marker='.', linewidth=1)

# # Plot left/right borders
# for n in ns:
#     # NOTE: We're cheating here since T=376.3
#     left_border = model._replace(T=377).left_border(n)
#     right_border = model._replace(T=377).right_border(n)
#     ax.axvline(left_border, color=f'C{n}')
#     # ax.axvline(right_border, color=f'C{n}')

fig.savefig("tmp/bif_dmax.pdf")

#%% Check what the solution on the left border looks like
df4 = df[df.n == 4]
row = df4.iloc[df4.g.argmin()]
sol = row.sol

fig, axs = plt.subplots(nrows=2, tight_layout=True, sharex=True)
axs[0].plot(sol.t, sol.v1)
axs[0].set(ylabel="v1")
axs[1].plot(sol.t, sol.s1 * row.g)
axs[1].plot(sol.t, model.gstar * np.ones(len(sol.t)), "k--")
axs[1].set(ylim=(0, 0.02), xlabel="time", ylabel="gtot")
fig.savefig("tmp/lsol.pdf")

#%% Check whether numeric value of d* produces better L/R match

# Load numericly computed diagram
df = pickle.load(open("data/bif-diagram.pkl", "rb"))
# ns = df.n.unique()
ns = [2, 3, 4, 5]

fig, ax = plt.subplots()
for idx, n in enumerate(ns):
    dfn = df[(df.n == n) & (df.g > 0.1)]
    ax.plot(dfn.g, dfn.period, c=f"C{idx}")
    # Compute analytic period
    dmax_ana = [model.phi_stable(g, n) for g in dfn.g]


# Plot left/right borders
for idx, n in enumerate(ns):
    dfn = df[(df.n == n) & (df.g > 0.1)]
    # NOTE: We're cheating here since T=376.3
    left_border = model.left_branch_border(n)
    right_border = model.right_branch_border(n)
    ax.axvline(left_border, color=f"C{idx}", linestyle="--")
    ax.axvline(right_border, color=f"C{idx}", linestyle="--")

    dmax_left = dfn.dmax.min()
    dmax_right = dfn.dmax.max()
    left_border_num = model.left_border2(n, dmax_left)
    right_border_num = model.right_border2(n, dmax_right)
    ax.axvline(left_border_num, color=f"C{idx}")
    ax.axvline(right_border_num, color=f"C{idx}")

ax.set(xlabel=r"$\bar g$", ylabel="P")


fig.savefig("tmp/bif.pdf")

#%% How well does our fixed point d* match?

# Load numericly computed diagram
df = pickle.load(open("data/bif-diagram.pkl", "rb"))
# ns = df.n.unique()
ns = [2, 3, 4, 5]

fig, ax = plt.subplots()
for n in ns:
    dfn = df[(df.n == n) & (df.g > 0.1)]
    ax.plot(dfn.g, dfn.dmax, c="C0")
    # Compute analytic period
    dmax_ana = [model.phi_stable(g, n) for g in dfn.g]
    # Ps = [model.Period(g, n) for g in dfn.g]
    ax.plot(dfn.g, dmax_ana, c="C1", marker=".", linewidth=1)

# # Plot left/right borders
# for n in ns:
#     # NOTE: We're cheating here since T=376.3
#     left_border = model._replace(T=377).left_border(n)
#     right_border = model._replace(T=377).right_border(n)
#     ax.axvline(left_border, color=f'C{n}')
#     # ax.axvline(right_border, color=f'C{n}')

fig.savefig("tmp/bif_dmax.pdf")

#%% Check what the solution on the left border looks like
df4 = df[df.n == 4]
row = df4.iloc[df4.g.argmin()]
sol = row.sol

fig, axs = plt.subplots(nrows=2, tight_layout=True, sharex=True)
axs[0].plot(sol.t, sol.v1)
axs[0].set(ylabel="v1")
axs[1].plot(sol.t, sol.s1 * row.g)
axs[1].plot(sol.t, model.gstar * np.ones(len(sol.t)), "k--")
axs[1].set(ylim=(0, 0.02), xlabel="time", ylabel="gtot")
fig.savefig("tmp/lsol.pdf")

#%% Check whether numeric value of d* produces better L/R match

# Load numericly computed diagram
df = pickle.load(open("data/bif-diagram.pkl", "rb"))
# ns = df.n.unique()
ns = [2, 3, 4, 5]

fig, ax = plt.subplots()
for idx, n in enumerate(ns):
    dfn = df[(df.n == n) & (df.g > 0.1)]
    ax.plot(dfn.g, dfn.period, c=f"C{idx}")
    # Compute analytic period
    dmax_ana = [model.phi_stable(g, n) for g in dfn.g]
    # Ps = [model.Period(g, n) for g in dfn.g]
    ax.plot(dfn.g, dmax_ana, c="C1", marker=".", linewidth=1)


# Plot left/right borders
for idx, n in enumerate(ns):
    dfn = df[(df.n == n) & (df.g > 0.1)]
    # NOTE: We're cheating here since T=376.3
    # left_border = model.left_border(n)
    # right_border = model.right_border(n)
    # ax.axvline(left_border, color=f'C{idx}')
    # ax.axvline(right_border, color=f'C{idx}')

    dmax_left = dfn.dmax.min()
    dmax_right = dfn.dmax.max()
    left_border_num = model.left_border2(n, dmax_left)
    right_border_num = model.right_border2(n, dmax_right)
    ax.axvline(left_border_num, color=f"C{idx}")
    ax.axvline(right_border_num, color=f"C{idx}")


#%% Try to check individual solutions
from poincare_map import ode

row = df.iloc[50]
n = row.n
g = row.g
model = model._replace(gstar=0.00684, g=g)

sol = row.sol
ids1, ids2 = ode.spikes(sol)
tks1 = sol.t.iloc[ids1].values
tks2 = sol.t.iloc[ids2].values
delta_t = tks2[0] - tks1[-1] - model.Tact
2 * (delta_t + model.Tact + (n - 1) * model.T)
row.period


delta_t_ana = model._replace(g=g).F_map(sol.d1.iloc[0], 3)
delta_t
delta_t_ana

# Is delta_n correct? -> Yes!
d0 = sol.d1.iloc[0]
model.delta(d0, 3)
sol.d1.iloc[ids1[-1]]

dlast = sol.iloc[ids1[-1]].d1

id0 = sol.d1.argmin()
t0 = sol.t.iloc[id0]
d0 = sol.d1.iloc[id0]

new_delta_t = model.tauk * np.log(g / model.gstar * d0)
new_delta_t

#%%

fig, axs = plt.subplots(nrows=3)
axs[0].plot(sol.t, sol.v1)
axs[1].plot(sol.t, sol.d1)
axs[1].plot([t0], [d0], "x")
axs[2].plot(sol.t, sol.s1 * g)
axs[2].axvline(tks2[0], color="k")
# axs[2].set(ylim=(0, 0.1))
fig.savefig("tmp/sol.pdf")


#%% Estimate numerical g*
# Find value of \bar g * s when cell 2 spikes first
# Do that for n>=2


def gstar(df_row):
    """Find gstar at index of first spike of cell 2 from LC solution."""
    sol = df_row.sol
    _, ids2 = ode.spikes(sol)
    return df_row.g * sol.iloc[ids2[0]].s1


gstars = df.apply(gstar, axis=1)
df["gstar"] = gstars

fig, ax = plt.subplots(tight_layout=True)
for n in df.n.unique():
    dfn = df[df.n == n]
    # ax.plot(dfn.g, dfn.period, c='C0')
    # Compute analytic period
    ax.plot(dfn.g, dfn.gstar, ".", label=f"${n}:{n}$")
    ax.set(xlabel=r"$\bar g$", ylabel=r"$g^\star$")
# ax.set(ylim=(0.0066, 0.00705), xlim=(0.3, 0.6))
ax.legend(loc=1)
fig.savefig("tmp/gstar-big-diag.pdf")


# NOTE: FANCY PLOT HERE!!!!
# def gstar(df_row):
#     """Find gstar at index of first spike of cell 2 from LC solution."""
#     sol = df_row.sol
#     _, ids2 = ode.spikes(sol)
#     return df_row.g * sol.iloc[ids2[0]].s1



# df = pickle.load(open("data/bif-diagram.pkl", "rb"))
# df = df[df['n']>1]
# gstars = df.apply(gstar, axis=1)
# df["gstar"] = gstars

# # fig, ax = plt.subplots(tight_layout=True)
# fig = plt.figure(figsize=(8, 6), tight_layout=True)
# grid_spec = plt.GridSpec(7, 5)
# ax1 = fig.add_subplot(grid_spec[:, 0:-2])
# ax2 = fig.add_subplot(grid_spec[:, -2:], sharey=ax1)

# for n in df.n.unique():
#     dfn = df[df.n == n]
#     # ax.plot(dfn.g, dfn.period, c='C0')
#     # Compute analytic period
#     ax1.plot(dfn.g, dfn.gstar, ".", label=f"${n}:{n}$")
#     ax1.set(xlabel=r"$\bar g$", ylabel=r"$g^\star$")
# # ax.set(ylim=(0.0066, 0.00705), xlim=(0.35, 0.6))
# ax1.legend(loc=2)

# ax2.hist(df['gstar'], density=True, bins=5, orientation='horizontal')
# # ax2.set(yticks=[])
# fig.savefig("tmp/gstar-big-diag.pdf")


#%% Compute bounds with better gstar
# 3-3 branch
N = 5
branch3 = df[df['n'] == N]
gstar_left = branch3['gstar'].iloc[0]
gstar_right = branch3['gstar'].iloc[-1]

# lb = model._replace(gstar=gstar_right).left_border(N)
lb = model._replace(gstar=gstar_left).left_branch_border(N)
rb = model._replace(gstar=gstar_right).right_branch_border(N)


lb_orig = model.left_branch_border(N)
rb_orig = model.right_branch_border(N)

fig, ax = plt.subplots(tight_layout=True)
ax.plot(branch3['g'], branch3['gstar'], '.')
ax.set(xlabel=r'$\bar g$', ylabel=r'$g^\star$', ylim=(0.006, 0.007))
ax.axvline(lb, linestyle='--', color='C0')
ax.axvline(rb, linestyle='--', color='C0')
ax.axvline(lb_orig, linestyle='--', color='C1')
ax.axvline(rb_orig, linestyle='--', color='C1')
fig.savefig("tmp/gstar-n-n.pdf")


#%% Try finding 5-5 branch
p0 = 0.56
step = 0.001
nsteps = 100
total = 15000
left = ode.cont(xppfile, g=p0, parstep=-step, nsteps=nsteps, total=total)
right = ode.cont(xppfile, g=p0, parstep=step, nsteps=nsteps, total=total)

#%% Compute dmin

df = df.assign(dmin=df["sol"].apply(lambda x: x["d1"].min()))

ns = [2, 3, 4, 5]

fig, axs = plt.subplots(nrows=2, sharex=True, tight_layout=True, figsize=(4, 6))
for idx, n in enumerate(ns):
    dfn = df[df.n == n]
    axs[0].plot(dfn.g, dfn.dmax, c=f"C{idx}")
    axs[1].plot(dfn.g, dfn.dmin, c=f"C{idx}")

axs[0].set(ylabel=r"$d_{max}$")
axs[1].set(xlabel=r"$\bar g$", ylabel=r"$d_{min}$")
fig.savefig("tmp/bif-dmindmax.pdf")


#%% Find right border fixed point analytically
def rfp(model, n, g) -> float:
    _model = model._replace(g=g)
    K = (
        _model.gstar
        / (_model.g * _model.Lambda)
        * np.exp(_model.Tinact / _model.tauk)
    )
    return (K - _model.beta(n)) / _model.alpha(n)


# # Load numericly computed diagram
# df = pickle.load(open("data/bif-diagram.pkl", 'rb'))
# # ns = df.n.unique()
# ns = [2, 3, 4, 5]

# fig, ax = plt.subplots()
# for idx, n in enumerate(ns):
#     dfn = df[(df.n == n) & (df.g > 0.1)]
#     ax.plot(dfn.g, dfn.period, c=f'C{idx}')
#     right_dmax = rfp(model, n, dfn['g'].max())
#     ax.axvline(dfn['g'].max(), '--')


# fig.savefig('tmp/bif.pdf')

dfn = df[df.n == 3]
right_dmax_3 = dfn["dmax"].iloc[-1]
right_dmax_3_ana = rfp(model, 3, dfn["g"].max())
right_dmax_3_ana = rfp(model, 3, model.right_branch_border(3))

right_dmax_3
right_dmax_3_ana

model.right_branch_border(3)


def g_right(model, n):
    Qn = model.Q_map(model.Tinact, n)
    A = 1 / model.delta


#%% Plot all (v*,w*)
import intersect
from poincare_map import ode


def get_release_state(sol, time_offset=10):
    """Find value of (v,w) when release cond. satisfied."""
    # Take time offset since in the first steps rel cond might be satisfied.
    release_idx = np.where((sol.gtot1 < model.gstar) & (sol.t > time_offset))[
        0
    ][0]
    return sol.iloc[release_idx - 1]


row = df.iloc[100]
sol = row["sol"]
sol_star = get_release_state(sol)

fig, axs = plt.subplots(nrows=2, tight_layout=True)
axs[0].plot(sol.t, sol.gtot1, linewidth=0.1)
axs[0].plot(sol.t, np.ones(len(sol)) * model.gstar, linewidth=0.1)
axs[0].plot(sol_star["t"], sol_star["gtot1"], "x")
axs[1].plot(sol.t, sol.v2)
axs[1].plot(sol_star["t"], sol_star["v1"], "x")
fig.savefig("tmp/sol.pdf")


#%%

df_not1 = df[df['n'] > 1]

# Compute all (v*,w*)
release_states = df_not1.apply(
    lambda row: get_release_state(row["sol"], model.gstar), axis=1
)


vnull, wnull = ode.nullclines(xppfile, gtot=model.gstar)
v_fp, w_fp = intersect.intersection(
    vnull[:, 0], vnull[:, 1], wnull[:, 0], wnull[:, 1]
)


fig, ax = plt.subplots()
ax.plot(vnull[:, 0], vnull[:, 1])
ax.plot(wnull[:, 0], wnull[:, 1])
ax.plot(v_fp, w_fp, "x")
ax.plot(release_states['v1'], release_states['w1'], '.', markersize=0.8)
ax.set(xlabel=r"$v$", ylabel=r"$w$")
fig.savefig("tmp/nullclines.pdf")
