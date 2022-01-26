from poincare_map import ode
from poincare_map import helpers
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.figure import Figure
import intersect
import tqdm
from typing import Tuple
import pandas as pd

plt.ioff()

rcp = matplotlib.rcParamsDefault
rcp["axes.spines.right"] = False
rcp["axes.spines.top"] = False
matplotlib.rcParams.update(rcp)

GSTAR = 0.0036
T = 375

xppfile = "odes/ml.ode"
vnull, wnull = ode.nullclines(xppfile, gtot=GSTAR)
v_fp, w_fp = intersect.intersection(vnull[:, 0], vnull[:, 1], wnull[:, 0], wnull[:, 1])
v_fp = v_fp[0]
w_fp = w_fp[0]


def roots(x, y) -> np.ndarray:
    return intersect.intersection(x, y, x, np.zeros(len(x)))[0]


def calc(
    xppfile: str,
    v0: float = v_fp,
    w0: float = w_fp,
    gstar: float = GSTAR,
    g: float = 0.5,
    tauk: float = 100,
    total: float = 400,  # big enough even for large g?
    **kwargs,
) -> pd.DataFrame:
    """Compute solution given initial conditiosn"""
    s0 = gstar / g
    ics = np.array([v0, w0, s0])
    sol = ode.run(xppfile, ics=ics, g=g, tauk=tauk, total=total, **kwargs)
    return sol



def nth_ISI(sol: pd.DataFrame, n: int = 0) -> float:
    """Compute nth ISI."""
    x0s = roots(sol.t, sol.v)
    spikes = x0s[::2]
    ISIs = np.diff(spikes)
    return ISIs[n]


#* spikes

signs = np.sign(sol.v)
roots = np.where(signs + np.roll(signs, 1) == 0)[0]

id1 = helpers.spikes(sol.v)



fig, ax = plt.subplots()
ax.plot(sol.t, sol.v)
ax.plot(sol.t[id1], sol.v[id1], 'x')
ax.axhline(0, c='k')
fig.savefig('tmp.pdf')




# *


vnull, wnull = ode.nullclines(xppfile, gtot=GSTAR)
v_fp, w_fp = intersect.intersection(vnull[:, 0], vnull[:, 1], wnull[:, 0], wnull[:, 1])

fig, ax = plt.subplots()
ax.plot(vnull[:, 0], vnull[:, 1])
ax.plot(wnull[:, 0], wnull[:, 1])
ax.plot(v_fp, w_fp, "x")
fig.savefig("figures/nullclines.pdf")


# g = 0.5  # 3-3
# # Fixed points at gstar
# v_pf, w_pf = (-22.057, 0.030055)
# s_pf = gstar / g

# *


def bla(tauk=100, g=0.5):
    s0 = GSTAR / g
    ics = np.array([v_fp, w_fp, s0])
    sol = ode.run(xppfile, total=1000, ics=ics, tauk=tauk, g=g)
    spikes_times = roots(sol.t, sol.v)

    # Find the first time where v crosses 0 downwards

    fig, axs = plt.subplots(nrows=3, tight_layout=True)
    axs[0].plot(sol.t, sol.v)
    axs[1].plot(sol.t, sol.w)
    axs[2].plot(sol.t, sol.s)
    fig.savefig("figures/traces.pdf")
    return spikes_times[spikes_times > T][0]


# * Vary g, const tauk

gs = np.linspace(0.3, 0.55, 100)
# Initial I0s
stuff = [nth_ISI(xppfile, g=g, total=600, tauk=100) for g in tqdm.tqdm(gs)]

s0s = [GSTAR / g for g in gs]

fig, axs = plt.subplots(nrows=2, sharex=True, tight_layout=True)
axs[0].plot(gs, I0s, ".")
axs[0].set(ylabel=r"$ISI_0$", ylim=(400, 500))
axs[1].plot(gs, s0s)
axs[1].set(xlabel=r"$\bar g$", ylabel=r"$s(0)$")
fig.savefig("figures/ISI0s.pdf")


# * Vary both g and tauk

gs = np.linspace(0.3, 0.55, 100)
tauks = [10, 100, 1000]
# Initial I0s
I0s_list = []
sols = []
for tauk in tauks:
    I0s = []
    for g in tqdm.tqdm(gs):
        I0, sol = nth_ISI(xppfile, g=g, total=2000, tauk=tauk)
        I0s.append(I0)
    I0s_list.append(I0s)
    sols.append(sol)

fig, axs = plt.subplots(tight_layout=True, ncols=2, figsize=(10, 5))
for idx, I0s in enumerate(I0s_list):
    label = f"$\tau_k={tauks[idx]}$"
    axs[0].plot(gs, I0s, label=label)
    axs[0].set(xlabel=r"$\bar g$", ylabel=r"$ISI_0$")
    # Plot associated traces
    axs[1].plot(sols[idx].t, sols[idx].v, label=label)
    axs[1].set(xlim=(0, 600))
axs[0].legend()
axs[1].legend()
fig.savefig("figures/ISI0s.pdf")

# * Plot tauk vs ISI0

g = 0.5
tauks = np.linspace(10, 1000, 100)
ics = np.array([v_fp, w_fp, s0])

sols = []
t0s = []
I0s = []
ISI_lists = []
for tauk in tqdm.tqdm(tauks):
    s0 = GSTAR / g
    sol = ode.run(xppfile, ics=ics, g=g, tauk=tauk, total=1500, dt=0.1)
    ids = helpers.spikes(sol.v)
    spikes = sol.t[ids].to_numpy()
    sols.append(sol)
    ISI_lists.append(np.diff(spikes))
    t0s.append(spikes[0])


I0s = [ISIs[0] for ISIs in ISI_lists]
I1s = [ISIs[1] for ISIs in ISI_lists]
I2s = [ISIs[2] if len(ISIs)>2 else np.nan for ISIs in ISI_lists]

fig, ax = plt.subplots()
ax.plot(tauks, I0s, ".", label=r'$ISI_0$')
ax.plot(tauks, I1s, ".", label=r'$ISI_1$')
ax.plot(tauks, I2s, ".", label=r'$ISI_2$')
ax.set(xlabel=r"$\tau_k$", ylabel=r"$ISI$")
ax.legend()
fig.savefig("figures/ISI.pdf")



fig, ax = plt.subplots()
ax.plot(tauks, t0s, ".")
ax.set(xlabel=r"$\tau_k$", ylabel=r"$t_0$")
fig.savefig("figures/tk0.pdf")



#*

sol = ode.run(xppfile, ics=ics, g=g, tauk=100, total=1500, dt=0.5)
fig, ax = plt.subplots()
ax.plot(sol.t, sol.v)
fig.savefig("figures/trace.pdf")
