"""Find gstar numerically."""


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
from poincare_map.model import PoincareMap

plt.ioff()

rcp = matplotlib.rcParamsDefault
rcp["axes.spines.right"] = False
rcp["axes.spines.top"] = False
matplotlib.rcParams.update(rcp)

VTHETA = 0
GSTAR = 0.0068
T = 376

xppfile = "odes/ml.ode"


def roots(x, y) -> np.ndarray:
    return intersect.intersection(x, y, x, np.zeros(len(x)))[0]


def calc(
    xppfile: str = "odes/ml.ode",
    g: float = 0.5,
    tauk: float = 100,
    total: float = 400,  # big enough even for large g?
    **kwargs,
) -> pd.DataFrame:
    """Compute solution given initial conditions."""
    vnull, wnull = ode.nullclines(xppfile, gtot=GSTAR)
    v_fp, w_fp = intersect.intersection(
        vnull[:, 0], vnull[:, 1], wnull[:, 0], wnull[:, 1]
    )
    v0 = v_fp[0]
    w0 = w_fp[0]
    s0 = 1
    ics = np.array([v0, w0, s0])
    sol = ode.run(xppfile, ics=ics, g=g, tauk=tauk, total=total, **kwargs)
    return sol


def calc_sol(
    xppfile: str = "odes/ml.ode",
    g0: float = 0.1,
    total: float = 800,  # big enough even for large g?
    **kwargs,
) -> pd.DataFrame:
    """Compute solution given initial conditions."""
    vnull, wnull = ode.nullclines(xppfile, gtot=g0)
    v_fp, w_fp = intersect.intersection(
        vnull[:, 0], vnull[:, 1], wnull[:, 0], wnull[:, 1]
    )
    v0 = v_fp[0]
    w0 = w_fp[0]
    ics = np.array([v0, w0, g0])
    sol = ode.run(xppfile, ics=ics, total=total, **kwargs)
    return sol


# TODO this could be done explictely since we know equation
def left_knee(
    g: float = 0.1,
    xppfile: str = "odes/ml.ode",
    **kwargs,
) -> pd.DataFrame:
    """Get left knee of v-nullcline."""
    vnull, wnull = ode.nullclines(xppfile, gtot=g)
    v_fp, w_fp = intersect.intersection(
        vnull[:, 0], vnull[:, 1], wnull[:, 0], wnull[:, 1]
    )
    # That's a hack
    idx = np.argmin(vnull[vnull[:, 0]<0, 1])
    return vnull[idx]

#%%
G = 0.1

vnull, wnull = ode.nullclines('odes/ml.ode', gtot=G)
v_fp, w_fp = intersect.intersection(
    vnull[:, 0], vnull[:, 1], wnull[:, 0], wnull[:, 1]
)
lknee = left_knee(g=G)

fig, ax = plt.subplots()
ax.plot(vnull[:, 0], vnull[:, 1])
ax.plot(wnull[:, 0], wnull[:, 1])
ax.plot([lknee[0]], [lknee[1]], 'x')

fig.savefig('tmp/nullclines_test.pdf')

#%% Initialise just before gsup
model = PoincareMap.from_ode("odes/mlml.ode")
g0 = (model.gsup)*model.dsup

sol = calc_sol(g0=g0)

# At what g does v cross 0? -> ~0.007
sol[sol['v'] > 0].iloc[0]['g']

# At what g does w fall below left knee?
w_lknee = [left_knee(g)[1] for g in sol['g']]

#%%
fig, axs = plt.subplots(nrows=3, tight_layout=True, sharex=True,
                        figsize=(6, 10))
axs[0].plot(sol['t'], sol['v'])
axs[0].set(ylabel='v')
axs[1].plot(sol['t'], sol['w'])
axs[1].plot(sol['t'], w_lknee)
axs[1].set(ylabel='w')
axs[2].plot(sol['t'], sol['g'])
axs[2].set(xlabel='time', ylabel='g')
fig.savefig('tmp/sol.pdf')


#%%

g = 0.4
sol = calc(g=g, total=600)

release_row = sol[sol["v"] > 0].iloc[0]
gstar = release_row["s"] * g
release_time = release_row["t"]

fig, axs = plt.subplots(
    nrows=3, tight_layout=True, sharex=True, figsize=(7, 10)
)
axs[0].plot(sol["t"], sol["v"])
axs[0].plot(sol["t"], np.zeros(len(sol)))
axs[0].set(ylabel="v")
axs[1].plot(sol["t"], sol["w"])
axs[1].set(ylabel="w")
axs[2].plot(sol["t"], sol["s"] * g)
axs[2].axvline(release_time, color="k")
axs[2].axhline(gstar, color="k")
axs[2].set(ylabel=r"$g_{tot}$", xlabel="time", ylim=(0, 0.01))
fig.savefig("tmp/find_gstar.pdf")

gstar

#%% Plot 4-4 all silent variables with fixed points also
import pickle
import tqdm

df = pickle.load(open("data/bif-diagram.pkl", "rb"))
df4 = df[df["n"] == 4]

row5 = df4.iloc[5]
sol = row5['sol']

# Find fp values for respective gtot
from os import path
if not path.isfile('data/vfps.pkl'):
    vfps = []
    wfps = []
    for gtot in tqdm.tqdm(sol['gtot1']):
        vnull, wnull = ode.nullclines(xppfile, gtot=gtot)
        v_fp, w_fp = intersect.intersection(
            vnull[:, 0], vnull[:, 1], wnull[:, 0], wnull[:, 1]
        )
        vfps.append(v_fp)
        wfps.append(w_fp)

    vfps = [v[0] if v else None for v in vfps]
    wfps = [w[0] if w else None for w in wfps]
    with open('data/vfps.pkl', 'wb') as f:
        pickle.dump(vfps, f)
    with open('data/wfps.pkl', 'wb') as f:
        pickle.dump(wfps, f)
else:
    with open('data/vfps.pkl', 'rb') as f:
        vfps = pickle.load(f)
    with open('data/wfps.pkl', 'rb') as f:
        wfps = pickle.load(f)


#%%

fig, axs = plt.subplots(nrows=3, sharex=True, tight_layout=True)
axs[0].plot(sol['t'], sol['v2'])
axs[0].plot(sol['t'], vfps)
axs[0].set(ylabel='v')
axs[1].plot(sol['t'], sol['w2'])
axs[1].plot(sol['t'], wfps)
axs[1].set(ylabel='w')
axs[2].plot(sol['t'], sol['gtot1'])
axs[2].set(xlabel='time', ylabel='gtot')
fig.savefig('tmp/w-path.pdf')
