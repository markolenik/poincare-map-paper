import intersect
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

from poincare_map import helpers, ode, paths
from poincare_map.poincare_map import PoincareMap

plt.ioff()

rcp = matplotlib.rcParamsDefault
rcp["axes.spines.right"] = False
rcp["axes.spines.top"] = False
matplotlib.rcParams.update(rcp)

ml = ode.ML(paths.ml_file)
pmap = PoincareMap()


def calc_ISIs(sol: pd.DataFrame) -> np.ndarray:
    """Compute nth ISI."""
    spike_times = helpers.spike_times(sol["v"])
    ISIs = np.diff(spike_times)
    return ISIs


def calc_release_delay(sol: pd.DataFrame, gstar: float) -> float:
    """Compute the release time, that is when gs=g*."""
    spike_times = helpers.spike_times(sol["v"])
    if len(spike_times) > 0:
        # Get rid of transient
        _sol = sol.loc[sol.index > 5]
        # Release time as defined by the release condition
        release_condition_satisfied = _sol.loc[_sol["gtot"] < gstar].index[0]
        return release_condition_satisfied - spike_times[0]


gbif = 0.00379
# Make ICs and integrate
gstar = pmap.gstar
# TODO: Repeat with gbif
vfp = ml.fixed_point(gtot=gstar)
wfp = ml.winf(vfp)  # type: ignore

sols_tauk = []
sols_delay = []
tauks = np.linspace(10, 800, 50)
for tauk in tqdm.tqdm(tauks):
    # tauk
    gtot0 = gstar
    v0 = ml.pars["vtheta"] - 1  # Subtract 1 to have a root at t=0.
    w0 = wfp
    ics = np.array([v0, w0, gtot0])
    sol = ml.run(total=2000, ics=ics, tauk=tauk, dt=0.1)
    sols_tauk.append(sol)

    # delay
    gtot0 = 0.45
    v0 = ml.fixed_point(gtot=gtot0)
    w0 = ml.winf(v0)
    ics = np.array([v0, w0, gtot0])
    sol = ml.run(total=2000, ics=ics, tauk=tauk, dt=0.1)
    sols_delay.append(sol)

ISI_list = [calc_ISIs(sol) for sol in sols_tauk]
delay_list = [calc_release_delay(sol, gstar=gstar) for sol in sols_delay]
delay_list2 = [calc_release_delay(sol, gstar=gbif) for sol in sols_delay]

#%%

fig, axs = plt.subplots(nrows=2, sharex=True, tight_layout=True)
axs[0].plot(tauks, [ISIs[0] for ISIs in ISI_list], label=r"$ISI_1$")
axs[0].plot(tauks, [ISIs[1] for ISIs in ISI_list], label=r"$ISI_2$")
axs[0].plot(tauks, [ISIs[2] for ISIs in ISI_list], label=r"$ISI_3$")
axs[0].plot(tauks, np.ones(len(tauks)) * pmap.T, "k--")
axs[0].set(xlabel=r"$\tau_k$", ylabel=r"$ISI$")
axs[0].set(xlim=(0, 800), ylim=(350, 525))
axs[0].legend()
axs[1].plot(tauks, delay_list)
axs[1].plot(tauks, delay_list2)
fig.savefig("tmp/tauk-vs-ISI.pdf")


#%%
# sol = ml.run(total=500, ics=ics, tauk=100)
# sol2 = ml.run(total=500, ics=ics, tauk=300)
fig, ax = plt.subplots()
ax.plot(sol.index, sol.v)
ax.plot(sol2.index, sol2.v)
fig.savefig("tmp/sol.pdf")

ss = sols_delay[20]
fig, axs = plt.subplots(nrows=2)
axs[0].plot(ss.index, ss['v'])
axs[1].plot(ss.index, ss['gtot'])
axs[1].plot(ss.index, gbif*np.ones(len(ss)), linestyle="dashed", color='black')
axs[1].set(ylim=(0.003, 0.004))
fig.savefig('tmp/sol.pdf')

gbif=0.0038
release_time = ss.index[ss['gtot']<gbif][0]
fig, ax = plt.subplots()
ax.plot(ss['v'], ss['w'])
ax.plot(ss['v'][ss.index>release_time], ss['w'][ss.index>release_time], 'k')
ax.plot(np.linspace(-60, 45, 100), [np.linspace(-60, 45, 100)])
fig.savefig('tmp/n.pdf')

calc_release_delay(ss, gstar=gstar)

#%%
