# %% Init stuff
import matplotlib.pyplot as plt
import numpy as np
from poincare_map.poincare_map import PoincareMap
from poincare_map.bif import compute_analytic

xppfile = "odes/mlml.ode"

#%% Plot G

fig, ax = plt.subplots()
for n in [1, 2, 3, 4, 5, 6]:
    poincare = PoincareMap(n=n)
    branch = poincare.stable_branch()
    if branch is not None:
        ax.plot(branch['g'], branch['period'])

fig.savefig('tmp/ana-bif.pdf')

#%% phi stable

gs = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
ax.plot(gs, [PoincareMap(n=1).phi_stable(g) for g in gs])
fig.savefig('tmp/phi_stable.pdf')


#%% Plot diagram
PoincareMap(n=2).phi_stable(0.01)
