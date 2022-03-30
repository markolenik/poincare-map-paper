"""Check if dQ/dd is positive."""

ds = np.linspace(0.0001, 300, 100)

fig, ax = plt.subplots()
# ax.plot(ds, [PoincareMap(3).Q_map_slope(d, 0.5) for d in ds])
# ax.plot(ds, [PoincareMap(3).Q_map_slope_num(d, 0.5) for d in ds])

ax.plot(ds, [PoincareMap(4).Q_map_slope(d, 0.5) for d in ds])
ax.plot(ds, [PoincareMap(4).Q_map_slope_num(d, 0.5) for d in ds])
fig.savefig('tmp/tmp.pdf')
