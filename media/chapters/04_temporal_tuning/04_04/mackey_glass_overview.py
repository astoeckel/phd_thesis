import lmu_utils

np.random.seed(3913)

fig, ax = plt.subplots(figsize=(7.8, 1.5))

ax.fill_between([1, 15], [-3, -3], [3, 3],
                color=utils.oranges[1],
                alpha=0.3,
                lw=0.0)
ax.fill_between([-116, 0], [-3, -3], [3, 3],
                color=utils.blues[0],
                alpha=0.1,
                lw=0.0)
ax.fill_between([-37, 0], [-3, -3], [3, 3],
                color=utils.blues[0],
                alpha=0.3,
                lw=0.0)
ax.axhline(0.0, linestyle=':', color='grey', lw=0.5)

for i in range(1):
    if i == 0:
        alpha = 1.0
    else:
        alpha = 0.25
    ys = lmu_utils.mackey_glass(10000, 30, 1.0)
    ys = ys[5000:5117+15]

    ax.plot(
        np.arange(-len(ys) + 16, 1),
        ys[:-15],
        '-o' if i == 0 else 'o',
        color=utils.blues[0],
        lw=0.7,
        markersize=4,
        markeredgewidth=0.0,
        alpha=alpha,
    )
    ax.plot(
        np.arange(1, 16),
        ys[-15:],
        '-+' if i == 0 else '+',
        color=utils.oranges[1],
        lw=0.7,
        markersize=4,
        alpha=alpha,
    )

ax.set_xlim(-117, 16)
ax.set_ylim(-3, 3)
ax.spines["left"].set_visible(False)
ax.set_yticks([])
ax.set_ylabel("Input sample $u_t$")
ax.set_xlabel("Input sample index $t$")

utils.save(fig)

