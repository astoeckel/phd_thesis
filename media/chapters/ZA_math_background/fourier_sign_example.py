fig, axs = plt.subplots(3, 3, figsize=(8.0, 3.0),
                        sharex=True, sharey=True,
                        gridspec_kw={
                            "wspace": 0.15,
                            "hspace": 0.75,
                        })
ns = np.asarray(np.round(np.logspace(0, 7, axs.size - 1, base=2)), dtype=np.int)
for i, ax in enumerate(axs.flat):
    N = 1001
    lw = 0.7
    if i + 1 < axs.size:
        ts = np.linspace(-np.pi, np.pi, N)[1:-1]
        ys = np.zeros(N - 2)
        for j in range(1, 2 * ns[i] + 1, 2):
            ys += 4.0 / np.pi * np.sin(ts * j) / j
        ax.plot(ts, ys, 'k', linewidth=lw)
        ax.set_title("$q = {}$".format(ns[i]))
    else:
        ax.plot([-np.pi, 0], [-1, -1], color='k', linewidth=lw)
        ax.plot([0, np.pi], [1, 1], color='k', linewidth=lw)
        ax.plot([0], [-1], 'o', color='white', markeredgecolor='k', markersize=4, markeredgewidth=lw)
        ax.plot([0], [1], 'o', color='white', markeredgecolor='k', markersize=4, markeredgewidth=lw)
        ax.plot([0], [0], 'o', color='k', markeredgecolor='k', markersize=4, markeredgewidth=lw)
        ax.set_title("$q \\to \\infty$")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    #ax.set_xticklabels(["$-\\pi$", "$-\\frac{\\pi}2$", "$0$", "$\\frac{\\pi}2$", "$\\pi$"])
    for spine in ["left", "bottom", "top", "right"]:
        #ax.spines[spine].set_position(('outward', 5))
        ax.spines[spine].set_visible(False)
    #ax.set_xlabel("$x$")

utils.save(fig)

