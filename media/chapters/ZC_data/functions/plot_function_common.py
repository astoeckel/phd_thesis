from two_comp_parameters import BENCHMARK_FUNCTIONS

def do_plot(key):
    f = lambda x, y: 2.0 * BENCHMARK_FUNCTIONS[key](0.5 * (x + 1.0), 0.5 *
                                                    (y + 1.0)) - 1.0

    xs, ys = np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)
    xss, yss = np.meshgrid(xs, ys)
    zss = f(xss, yss)

    fig, ax = plt.subplots(figsize=(1.0, 1.0))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect(1)
    C = ax.contourf(xs, ys, zss, cmap='RdBu', vmin=-1, vmax=1, zorder=-1)
    ax.contour(xs,
               ys,
               zss,
               levels=C.levels,
               colors=['white'],
               linestyles=['--'],
               linewidths=[0.7], zorder=-1)

    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.tick_params(axis='both', which='both', left=True, right=True, top=True, bottom=True)

    ax.set_xticks(np.linspace(-1, 1, 3))
    ax.set_xticks(np.linspace(-1, 1, 5), minor=True)

    ax.set_yticks(np.linspace(-1, 1, 3))
    ax.set_yticks(np.linspace(-1, 1, 5), minor=True)

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    utils.save(fig)

    print(key, "rms =", np.sqrt(np.mean(np.square(zss))))

