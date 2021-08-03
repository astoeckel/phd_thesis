import h5py

colors = ['#000000', mpl.cm.get_cmap('viridis')(0.3), mpl.cm.get_cmap('viridis')(0.6), mpl.cm.get_cmap('viridis')(0.9), utils.oranges[1]]

styles = {
    "additive": {
        "label": "Sum of 1D bases",
        "color": "k",
        "linewidth": 2.0,
        "marker": 'o',
        "markersize": 4,
        "markevery": (0, 5),
        "markeredgewidth": 0.5,
        "zorder": 9,
    },
    "additive_2d": {
        "label": "Sum of 2D bases",
        "color": colors[4],
        "linewidth": 2.0,
        "marker": 'o',
        "markersize": 4,
        "markevery": (0, 5),
        "markeredgewidth": 0.5,
        "zorder": 9,
    },
    "multiplicative": {
        "label": "Product of 1D bases",
        "color": colors[1],
        "linewidth": 1.5,
        "marker": 's',
        "markersize": 4,
        "markevery": (0, 6),
        "markeredgewidth": 0.5,
    },
}

def plot_file(ax, ax1, fn, d):
    with h5py.File(utils.datafile(fn), "r") as f:
        sigmas = f["SIGMAS"][()]
        n_repeat = f["REPEAT"][()]
        Es_add = f["Es_add"][()]
        Es_mul = f["Es_mul"][()]
        Es_a2d = f["Es_mlp"][()]

    ax.set_title(f"Basis function order $o = {d}$")
    linewidth = 1.0

    for key, Es in [("additive", Es_add), ("multiplicative", Es_mul), ("additive_2d", Es_a2d)]:
        style = dict(**styles[key])
        if "linewidth" in style:
            style["linewidth"] *= linewidth
        style_1 = dict(style)
        style_2 = dict(style)
        style_2["label"] = None
        style_2["zorder"] = 10

        E25 = np.percentile(Es, 25, axis=-1)
        E50 = np.percentile(Es, 50, axis=-1)
        E75 = np.percentile(Es, 75, axis=-1)


        ax.fill_between(1.0/sigmas, E25, E75, color=style["color"], alpha=0.5, linewidth=0.0)
        ax.loglog(1.0/sigmas, E50, markeredgecolor="k", linestyle='-', **style_1)
        ax.loglog(1.0/sigmas, E50, markeredgecolor="k", linestyle='', **style_2)

        ax1.loglog(1.0/sigmas, E50, linestyle='--', color=style["color"])

#    y_cen = -0.15
    ax.set_xlabel("Filter coefficient $\\rho^{-1}$")#, labelpad=15.0)
#    ax.add_patch(
#        mpl.patches.Polygon([(0.0, y_cen), (0.03, y_cen - 0.02),
#                             (0.03, y_cen + 0.02)],
#                            facecolor='k',
#                            transform=ax.transAxes,
#                            clip_on=False))
#    ax.add_patch(
#        mpl.patches.Polygon([(1.0, y_cen), (0.97, y_cen - 0.02),
#                             (0.97, y_cen + 0.02)],
#                            facecolor='k',
#                            transform=ax.transAxes,
#                            clip_on=False))
#    ax.text(0.05,
#            y_cen - 0.005,
#            'Low freqs.',
#            va='center',
#            ha='left',
#            transform=ax.transAxes)
#    ax.text(0.95,
#            y_cen - 0.005,
#            'Higher freqs.',
#            va='center',
#            ha='right',
#            transform=ax.transAxes)

    ax.axvline(2.0, linestyle=':', linewidth=0.5, color='k')

    ax.set_ylabel("Median NRMSE")
    ax.set_ylim(3e-3, 1.1)


def plot_stats(ax, fn):
    E1, E2, E3 = [], [], []
    with h5py.File(utils.datafile(fn), "r") as f:
        ds = f["DS"][()]
        Es_add = f["Es_add"][()]
        Es_mul = f["Es_mul"][()]
        Es_a2d = f["Es_mlp"][()]

    for key, Es in [("additive", Es_add), ("multiplicative", Es_mul), ("additive_2d", Es_a2d)]:
        style = dict(**styles[key])
        del style["markevery"]

        E25 = np.percentile(Es[0], 25, axis=-1)
        E50 = np.mean(Es[0], axis=-1) #np.percentile(Es[0], 50, axis=-1)
        E75 = np.percentile(Es[0], 75, axis=-1)

        ax.plot(ds, E50, markeredgecolor="k", linestyle='-', **style)
        ax.fill_between(ds, E25, E75, color=style["color"], alpha=0.5, linewidth=0.0)

    ax.set_xticks(ds)
    ax.set_ylabel("Median RMSE")
    ax.set_ylim(3e-3, 1.1)
    ax.set_yscale('log')

    ax.set_title("Filter coefficient $\\rho^{-1} = 2$")
    ax.set_xlabel("Basis function order $o$")

fig, axs = plt.subplots(1, 3, figsize=(7.45, 2.25), gridspec_kw={
    "wspace": 0.2,
})

plot_file(axs[0], axs[1], utils.datafile("dendritic_computation_fourier_example_d5.h5"), 5)
plot_file(axs[1], axs[0], utils.datafile("dendritic_computation_fourier_example_d9.h5"), 9)
plot_stats(axs[2], utils.datafile("dendritic_computation_fourier_example_rho050.h5"))

axs[1].legend([
    mpl.lines.Line2D([], [], **styles["additive"], markeredgecolor='k'),
    mpl.lines.Line2D([], [], **styles["multiplicative"], markeredgecolor='k'),
    mpl.lines.Line2D([], [], **styles["additive_2d"], markeredgecolor='k'),
], ["Additive $f_\\mathrm{add}$ ($2d$ DOF)", "Multiplicative $f_\\mathrm{den}$ ($4d$ DOF)", "Additive with 2D basis $f_\\mathrm{mlp}$ ($d^2$ DOF)"], ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.35))

for i in range(3):
    axs[i].text(-0.25 if i == 0 else -0.15, 1.0275, "\\textbf{{{}}}".format(chr(ord('A') + i)), va="bottom", ha="left", size=12, transform=axs[i].transAxes)
    if i != 0:
        #axs[i].set_yticklabels([])
        axs[i].set_ylabel(None)

utils.save(fig)
