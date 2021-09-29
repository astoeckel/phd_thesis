import h5py
import json

def plot_result(datafile, plot_legend=True):
    with h5py.File(utils.datafile(datafile),
                   "r") as f:
        filters = json.loads(f.attrs["filters"])[:4]
        networks = json.loads(f.attrs["networks"])
        target = json.loads(f.attrs["targets"])
        freqs = json.loads(f.attrs["freqs"])
        errs = f["errs"][()]

    fig, axs = plt.subplots(1, 2, figsize=(7.4, 1.8))


    def plot_single(ax, Es, color, linestyle='-', linewidth=1.0):
        ax.plot(freqs,
                np.median(Es, axis=0),
                linestyle=linestyle,
                linewidth=linewidth,
                color=color)
        ax.fill_between(freqs,
                        np.percentile(Es, 25, axis=0),
                        np.percentile(Es, 75, axis=0),
                        color=color,
                        alpha=0.33,
                        linewidth=0.0)


    cmap1 = mpl.cm.get_cmap("Blues")
    cmap2 = mpl.cm.get_cmap("Oranges")
    colors1 = [
        cmap1(0.4 + 0.6 * i / (len(filters) - 1)) for i in range(len(filters))
    ]
    colors2 = [
        cmap2(0.4 + 0.6 * i / (len(filters) - 1)) for i in range(len(filters))
    ]


    def plot_experiment(ax, Es):
        for i, n_flts in enumerate(filters):
            plot_single(ax, Es[1, i], color=colors2[i], linestyle=':')

        for i, n_flts in enumerate(filters):
            plot_single(ax, Es[2, i], color=colors1[i], linewidth=1.75)

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlim(freqs[0], freqs[-1])
        ax.set_ylim(1e-3, 1e1)
        ax.set_xlabel("Input bandlimit $B$ (Hz)")
        ax.set_ylabel("NRMSE")

        plot_single(ax, Es[0, 0], color='k', linestyle='--')


    plot_experiment(axs[0], errs[0])
    plot_experiment(axs[1], errs[1])

    for i in range(2):
        axs[i].set_title([
            "\\textbf{Integrator}",
            "\\textbf{Short first-order lowpass} ($\\tau = 10\,\\mathrm{ms}$)"
        ][i])
        axs[i].text(-0.165,
                    1.055,
                    "\\textbf{{{}}}".format(chr(ord('A') + i)),
                    size=12,
                    va="baseline",
                    ha="left",
                    transform=axs[i].transAxes)


    if plot_legend:
        fig.legend([
            mpl.lines.Line2D(
                [], [], linewidth=1.75, color=colors1[i])
            for i in range(len(filters))
        ], [
            "{} filter{}".format(n_flts, "s" if n_flts > 1 else "")
            for n_flts in filters
        ],
                   loc="upper center",
                   bbox_to_anchor=(0.6, 1.15),
                   ncol=len(filters))

        fig.legend([
            mpl.lines.Line2D([], [], linestyle=':', color=colors2[i])
            for i in range(len(filters))
        ], [
            "{} filter{}".format(n_flts, "s" if n_flts > 1 else "")
            for n_flts in filters
        ],
                   loc="upper center",
                   bbox_to_anchor=(0.6, 1.25),
                   ncol=len(filters))

        fig.text(0.325, 1.1465, "{Feed-forward network}", ha="right", va="baseline")
        fig.text(0.325, 1.0465, "{Recurrent network}", ha="right", va="baseline")

    utils.save(fig)

plot_result("heterogeneous_recurrence_exploration.h5")
