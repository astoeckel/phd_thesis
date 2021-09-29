import h5py
import json

def plot_result(datafile, plot_legend=True):
    with h5py.File(utils.datafile(datafile),
                   "r") as f:
        filters = json.loads(f.attrs["filters"])[:4]
        xs_filters = json.loads(f.attrs["xs_filters"])
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


    cmap1 = mpl.cm.get_cmap("inferno")
    colors1 = [
        cmap1(0.2 + 0.6 * i / (len(xs_filters) - 1)) for i in range(len(xs_filters))
    ]

    def plot_experiment(ax, Es):
        for i, n_flts in enumerate(xs_filters):
            plot_single(ax, Es[0, 0, i], linewidth=1.75, color=colors1[i])

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlim(freqs[0], freqs[-1])
        ax.set_ylim(1e-3, 5e0)
        ax.set_xlabel("Input bandlimit $B$ (Hz)")
        ax.set_ylabel("NRMSE")


    print(errs.shape)

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
            for i in range(len(xs_filters))
        ], [
            "$\\vartheta = {:0.0f}\\,\\mathrm{{ms}}$".format(xs_tau * 1e3) if xs_tau < 1.0 else "$\\vartheta = {:.2g}\\,\\mathrm{{s}}$".format(xs_tau)
            for xs_tau in xs_filters
        ],
                   loc="upper center",
                   bbox_to_anchor=(0.488, 1.15),
                   ncol=len(xs_filters),
                   columnspacing=1.25,
                   handlelength=1.25,
                   )

    utils.save(fig)

plot_result("heterogeneous_recurrence_exploration_xs_flt.h5")
