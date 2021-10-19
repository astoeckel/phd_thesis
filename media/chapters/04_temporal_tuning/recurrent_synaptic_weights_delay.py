import scipy.stats

mode_colors = {
    "mod_fourier_bartlett": mpl.cm.get_cmap("viridis")(0.1),
    "mod_fourier_erasure": mpl.cm.get_cmap("viridis")(0.4),
    "legendre_erasure": mpl.cm.get_cmap("viridis")(0.95),
}

mode_markers = {
    "mod_fourier_bartlett": "o",
    "mod_fourier_erasure": "h",
    "legendre_erasure": "^",
}

files = [
    utils.datafile(f"evaluate_synaptic_weight_computation_{i}.h5")
    for i in range(10)
]


errs_tuning, errs_delay = [None] * 2
for i, fn in enumerate(files):
    print(f"Loading {fn}...")
    with h5py.File(fn, "r") as f:
        if i == 0:
            solver_modes = json.loads(f.attrs["solver_modes"])
            modes = json.loads(f.attrs["modes"])
            qs = json.loads(f.attrs["qs"])
            neurons = json.loads(f.attrs["neurons"])
            xs_sigma_test = json.loads(f.attrs["xs_sigma_test"])
            errs_delay = f["errs_delay"][()]
        else:
            errs_delay_new = f["errs_delay"][()]
            invalid = np.isnan(errs_delay)
            errs_delay[invalid] = errs_delay_new[invalid]

styles = {
    "nef": {
        "color": "k",
        "linestyle": "--",
        "marker": '+',
        "markersize": 5,
    },
    "biased_xs": {
        "color": utils.blues[0],
        "linestyle": '-',
        "marker": 'x',
        "markersize": 5,
    },
    "unbiased_xs": {
        "color": utils.oranges[1],
        "linestyle": '-',
        "marker": '2',
        "markersize": 7,
    }
}

titles = {
    "mod_fourier_bartlett":
    "\\textbf{Modified Fourier}\n(Bartlett window; $q = 7$)",
    "mod_fourier_erasure":
    "\\textbf{Modified Fourier}\n(Rectangle window; $q = 7$)",
    "legendre_erasure": "\\textbf{Legendre (LDN)}\n(Rectangle window; $q = 7$)",
}

colors = ['tab:blue', 'tab:orange', 'tab:green']
fig = plt.figure(figsize=(7.5, 4.0))
gs1 = fig.add_gridspec(1, 3, wspace=0.1, bottom=0.55)
axs = [fig.add_subplot(gs1[0, i]) for i in range(3)]
for i, solver_mode in enumerate(solver_modes):
    for j, mode in enumerate(modes):
        if solver_mode == "nef" and mode == "non_lindep_cosine":
            continue
        for k, q in enumerate(qs):
            if q != 7:
                continue
            Es = np.nanmean(errs_delay[i, j, k, :, :, 0, :], axis=-1) * 100.0
            Es = Es.reshape(len(neurons), -1)
            E25 = np.nanpercentile(Es, 25, axis=1)
            E50 = np.nanpercentile(Es, 50, axis=1)
            E75 = np.nanpercentile(Es, 75, axis=1)
            if solver_mode == "nef":
                print(i, j, k, E50)
            color = colors[i]
            axs[j].errorbar(neurons,
                            E50, ((E50 - E25), (E75 - E50)),
                            capsize=2,
                            **styles[solver_mode],
                            linewidth=1.25,
                            clip_on=True,
                            zorder=100 + i)

        if i == 0:
            Es = np.nanmean(errs_delay[:, j, k, -1, :, 0, :], axis=-1) * 100.0
            axs[j].arrow(1000,
                         np.nanpercentile(Es, 90) + 12,
                         0.0,
                         -4,
                         width=75,
                         head_width=175,
                         head_length=5,
                         linewidth=0.0,
                         overhang=0.1,
                         color='k')

            axs[j].set_xlim(8, 1200)
            axs[j].set_ylim(20, 70)
            axs[j].set_xscale('log')

            if j == 0:
                axs[j].set_ylabel("Mean delay NRMSE (\\%)")
            else:
                axs[j].set_yticklabels([])
            axs[j].set_xlabel("Number of neurons $n$")
            axs[j].set_title(titles[mode])

            axs[j].text(-0.01 if j > 0 else -0.195,
                        1.16,
                        "\\textbf{{{}}}".format(chr(ord('A') + j)),
                        size=12,
                        transform=axs[j].transAxes)

            axs[j].plot(0.95,
                        1.2,
                        marker=mode_markers[mode],
                        markersize=5,
                        markeredgecolor='k',
                        markeredgewidth=0.7,
                        transform=axs[j].transAxes,
                        color=mode_colors[mode],
                        clip_on=False)

fig.legend([
    mpl.lines.Line2D([0], [0], **styles["nef"]),
    mpl.lines.Line2D([0], [0], **styles["unbiased_xs"]),
    mpl.lines.Line2D([0], [0], **styles["biased_xs"]),
], ["Standard NEF", "Na\\\"ive sampling", "Uniform activation sampling"],
           loc="upper center",
           bbox_to_anchor=(0.5, 1.04),
           ncol=3)

gs2 = fig.add_gridspec(1, 3, wspace=0.1, top=0.36, bottom=0.0)
axs = [fig.add_subplot(gs2[0, i]) for i in range(3)]

i = solver_modes.index("nef")
for k, q in enumerate(qs):
    y_max_total = 0.0
    for j, mode in enumerate(modes):
        Es = np.nanmean(errs_delay[i, j, k, -1, :, 0, :], axis=-1).flatten()
        Es = Es[~np.isnan(Es)] * 100
        median_style = dict(styles["nef"])
        median_style["marker"] = None
        axs[k].boxplot(Es,
                   positions=[j],
                   widths=[0.4],
                   bootstrap=99,
                   notch=True,
                   showfliers=False,
                   flierprops={
                       "marker": "d",
                       "markersize": 3,
                       "markerfacecolor": "k",
                       "markeredgecolor": "k",
                   },
                   medianprops=median_style,
                   boxprops={
                       "linewidth": 0.7,
                   },
                   whiskerprops={
                       "linewidth": 0.7,
                   })

        axs[k].plot(j,
                np.percentile(Es, 1) - 7,
                marker=mode_markers[mode],
                markersize=5,
                markeredgecolor='k',
                markeredgewidth=0.7,
                color=mode_colors[mode],
                clip_on=False)

        for j2 in range(j + 1, len(modes)):
            #if k == 0:
            #    continue
            Es2 = np.mean(errs_delay[i, j2, k, 3, :, 0, :], axis=-1).flatten()
            Es2 = Es2[~np.isnan(Es2)] * 100

            stat, p = scipy.stats.kstest(Es, Es2)
            stars = "***"
            if p > 0.001:
                stars = "**"
            if p > 0.01:
                stars = "*"
            if p > 0.05:
                continue

            y_max = max(np.nanpercentile(Es, 98), np.nanpercentile(Es2, 98))
            y_max_total = max(y_max, y_max_total) + 6

#            pos2 = j2 + k * (len(modes) + 1)

#            ax.plot([pos, pos2], [y_max_total, y_max_total],
#                    'k',
#                    linewidth=0.7)
#            ax.plot([pos2, pos2], [y_max_total, y_max_total - 2],
#                    'k',
#                    linewidth=0.7)
#            ax.plot([pos, pos], [y_max_total, y_max_total - 2],
#                    'k',
#                    linewidth=0.7)
#            ax.text(0.5 * (pos + pos2),
#                    y_max_total - 1,
#                    stars,
#                    ha="center",
#                    va="baseline",
#                    size=8)

            print(j, j2, p, stars)

    axs[k].text(1, 83,
            "Order $q = {}$".format(q),
            va="baseline",
            ha="center")

    axs[k].set_yticks(np.linspace(0, 100, 11), minor=True)
    axs[k].set_ylim(10, 80)
    if k == 0:
        axs[k].set_ylabel("Mean delay NRMSE $E$ (\\%)")
    else:
        axs[k].set_yticklabels([])

    axs[k].set_xticks([])
    axs[k].spines["bottom"].set_visible(False)


axs[1].set_title(
    "\\textbf{Comparison of different basis orders $q$} ($n = 1000$, standard NEF; all $p < 0.1\\%$)",
    y=1.11)

axs[0].text(-0.195, 1.16, "\\textbf{D}", size=12, transform=axs[0].transAxes)

utils.save(fig)

