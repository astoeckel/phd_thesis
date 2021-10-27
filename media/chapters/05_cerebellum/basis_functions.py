def plot_input(ax, ts, xs, y0, y1):
    t0 = None
    for t, x0, x1 in zip(ts, xs[:-1], xs[1:]):
        if x0 != x1:
            if x1:
                t0 = t
            else:
                ax.plot([t0, t], [y1, y1],
                        'k-',
                        linewidth=2,
                        solid_capstyle='butt',
                        clip_on=False)
                ax.fill_betweenx([y0, y1], [t0, t0], [t, t],
                                 color="#C0C0C0",
                                 linewidth=0)


def plot_signals(ax, ts, xs, ys):
    y0, y1 = -1.1, 1.1

    # Normalise the input
    ys = ys - ys[0, :]
    ys = ys / np.max(np.abs(ys), axis=0)

    cmap = mpl.cm.get_cmap("tab10")

    plot_input(ax, ts, xs, y0, y1)
    for i in range(ys.shape[1]):
        j = ys.shape[1] - i - 1
        ax.plot(ts, ys[:, j], color=cmap(j / 10))
    ax.set_xlabel("Time $t$ (\\si{\\milli\\second})")
    ax.set_xlim(-100, 500)
    ax.set_ylim(y0, y1)
    ax.axvline(0, linestyle='--', color='k', linewidth=0.5)
    ax.axvline(400, linestyle='--', color='k', linewidth=0.5)

    ax.set_xticks([0, 200, 400])
    ax.set_xticks([0, 100, 200, 300, 400], minor=True)


def plot_row(axs, titles, labels, fn):
    data = np.load(utils.datafile(fn))
    ts = data["ts"][0]
    xs = data["xs"][0]
    ys = data["ys"][0]
    basis = data["basis"][0]
    sigmas = data["sigma"]
    sigmas = sigmas / sigmas[:, 0][:, None]

    sigma = np.median(sigmas, axis=0)
    sigma_p25 = np.percentile(sigmas, 25, axis=0)
    sigma_p75 = np.percentile(sigmas, 75, axis=0)

    t0, t1 = 1.4, 2.0
    idcs = np.logical_and(ts >= t0, ts < t1)
    ts = (ts[idcs] - 1.5) * 1e3
    xs = xs[idcs]
    ys = ys[idcs]
    basis = basis[idcs]

    plot_signals(axs[0], ts, xs, ys)
    axs[0].set_ylabel("${\hat m}_i(t)$")

    plot_signals(axs[1], ts, xs, basis)
    axs[1].set_ylabel("$x_i(t)$")

    bidcs = np.arange(0, 6) + 1
    axs[2].plot(bidcs,
                sigma[:6],
                    color="k",
                    linestyle=":",
                    linewidth=0.7,
                    clip_on=False)
    for i in range(6):
        color = mpl.cm.get_cmap('tab10')(i / 10)
        axs[2].plot(bidcs[i],
                    sigma[i],
                    'o',
                    markersize=3,
                    color=color,
                    clip_on=False)
        axs[2].errorbar(bidcs[i],
                        sigma[i],
                        ((sigma[i] - sigma_p25[i],), (sigma_p75[i] - sigma[i],)),
                        color=color,
                        clip_on=False,
                        capsize=3)
    axs[2].set_ylim(0, 1)
    axs[2].set_xticks(bidcs)
    axs[2].set_ylabel("$\\sigma_i$")
    axs[2].set_yticks(np.linspace(0, 1, 5), minor=True)

    Sigma = np.sum(sigma[:6]) #np.sqrt(np.sum(np.square(sigma)))
    if sigma[3] > 0.5:
        axs[2].text(0.05,
                    0.1,
                    #"$\\|\\vec \\sigma\\| = {:0.2f}$".format(Sigma),
                    "$\\Sigma = {:0.2f}$".format(Sigma),
                    ha="left",
                    va="bottom",
                    transform=axs[2].transAxes)
    else:
        axs[2].text(1.0,
                    1.0,
                    #"$\\|\\vec \\sigma\\| = {:0.2f}$".format(Sigma),
                    "$\\Sigma = {:0.2f}$".format(Sigma),
                    ha="right",
                    va="top",
                    transform=axs[2].transAxes)

    for i in range(3):
        utils.outside_ticks(axs[i])

    if labels:
        axs[2].set_xlabel("Basis function index $i$")
    else:
        axs[0].set_xlabel("")
        axs[1].set_xlabel("")

    if titles:
        axs[0].set_title("\\textbf{Decoded granule activities}", y=1.5)
        axs[1].set_title("\\textbf{Granule activity SVD}", y=1.5)
        axs[2].set_title("\\textbf{Singular values}", y=1.5)


fig, axs = plt.subplots(7,
                        3,
                        figsize=(7.5, 8.0),
                        gridspec_kw={
                            "wspace": 0.4,
                            "hspace": 1.2,
                            "width_ratios": [3, 3, 2]
                        })

plot_row(axs[0], True, False, "temporal_basis_direct.npz")
plot_row(axs[1], False, False, "temporal_basis_single_population.npz")
plot_row(axs[2], False, False, "temporal_basis_two_populations.npz")
plot_row(axs[3], False, False,
         "temporal_basis_two_populations_dales_principle.npz")
plot_row(axs[4], False, False,
         "temporal_basis_two_populations_dales_principle_detailed.npz")
plot_row(
    axs[5], False, False,
    "temporal_basis_two_populations_dales_principle_detailed_control.npz")
plot_row(
    axs[6], False, True,
    "temporal_basis_two_populations_dales_principle_detailed_no_jbias.npz")

for i in range(7):
    #    if i < 4:
    axs[i, 0].text(-0.225,
                   1.25,
                   "\\textbf{{{}}}".format(chr(ord('A') + i)),
                   ha="left",
                   va="baseline",
                   size=12,
                   transform=axs[i, 0].transAxes)
    #    else:
    #        axs[i, 0].text(-0.225,
    #                       1.5,
    #                       "\\textbf{{E.{}}}".format(i - 3),
    #                       ha="left",
    #                       va="baseline",
    #                       size=12,
    #                       transform=axs[i, 0].transAxes)
    axs[i, 0].text(
        -0.1,
        1.25,
        "{{{}}}".format([
            "Direct implementation",
            "Single population",
            "Two populations",
            "Two populations with Dale's principle",
            "Two populations with Dale's principle and sparsity",
            "Two populations with Dale's principle and sparsity (random control)",
            "Two populations with Dale's principle and sparsity (no intrinsic biases)",
        ][i]),
        ha="left",
        va="baseline",
        transform=axs[i, 0].transAxes)

utils.save(fig)

