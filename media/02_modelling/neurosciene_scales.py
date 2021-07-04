fig, axs = plt.subplots(1, 2, figsize=(7.75, 2.5))

axs[0].set_xscale("log")
axs[0].set_yscale("linear")
axs[0].xaxis.set_ticks(np.logspace(-9, 0, 4), minor=False)
axs[0].set_xlim(1e-10, 1e1)
axs[0].set_ylim(0, 1)
axs[0].xaxis.set_ticklabels([
    "$\\mathbf{10^{-9}}$\n$\\mathbf{\\si{\\nano\\metre}}$",
    "$\\mathbf{10^{-6}}$\n$\\mathbf{\\si{\\micro\\metre}}$",
    "$\\mathbf{10^{-3}}$\n$\\mathbf{\\si{\\milli\\metre}}$",
    "$\\mathbf{10^{0}}$\n$\\mathbf{\\si{\\metre}}$",
])
axs[0].set_xticks(np.logspace(-10, 1, 12), minor=True)
axs[0].set_xticklabels([], minor=True)
axs[0].set_yticklabels([])
axs[0].spines["left"].set_visible(False)
axs[0].set_yticks([])
axs[0].set_xlabel("\\textbf{Spatial scale} (in metres)")
axs[0].text(-0.04,
            1.0,
            "\\textbf{A}",
            ha="left",
            va="top",
            size=12,
            transform=axs[0].transAxes)

axs[1].set_xscale("log")
axs[1].set_yscale("linear")
axs[1].xaxis.set_ticks(np.logspace(-9, 0, 4), minor=False)
axs[1].set_ylim(0, 1)
axs[1].set_xlim(1e-3, 3e9)
axs[1].set_xticks(np.logspace(-3, 9, 5), minor=False)
axs[1].set_xticklabels([
    "$\\mathbf{10^{-3}}$\n{Millisecond}",
    "$\\mathbf{10^{0}}$\n{Second}",
    "$\\mathbf{10^{3}}$\n{Minutes}",
    "$\\mathbf{10^{6}}$\n{Days}",
    "$\\mathbf{10^{9}}$\n{Decades}",
])
axs[1].set_xticklabels([], minor=True)
axs[1].set_yticklabels([])
axs[1].spines["left"].set_visible(False)
axs[1].set_yticks([])
axs[1].set_xlabel("\\textbf{Temporal scale} (in seconds)")
axs[1].text(0.0,
            1.0,
            "\\textbf{B}",
            ha="left",
            va="top",
            size=12,
            transform=axs[1].transAxes)


def annotate(ax, xs, y, s, va="bottom", ha="center", txoffs=0.0, tyoffs=0.0, axoffs=0.0, ayoffs=0.0, ascale=0.2):
    if not hasattr(xs, '__len__'):
        xs = {xs}

    if len(xs) == 0:
        return

    x0, x1 = min(xs), max(xs)

    cx = np.power(10, 0.5 * (np.log10(x0) + np.log10(x1)) + txoffs)
    cx0 = np.power(10, 0.5 * (np.log10(x0) + np.log10(x1)) + axoffs)
    cx1 = np.power(10, 0.5 * (np.log10(x0) + np.log10(x1)) + ascale * txoffs)
    cy = y + 0.0175 + tyoffs
    if x0 == x1:
        ax.plot(x0, y, 'k+', markersize=5, markeredgewidth=1.0, clip_on=False)
    else:
        utils.timeslice(ax, x0, x1, y)
    ax.text(cx, cy, s, va=va, ha=ha)
    if cx != cx0 or tyoffs != 0.0:
        utils.annotate(ax, cx0, y + (0.015 if x0 != x1 else 0.03) + ayoffs, cx1,
                       cy + 0.005)

    for x in xs:
        if x == x0 or x == x1:
            continue
        ax.plot(x, y, 'kd', markersize=1)


P0, Poffs = 0.075, 0.14
P = lambda i: P0 + i * Poffs

annotate(axs[0], {133e-12, 95e-12, 167e-12},
         P(0),
         'Ions',
         txoffs=0.5,
         tyoffs=0.6 * Poffs,
         ha="right")
annotate(axs[0], {5e-9, 5.5e-9, 15e-9, 18e-9}, P(0), 'Ion channels', txoffs=-0.4, tyoffs=0.125*Poffs)

annotate(axs[0], {50e-9}, P(0), 'Vesicle', txoffs=1.2, tyoffs=0.125*Poffs, axoffs=0.1, ayoffs=-0.0125, ascale=0.4)

annotate(axs[0], {15e-2, 21e-2}, P(4), 'Brain')

annotate(axs[0], {0.2e-6, 20e-6, 2}, P(0), 'Axons')

annotate(axs[0], {400e-6, 700e-6, 1000e-6},
         P(3),
         'Cortical maps',
         txoffs=-1.15,
         tyoffs=Poffs)

annotate(axs[0], {0.3, 2}, P(5), 'Body')
annotate(axs[0], {100e-6, 2}, P(2), 'Neurons')
annotate(axs[0], {50e-6}, P(1), 'Soma')
annotate(axs[0], {100e-6, 2e-3}, P(1), 'Dendrites')
annotate(axs[0], {1e-9, 10e-6}, P(1), 'Molecules')
annotate(axs[0], {5e-6, 20e-6}, P(2), 'Synapses')
annotate(axs[0], {0.1e-3, 1e-3},
         P(3),
         'Brain networks',
         txoffs=-1.15,
         tyoffs=0.25 * Poffs)

utils.save(fig)

