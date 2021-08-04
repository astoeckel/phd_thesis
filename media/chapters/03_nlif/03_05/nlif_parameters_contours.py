# Helper-function used to plot a contour plot with colour bar
def plot_contour(ax, cax, xs, ys, zs, zs2, levels, ylabel="", zlabel=""):
    contour = ax.contourf(xs, ys, zs, levels)
    ax.set_xlim(np.min(xs), np.max(xs))
    ax.set_ylim(np.min(ys), np.max(ys))
    ax.set_ylabel(ylabel)

    ax.contour(xs,
               ys,
               zs2,
               levels=levels,
               colors=['w'],
               linewidths=[1.0],
               linestyles=['--'])

    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))

    if not cax is None:
        cb = ax.get_figure().colorbar(contour,
                                      ax=ax,
                                      cax=cax,
                                      orientation='vertical')
        cb.outline.set_visible(False)
        cax.set_ylabel(zlabel, labelpad=1.0)
        utils.outside_ticks(cax)


def plot_cross_section(ax,
                       axcross,
                       y0,
                       xs,
                       ys,
                       zs,
                       zs2,
                       xlabel="",
                       levels=None):
    ys_median = ys[len(ys) // 2]
    ax.plot(xs, np.ones_like(xs) * ys_median, 'k', linewidth=1)
    ax.plot(xs,
            np.ones_like(xs) * ys_median,
            'w',
            linewidth=1,
            linestyle=(0, (1, 1)))
    axcross.plot(xs, zs[len(ys) // 2, :], linewidth=1, color='k')
    axcross.plot(xs, zs2[len(ys) // 2, :], linewidth=1, color='k')
    axcross.plot(xs,
                 zs2[len(ys) // 2, :],
                 linewidth=1,
                 color='w',
                 linestyle=(1, (1, 2)))
    axcross.set_xlim(np.min(xs), np.max(xs))
    axcross.set_ylim(y0, None if levels is None else levels[-1])
    axcross.set_xlabel(xlabel, labelpad=2.5)
    axcross.set_ylabel('Rate ($\\mathrm{s}^{-1}$)')

    axcross.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    axcross.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))


def write_error(ax, Zmodel, Ztruth, mul, unit, clip=True):
    if clip:
        valid = np.logical_or(Ztruth > 12.5, Zmodel > 12.5)
        E = np.sqrt(np.mean((Zmodel[valid] - Ztruth[valid])**2))
    else:
        E = np.sqrt(np.mean((Zmodel - Ztruth)**2))
    if mul is None:
        scale = np.log10(E + 1e-12)
        if scale <= -9:
            mul = 1e12
            unit = 'p' + unit
        elif scale <= -6:
            mul = 1e9
            unit = 'n' + unit
        else:
            mul = 1e6
            unit = '\\mu ' + unit
    ax.text(0.03, 0.9, "$E = {:.1f}\,\\mathrm{{{}}}$".format(E * mul, unit), transform=ax.transAxes, va="baseline", ha="left", bbox={\
        'pad': 1.5,
        'facecolor': 'w',
        'linewidth': 0
    }, size=8)


def plot_analysis(ax, ax_cross, cax, fn, res=100, xlabel=None, ylabel=None):
    with np.load(utils.datafile(fn), allow_pickle=True) as data:
        utils.outside_ticks(ax)
        utils.outside_ticks(ax_cross)

        levels_rate = np.linspace(0, 100, 11)

        plot_contour(ax,
                     cax,
                     data["c1s"] * 1e9,
                     data["c2s"] * 1e9,
                     data["As"],
                     data["As_pred"],
                     levels=levels_rate,
                     ylabel=ylabel,
                     zlabel="Average spike rate ($\mathrm{s}^{-1}$)")
        plot_cross_section(ax,
                           ax_cross,
                           0.0,
                           data["c1s"] * 1e9,
                           data["c2s"] * 1e9,
                           data["As"],
                           data["As_pred"],
                           xlabel=xlabel,
                           levels=levels_rate)
        write_error(ax, data["As_pred"], data["As"], 1, 's^{-1}', clip=True)

        ax.set_xticklabels([])

    return ax, ax_cross

fig = plt.figure(figsize=(6.5, 4.4))
gs1 = fig.add_gridspec(2,
                       3,
                       height_ratios=[4, 1],
                       wspace=0.5,
                       hspace=0.2,
                       top=0.95,
                       bottom=0.575)
gs2 = fig.add_gridspec(2,
                       3,
                       height_ratios=[4, 1],
                       wspace=0.5,
                       hspace=0.2,
                       top=0.375,
                       bottom=0.0)
cgs = fig.add_gridspec(1, 1, left=0.925, right=0.94, top=1.0322, bottom=-0.091)
cax = fig.add_subplot(cgs[0, 0])

plot_analysis(
    fig.add_subplot(gs1[0, 0]),
    fig.add_subplot(gs1[1, 0]),
    cax,
    "nlif_params_contour_two_comp_lif.npz",
    xlabel="$g_\\mathrm{E}$ ($\\mathrm{nS}$)",
    ylabel="$g_\\mathrm{I}$ ($\\mathrm{nS}$)",
)

plot_analysis(
    fig.add_subplot(gs1[0, 1]),
    fig.add_subplot(gs1[1, 1]),
    None,
    "nlif_params_contour_three_comp_lif_1.npz",
    xlabel="$g_\\mathrm{E}^2$ ($\\mathrm{nS}$)",
    ylabel="$g_\\mathrm{I}^2$ ($\\mathrm{nS}$)",
)

plot_analysis(
    fig.add_subplot(gs1[0, 2]),
    fig.add_subplot(gs1[1, 2]),
    None,
    "nlif_params_contour_three_comp_lif_2.npz",
    xlabel="$g_\\mathrm{E}^1$ ($\\mathrm{nS}$)",
    ylabel="$g_\\mathrm{I}^2$ ($\\mathrm{nS}$)",
)

ax, axcross = plot_analysis(
    fig.add_subplot(gs2[0, 0]),
    fig.add_subplot(gs2[1, 0]),
    None,
    "nlif_params_contour_four_comp_lif_1.npz",
    xlabel="$g_\\mathrm{E}^1$ ($\\mathrm{nS}$)",
    ylabel="$g_\\mathrm{I}^3$ ($\\mathrm{nS}$)",
)
ax.set_ylabel("$g_\\mathrm{I}^3$ ($\\mathrm{nS}$)", labelpad=2.5)
axcross.set_ylabel("Rate $\\mathrm{s^{-1}}$", labelpad=2.5)

plot_analysis(
    fig.add_subplot(gs2[0, 1]),
    fig.add_subplot(gs2[1, 1]),
    None,
    "nlif_params_contour_four_comp_lif_2.npz",
    xlabel="$g_\\mathrm{E}^2$ ($\\mathrm{nS}$)",
    ylabel="$g_\\mathrm{I}^3$ ($\\mathrm{nS}$)",
)

plot_analysis(
    fig.add_subplot(gs2[0, 2]),
    fig.add_subplot(gs2[1, 2]),
    None,
    "nlif_params_contour_four_comp_lif_3.npz",
    xlabel="$g_\\mathrm{E}^3$ ($\\mathrm{nS}$)",
    ylabel="$g_\\mathrm{I}^3$ ($\\mathrm{nS}$)",
)

fig.add_artist(
    mpl.patches.Rectangle((0.047, 0.48),
                          0.2925,
                          0.55,
                          linewidth=0.5,
                          edgecolor='k',
                          facecolor='#f6f6f6',
                          transform=fig.transFigure,
                          zorder=-10))

fig.add_artist(
    mpl.patches.Rectangle((0.345, 0.48),
                          0.575,
                          0.55,
                          linewidth=0.5,
                          edgecolor='k',
                          facecolor='#f6f6f6',
                          transform=fig.transFigure,
                          zorder=-10))

fig.add_artist(
    mpl.patches.Rectangle((0.047, -0.09),
                          0.873,
                          0.562,
                          linewidth=0.5,
                          edgecolor='k',
                          facecolor='#f6f6f6',
                          transform=fig.transFigure,
                          zorder=-10))

fig.text(0.05, 1.0, "\\textbf{A}", va="baseline", ha="left", size=12, transform=fig.transFigure)
fig.text(0.22, 1.0, "\\textbf{Two-compartment LIF}", va="baseline", ha="center", transform=fig.transFigure)
#fig.text(0.22, 0.9625, "", va="baseline", ha="center", transform=fig.transFigure)

fig.text(0.3475, 1.0, "\\textbf{B}", va="baseline", ha="left", size=12, transform=fig.transFigure)
fig.text(0.66, 1.0, "\\textbf{Three-compartment LIF}", va="baseline", ha="center", transform=fig.transFigure)
fig.text(0.51, 0.9625, "Apical excitation", va="baseline", ha="center", transform=fig.transFigure)
fig.text(0.81, 0.9625, "Basal excitation", va="baseline", ha="center", transform=fig.transFigure)

fig.text(0.05, 0.433, "\\textbf{C}", va="baseline", ha="left", size=12, transform=fig.transFigure)
fig.text(0.51, 0.433, "\\textbf{Four-compartment LIF}", va="baseline", ha="center", transform=fig.transFigure)
fig.text(0.22, 0.3925, "Basal excitation", va="baseline", ha="center", transform=fig.transFigure)
fig.text(0.51, 0.3925, "Proximal excitation", va="baseline", ha="center", transform=fig.transFigure)
fig.text(0.81, 0.3925, "Distal excitation", va="baseline", ha="center", transform=fig.transFigure)


utils.save(fig)

