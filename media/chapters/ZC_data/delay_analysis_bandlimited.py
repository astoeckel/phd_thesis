import h5py
import scipy.linalg
from basis_delay_analysis_common import *

with h5py.File(utils.datafile("evaluate_bases_delays_bandlimited.h5"),
               "r") as f:
    try:
        BASES = json.loads(f.attrs["bases"])
        WINDOWS = json.loads(f.attrs["windows"])
    except TypeError:
        BASES = list(f.attrs["bases"])
        WINDOWS = list(f.attrs["windows"])
    QS = list(f.attrs["qs"])
    THETAS = list(f.attrs["thetas"])

    errs = f["errs"][()]

errs = np.array([
    errs[BASES.index("mod_fourier"),
         WINDOWS.index("bartlett")],
    errs[BASES.index("mod_fourier"),
         WINDOWS.index("erasure")],
    errs[BASES.index("legendre"),
         WINDOWS.index("erasure")],
])

fig = plt.figure(figsize=(7.4, 1.0))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 2], wspace=0.4)
axs1 = np.array([fig.add_subplot(gs[0, i]) for i in range(3)])
ax2 = fig.add_subplot(gs[0, 3])

levels = np.linspace(-3, 0, 11)  #  For contourf below


def plot_contour(ax, errs):
    C = ax.contourf(THETAS,
                    QS,
                    np.log10(np.mean(errs, axis=-1) + 1e-3),
                    cmap="inferno",
                    levels=levels)

    ax.set_aspect('auto')
    ax.set_xlim(0, THETAS[-1])
    ax.set_xticks(np.linspace(0, 1, 3))
    ax.set_xticks(np.linspace(0, 1, 5), minor=True)
    ax.set_ylim(QS[0], QS[-1])
    ax.set_yticks(np.linspace(QS[0], QS[-1], 3, dtype=int))
    ax.set_yticks(np.linspace(QS[0], QS[-1], 5, dtype=int), minor=True)
    ax.set_xlabel("Delay $\\theta' / \\theta$")
    ax.set_ylabel("Basis order $q$")

    err_total = np.mean(errs)
    ax.text(0.94,
            0.96,
            "$E = {:0.2f}\\%$".format(err_total * 100),
            va="top",
            ha="right",
            bbox={
                "color": "white",
                "pad": 0.1,
            },
            transform=ax.transAxes)


plot_contour(axs1[0], errs[0])
axs1[0].set_title("\\textbf{(1) Bartlett}\nMod. Fourier")

plot_contour(axs1[1], errs[1])
axs1[1].set_title("\\textbf{(2) Rectangle}\nMod. Fourier")

plot_contour(axs1[2], errs[2])
axs1[2].set_title("\\textbf{(3) Rectangle}\nLegendre (LDN)")

y_max_total = 0.0
for i in range(3):
    errs_mean = np.mean(errs[i].reshape(-1, errs.shape[-1]), axis=0) * 100
    ax2.boxplot(errs_mean,
                positions=[i + 1],
                labels=[f"({i + 1})"],
                vert=False,
                widths=[0.5],
                bootstrap=99,
                notch=True,
                showfliers=False,
                flierprops={
                    "marker": "d",
                    "markersize": 3,
                    "markerfacecolor": "k",
                    "markeredgecolor": "k",
                },
                medianprops={
                    "color": "k",
                    "linestyle": ":",
                    "linewidth": 0.7,
                },
                boxprops={
                    "linewidth": 0.7,
                },
                whiskerprops={
                    "linewidth": 0.7,
                })
    ax2.set_ylim(3.5, 0.5)

    for j in range(i + 1, 3):
        errs_mean1 = errs_mean
        errs_mean2 = np.mean(errs[j].reshape(-1, errs.shape[-1]), axis=0) * 100

        stat, p = scipy.stats.kstest(errs_mean1, errs_mean2)
        stars = "***"
        if p > 0.001:
            stars = "**"
        if p > 0.01:
            stars = "*"
        if p > 0.05:
            continue

        y_max = max(np.max(errs_mean1), np.max(errs_mean2))
        y_max_total = max(y_max, y_max_total) + 0.75

        ax2.plot([y_max_total, y_max_total], [i + 1, j + 1],
                 'k',
                 linewidth=0.7)
        ax2.plot([y_max_total, y_max_total - 0.2], [i + 1, i + 1],
                 'k',
                 linewidth=0.7)
        ax2.plot([y_max_total, y_max_total - 0.2], [j + 1, j + 1],
                 'k',
                 linewidth=0.7)
        ax2.text(y_max_total + 0.3,
                 0.5 * (i + j) + 1,
                 stars,
                 ha="center",
                 va="center",
                 size=8,
                 rotation=90)

ax2.set_title("\\textbf{Mean delay decoding error}")
ax2.set_xlabel("Mean NRMSE $E$ (\\%)")

cgs = fig.add_gridspec(1, 1, left=0.125, right=0.9, top=-0.3, bottom=-0.4)
cax = fig.add_subplot(cgs[0, 0])

xss, yss = np.meshgrid(np.power(10, levels), [0, 1])
cax.pcolormesh(xss,
               yss,
               np.linspace(0 + 0.5 / len(levels), 1 - 0.5 / len(levels),
                           len(levels)).reshape(1, -1),
               cmap='inferno',
               shading='auto')
cax.set_xscale('log')
cax.set_yticks([])
cax.spines["left"].set_visible(False)
cax.set_xlabel("Mean NRMSE $E$")

utils.save(fig)

