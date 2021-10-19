import h5py
import scipy.linalg
from basis_delay_analysis_common import *

with h5py.File(utils.datafile("evaluate_bases_delays.h5"), "r") as f:
    try:
        BASES = json.loads(f.attrs["bases"])
        WINDOWS = json.loads(f.attrs["windows"])
    except TypeError:
        BASES = list(f.attrs["bases"])
        WINDOWS = list(f.attrs["windows"])
    QS = list(f.attrs["qs"])
    THETAS = list(f.attrs["thetas"])

    errs = f["errs"][()]

fig, axs = plt.subplots(2,
                        3,
                        figsize=(7.5, 3.5),
                        gridspec_kw={
                            "hspace": 0.3,
                            "wspace": 0.6
                        })

levels = np.linspace(-4, 0, 11)  #  For contourf below

basis_names = ["LDN", "Mod.~Fourier", "Fourier", "Cosine", "Haar", "DLOP"]
basis_wnds = [
    WINDOWS.index("erasure"),
    WINDOWS.index("erasure"),
    WINDOWS.index("optimal"),
    WINDOWS.index("optimal"),
    WINDOWS.index("optimal"),
    WINDOWS.index("optimal"),
]

basis_idcs = [
    BASES.index("legendre"),
    BASES.index("mod_fourier"),
    BASES.index("fourier"),
    BASES.index("cosine"),
    BASES.index("haar"),
    BASES.index("legendre"),
]

basis_types = ["lti", "lti", "slide", "slide", "slide", "fir"]

for i_row in range(axs.shape[0]):
    for i_col in range(axs.shape[1]):
        i = i_row * axs.shape[1] + i_col
        ax = axs[i_row, i_col]

        j_basis = basis_idcs[i]
        j_wnd = basis_wnds[i]

        C = ax.contourf(
            THETAS,
            QS,
            np.log10(np.mean(errs[j_basis, j_wnd], axis=-1) + 1e-4),
            cmap="inferno",
            levels=levels)
        for c in C.collections:
            c.set_edgecolor("face")

        ax.contour(THETAS,
                   QS,
                   np.log10(np.mean(errs[j_basis, j_wnd], axis=-1) + 1e-4),
                   colors=['white'],
                   linestyles=[':'],
                   linewidths=[0.7],
                   levels=C.levels)

        errs_mean = np.mean(errs[j_basis, j_wnd].reshape(-1, errs.shape[-1]),
                            axis=0)
        ci = scipy.stats.bootstrap((errs_mean, ),
                                   np.mean,
                                   confidence_level=0.99).confidence_interval

        err_total = np.mean(errs[j_basis, j_wnd])
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

        ax.set_aspect('auto')
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 3))
        ax.set_xticks(np.linspace(0, 1, 5), minor=True)
        ax.set_ylim(QS[0], QS[-1])
        ax.set_yticks(np.linspace(QS[0], QS[-1], 3, dtype=int))
        ax.set_yticks(np.linspace(QS[0], QS[-1], 5, dtype=int), minor=True)
        if i_row == 1:
            ax.set_xlabel("Delay $\\theta' / \\theta$")
        else:
            ax.set_xticklabels([])
        ax.set_ylabel("Basis order $q$")
        ax.set_title(f"\\textbf{{{basis_names[i]}}}")

        if basis_types[i] == "lti":
            ax.text(1.0,
                    1.065,
                    "\\emph{LTI}",
                    ha="right",
                    va="baseline",
                    transform=ax.transAxes)
            ax.plot(0.825,
                    1.105,
                    'o',
                    color=utils.blues[0],
                    markeredgecolor='k',
                    markeredgewidth=0.7,
                    transform=ax.transAxes,
                    clip_on=False)
        elif basis_types[i] == "slide":
            ax.text(1.0,
                    1.065,
                    "\\emph{SDT}",
                    ha="right",
                    va="baseline",
                    transform=ax.transAxes)
            ax.plot(0.8,
                    1.105,
                    'h',
                    color=utils.oranges[1],
                    markeredgecolor='k',
                    markeredgewidth=0.7,
                    transform=ax.transAxes,
                    clip_on=False)
        elif basis_types[i] == "fir":
            ax.text(1.0,
                    1.065,
                    "\\emph{FIR}",
                    ha="right",
                    va="baseline",
                    transform=ax.transAxes)
            ax.plot(0.825,
                    1.105,
                    '^',
                    color=utils.yellows[1],
                    markeredgecolor='k',
                    markeredgewidth=0.7,
                    transform=ax.transAxes,
                    clip_on=False)

cgs = fig.add_gridspec(1, 1, left=0.125, right=0.9, top=0.00, bottom=-0.02)
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
cax.set_xlabel("Mean NRMSE $E$")  #

utils.save(fig)

