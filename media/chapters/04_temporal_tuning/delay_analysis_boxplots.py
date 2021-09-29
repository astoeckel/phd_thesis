import h5py
import scipy.stats
import scipy.linalg
from basis_delay_analysis_common import *

SYMBOLS = {
	"fourier": {
		"marker": "o",
		"markeredgecolor": utils.blues[0],
		"markerfacecolor": utils.blues[0],
		"markersize": 5,
	},
	"cosine": {
		"marker": "o",
		"markeredgecolor": utils.oranges[1],
		"markerfacecolor": utils.oranges[1],
		"markersize": 5,
	},
	"mod_fourier": {
		"marker": "o",
		"markeredgecolor": utils.blues[0],
		"markerfacecolor": "white",
		"markersize": 5,
	},
	"legendre": {
		"marker": "s",
		"markeredgecolor": utils.greens[0],
		"markerfacecolor": utils.greens[0],
		"markersize": 5,
	}
}

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

fig, axs = plt.subplots(1,
                        3,
                        figsize=(7.55, 2.0),
                        gridspec_kw={
                            "wspace": 0.4,
                        })

for i_wnd, window in enumerate(["optimal", "bartlett", "erasure"]):
    y_max_total = 0.0
    ax = axs[i_wnd]
    for i_basis, basis in list(
            enumerate(["fourier", "cosine", "mod_fourier", "legendre"]))[::-1]:
        if not ((basis in BASES) and (window in WINDOWS)):
            continue

        j_wnd = WINDOWS.index(window)
        j_basis = BASES.index(basis)

        errs_mean = np.mean(errs[j_basis, j_wnd].reshape(-1, errs.shape[-1]),
                            axis=0) * 100
        ax.boxplot(errs_mean,
                   positions=[i_basis],
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
        if np.median(errs_mean) > 22:
            ax.arrow(i_basis, (0.1875 + 0.01) * 100,
                     0.0,
                     0.0075 * 100,
                     width=0.07,
                     head_width=0.2,
                     head_length=0.005 * 100,
                     linewidth=0.0,
                     overhang=0.1,
                     color='k')

        # Statistical significance test
        last_res = None
        for i_basis2, basis2 in enumerate(
            ["fourier", "cosine", "mod_fourier", "legendre"]):
            if (not (basis2 in BASES)) or (i_basis2 <= i_basis):
                continue

            j_basis2 = BASES.index(basis2)

            errs_mean1 = errs_mean
            errs_mean2 = np.mean(errs[j_basis2, j_wnd].reshape(
                -1, errs.shape[-1]),
                                 axis=0) * 100

            if np.median(errs_mean1) > 22 or np.median(errs_mean2) > 22:
                continue

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

            ax.plot([i_basis, i_basis2], [y_max_total, y_max_total],
                    'k',
                    linewidth=0.7)
            ax.plot([i_basis, i_basis], [y_max_total, y_max_total - 0.2],
                    'k',
                    linewidth=0.7)
            ax.plot([i_basis2, i_basis2], [y_max_total, y_max_total - 0.2],
                    'k',
                    linewidth=0.7)
            ax.text(0.5 * (i_basis + i_basis2),
                    y_max_total - 0.2,
                    stars,
                    ha="center",
                    va="baseline",
                    size=8)

    ax.set_ylim(8.5, 21)
    ax.set_ylabel("Mean NRMSE $E$ (\\%)")
    ax.set_title("\\textbf{{{}}}".format([
        "Optimal rectangle window", "Approx.~Bartlett window",
        "Approx.~rectangle window"
    ][i_wnd]))
    ax.text(-0.29,
            1.054,
            "\\textbf{{{}}}".format(chr(ord("A") + i_wnd)),
            size=12,
            transform=ax.transAxes)
    ax.set_xticklabels([])
    ax.set_xticklabels(["Fourier", "Cosine", "Mod.~Fourier", "Legendre"][::-1],
                       rotation=25, ha="right")

utils.save(fig)

