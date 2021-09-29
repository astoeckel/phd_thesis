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

fig = plt.figure(figsize=(7.4, 6.8))

levels = np.linspace(-4, 0, 11)  #  For contourf below

for i_wnd, window in enumerate(["optimal", "bartlett", "erasure"]):
    gs = fig.add_gridspec(2,
                          4,
                          top=1.0 - 0.33 * i_wnd,
                          bottom=1.0 - 0.33 * (i_wnd + 1) + 0.115,
                          height_ratios=[0.8, 2.5],
                          hspace=0.1,
                          wspace=0.3)
    axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(4)]
                    for i in range(2)])

    for i_basis, basis in enumerate(
        ["fourier", "cosine", "mod_fourier", "legendre"]):
        ts, ys = mk_impulse_response(basis, window, q=11)
        axs[0, i_basis].axhline(0.0,
                                color='grey',
                                linestyle=':',
                                lw=0.5,
                                zorder=-100)
        for k in range(6):
            axs[0, i_basis].plot(ts, ys[:, k], zorder=-k)
        axs[0, i_basis].axvline(1.0, color='k', linestyle='--', lw=0.7)

        utils.remove_frame(axs[0, i_basis])
        axs[0, i_basis].set_xlim(0.0, 1.5)
        axs[0, i_basis].set_ylim(-1.5, 1.5)
        utils.annotate(axs[0, i_basis],
                       1.05,
                       1.0,
                       1.2,
                       1.0,
                       "$\\theta$",
                       va="center",
                       ha="left")
        axs[0, i_basis].set_title(
            ["Fourier", "Cosine", "Modified~Fourier", "Legendre"][i_basis])

        if (basis in BASES) and (window in WINDOWS):
            j_wnd = WINDOWS.index(window)
            j_basis = BASES.index(basis)
            C = axs[1, i_basis].contourf(
                THETAS,
                QS,
                np.log10(np.mean(errs[j_basis, j_wnd], axis=-1) + 1e-4),
                cmap="inferno",
                levels=levels)
            for c in C.collections:
                c.set_edgecolor("face")

            axs[1, i_basis].contour(
                THETAS,
                QS,
                np.log10(np.mean(errs[j_basis, j_wnd], axis=-1) + 1e-4),
                colors=['white'],
                linestyles=[':'],
                linewidths=[0.7],
                levels=C.levels)

            errs_mean = np.mean(errs[j_basis, j_wnd].reshape(-1, errs.shape[-1]), axis=0)
            ci = scipy.stats.bootstrap((errs_mean,), np.mean, confidence_level=0.99).confidence_interval
            print(basis, window, f"CI: [{ci.low:0.4f}, {ci.high:0.4f}]")

            err_total = np.mean(errs[j_basis, j_wnd])
            axs[1, i_basis].text(0.94,
                                 0.96,
                                 "$E = {:0.2f}\\%$".format(err_total * 100),
                                 va="top",
                                 ha="right",
                                 bbox={
                                     "color": "white",
                                     "pad": 0.1,
                                 },
                                 transform=axs[1, i_basis].transAxes)

        axs[1, i_basis].set_aspect('auto')
        axs[1, i_basis].set_xlim(0, 1)
        axs[1, i_basis].set_xticks(np.linspace(0, 1, 3))
        axs[1, i_basis].set_xticks(np.linspace(0, 1, 5), minor=True)
        axs[1, i_basis].set_ylim(QS[0], QS[-1])
        axs[1, i_basis].set_yticks(np.linspace(QS[0], QS[-1], 3, dtype=int))
        axs[1, i_basis].set_yticks(np.linspace(QS[0], QS[-1], 5, dtype=int),
                                   minor=True)
        axs[1, i_basis].set_xlabel("Delay $\\theta' / \\theta$")
        if i_basis == 0:
            axs[1, i_basis].set_ylabel("Basis order $q$")

    axs[0, 0].text(-0.285,
                   1.8,
                   "\\textbf{{{}}}".format(chr(ord('A') + i_wnd)),
                   size=12,
                   va="baseline",
                   ha="left",
                   transform=axs[0, 0].transAxes)

    axs[0,
        0].text(2.4,
                1.8,
                "\\textbf{{{}}}".format([
                    "Optimal rectangle window", "LTI system generating bases with an approximated Bartlett window",
                    "LTI system generating basis with an approximated rectangle window (information erasure)"
                ][i_wnd]),
                va="baseline",
                ha="center",
                transform=axs[0, 0].transAxes)

cgs = fig.add_gridspec(1, 1, left=0.125, right=0.9, top=0.0625, bottom=0.05)
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

#fig.savefig(gs)

utils.save(fig)

