import h5py
import numpy as np
import matplotlib.pyplot as plt


def plot_analysis(utils, fn_network, fn_analysis, fn_esn, first_letter="A"):
    with h5py.File(utils.datafile(fn_network), "r") as f:
        if "W_rec" in f:
            is_nef = False
            W_rec = f["W_rec"][()][:, :, 0]
        else:
            import dlop_ldn_function_bases
            is_nef = True
            q, d = 5, 2
            tau = 0.1
            A, B = dlop_ldn_function_bases.mk_ldn_lti(q)
            Ap = np.zeros((q * d, q * d))
            for i in range(d):
                Ap[(i * q):((i+1) * q), (i * q):((i+1) * q)] = tau * A + np.eye(q)
            W_rec = Ap

    with h5py.File(utils.datafile(fn_esn), "r") as f:
        delays_1d_esn = f["delays_1d"][()]
        errs_1d_esn = f["errs_1d"][()]

    with h5py.File(utils.datafile(fn_analysis), "r") as f:
        delays_1d = f["delays_1d"][()]
        delays_2d = f["delays_2d"][()]
        errs_1d = f["errs_1d"][()]
        errs_2d = f["errs_2d"][()]

    fig, axs = plt.subplots(1,
                            4,
                            figsize=(7.35, 2.2),
                            gridspec_kw={
                                "wspace": 0.5,
                                "width_ratios": [3, 0.25, 4, 3]
                            })
    utils.remove_frame(axs[1])

    axs[0].plot(delays_1d_esn,
                errs_1d_esn[0] * 1e2,
                color=utils.blues[0],
                linestyle="--",
                lw=0.5)
    axs[0].plot(delays_1d_esn,
                errs_1d_esn[1] * 1e2,
                color=utils.oranges[1],
                linestyle="--",
                lw=0.5)

    axs[0].plot(delays_1d,
                errs_1d[0] * 1e2,
                color=utils.blues[0],
                label="$\\hat{\mathfrak{x}}_1(t - \\theta'_1)$")
    axs[0].plot(delays_1d,
                errs_1d[1] * 1e2,
                color=utils.oranges[1],
                label="$\\hat{\mathfrak{x}}_2(t - \\theta'_2)$")

    axs[0].set_ylim(0, 100)
    axs[0].set_xlim(0, 1)
    axs[0].set_yticks(np.arange(0, 100, 10), minor=True)
    axs[0].set_xticks(np.arange(0, 1, 0.25), minor=True)

    axs[0].axhline(np.mean(errs_1d) * 1e2, color='k', linestyle='--', lw=0.7)
    axs[0].set_xlabel("Delay $\\theta_i'/\\theta$")
    axs[0].set_ylabel("Decoding error (NRMSE)")

    axs[0].legend(loc="upper right",
                  handlelength=1.0,
                  handletextpad=0.5,
                  borderpad=0.0,
                  bbox_to_anchor=(1.0, 1.05),
                  facecolor="white",
                  framealpha=1.0,
                  edgecolor="white",
                  frameon=True)
    utils.outside_ticks(axs[0])
    utils.annotate(axs[0], 0.4, (np.mean(errs_1d) + 0.015) * 100.0, 0.65, 30,
                   "$\\bar E = {:0.1f}\\%$".format(np.mean(errs_1d) * 100.0))
    utils.annotate(axs[0], 0.49, 50, 0.75, 50, "ESN")

    C = axs[2].contourf(delays_2d,
                        delays_2d,
                        errs_2d * 1e2,
                        vmin=0.0,
                        vmax=100.0,
                        levels=np.linspace(0, 100, 11))
    axs[2].contour(delays_2d,
                   delays_2d,
                   errs_2d * 1e2,
                   vmin=0.0,
                   vmax=100.0,
                   colors=["white"],
                   linestyles=[":"],
                   linewidths=[0.7],
                   levels=C.levels)

    cax = fig.add_axes([0.315, 0.11, 0.02, 0.78])
    cbar = plt.colorbar(C, ax=axs[2], cax=cax)
    cbar.outline.set_visible(False)
    cax.set_ylim(0, 100)
    #cax.yaxis.set_ticks_position('left')
    #cax.yaxis.set_label_position('left')
    utils.outside_ticks(cax)

    axs[2].text(0.03,
                0.97,
                "$\\bar E = {:0.1f}\\%$".format(np.mean(errs_2d) * 1e2),
                transform=axs[2].transAxes,
                bbox={
                    "color": "white",
                    "pad": 0.7
                },
                va="top",
                ha="left")
    axs[2].set_xticks(np.arange(0, 1.0, 0.2))
    axs[2].set_xticks(np.arange(0, 1.0, 0.1), minor=True)
    axs[2].set_yticks(np.arange(0, 1.0, 0.2))
    axs[2].set_yticks(np.arange(0, 1.0, 0.1), minor=True)
    axs[2].set_xlabel("Delay $\\theta_1'/\\theta$")
    axs[2].set_ylabel("Delay $\\theta_2'/\\theta$")
    utils.outside_ticks(axs[2])

    iy, ix = np.unravel_index(np.argmin(errs_2d), errs_2d.shape)
    axs[2].plot(delays_2d[ix],
                delays_2d[iy],
                '+',
                color="white",
                markersize=6,
                markeredgewidth=2)
    axs[2].plot(delays_2d[ix], delays_2d[iy], 'k+', markersize=5)
    utils.annotate(axs[2],
                   delays_2d[ix] + 0.02,
                   delays_2d[iy] + 0.025,
                   delays_2d[ix] + 0.2,
                   delays_2d[iy] + 0.2,
                   '$E = {:0.1f}\\%$'.format(errs_2d[iy, ix] * 100),
                   ha="left",
                   va="bottom")

    _, S, _ = np.linalg.svd(W_rec)
    axs[3].plot(np.arange(1, len(S) + 1), S / np.max(S), 'k:', lw=0.5)
    axs[3].plot(np.arange(1, len(S) + 1), S / np.max(S), 'k+')
    axs[3].set_yscale("log")
    axs[3].set_xlim(0.5, 15.5)
    axs[3].set_ylim(1e-4, 1.5)
    axs[3].set_xticks([1, 5, 10, 15])
    axs[3].set_ylabel("Singular value $\\sigma_i / \\sigma_1$")
    axs[3].set_xlabel("Singular value index")

    axs[3].set_xticks(np.arange(1, 15), minor=True)

    fig.text(0.061,
             0.95,
             "\\textbf{{{}}}".format(chr(ord(first_letter) + 0)),
             size=12,
             ha="left",
             va="baseline")
    fig.text(0.315,
             0.95,
             "\\textbf{{{}}}".format(chr(ord(first_letter) + 1)),
             size=12,
             ha="left",
             va="baseline")
    fig.text(0.6775,
             0.95,
             "\\textbf{{{}}}".format(chr(ord(first_letter) + 2)),
             size=12,
             ha="left",
             va="baseline")

    fig.text(0.21,
             0.95,
             "\\textbf{Decoding delays}",
             ha="center",
             va="baseline")
    fig.text(0.51,
             0.95,
             "\\textbf{Decoding delayed multiplication}",
             ha="center",
             va="baseline")

    if is_nef:
        fig.text(0.815,
                 0.95,
                 "\\textbf{$\\mat A'$ singular values}",
                 ha="center",
                 va="baseline")
    else:
        fig.text(0.815,
                 0.95,
                 "\\textbf{$\\mathbf{W}_\\mathrm{rec}$ singular values}",
                 ha="center",
                 va="baseline")

    fig.align_labels(axs)

    return fig, axs

