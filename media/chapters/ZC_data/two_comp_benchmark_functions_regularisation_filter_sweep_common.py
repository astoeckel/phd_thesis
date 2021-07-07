import h5py
import scipy.interpolate
from two_comp_parameters import *

def do_plot(subth=True):
    with h5py.File(
            utils.datafile(
                "two_comp_benchmark_functions_regularisation_filter_sweep.h5")
    ) as f:
        network = bool(f["network"][()])
        tau_pre_filts = f["tau_pre_filts"][()]
        param_keys = str(f["params_keys"][()], "utf-8").split("\n")
        errs = f["errs"][()]

    param_keys_idx_map = {
        "linear": (0, 0),
        "linear_2d": (0, 1),
        "gc50_no_noise": (1, 0),
        "gc50_noisy": (2, 0),
        "gc100_no_noise": (1, 1),
        "gc100_noisy": (2, 1),
        "gc200_no_noise": (1, 2),
        "gc200_noisy": (2, 2),
    }

    param_keys_subtitle_map = {
        "linear": "Single layer",
        "linear_2d": "Two layers",
        "gc50_no_noise": "$c_{12} = 50\,\\mathrm{nS}$",
        "gc100_no_noise": "$c_{12} = 100\,\\mathrm{nS}$",
        "gc200_no_noise": "$c_{12} = 200\,\\mathrm{nS}$",
        "gc50_noisy": "$c_{12} = 50\,\\mathrm{nS}$",
        "gc100_noisy": "$c_{12} = 100\,\\mathrm{nS}$",
        "gc200_noisy": "$c_{12} = 200\,\\mathrm{nS}$",
    }


    def interpolate_in_log_space(xs, ys, zss, ss=10):
        xs, ys, zss = np.log10(xs), np.log10(ys), np.log10(zss)

        xsp = np.linspace(xs[0], xs[-1], len(xs) * ss)
        ysp = np.linspace(ys[0], ys[-1], len(ys) * ss)

        f = scipy.interpolate.interp2d(xs, ys, zss.T, 'linear')
        zssp = f(xsp, ysp).T

        return np.power(10.0, xsp), np.power(10.0, ysp), np.power(10.0, zssp)


    def plot_contour(ax, i, key, cax=None):
        regs = REGS_FLT_SWEEP_MAP[(key, bool(subth))]
        flts = tau_pre_filts
        E = np.median(errs[i, :, :, int(subth), :, 0], axis=-1)
        Elog = np.log10(E)
        vmin, vmax = np.log10(0.03), np.log10(0.4)
        C = ax.contourf(flts,
                        regs,
                        Elog,
                        vmin=vmin,
                        vmax=vmax,
                        levels=np.linspace(vmin, vmax, 21))
        if not cax is None:
    #        cax.pcolormesh(np.power(10.0, C.levels), [0, 1],
    #                       np.array((C.levels, C.levels)),
    #                       shading="nearest",
    #                       vmin=vmin,
    #                       vmax=vmax)
    #        cax.set_xscale("log")
            cax.pcolormesh(C.levels, [0, 1], np.array((C.levels, C.levels)), shading="nearest", vmin=vmin, vmax=vmax)
            cax.set_yticks([])
            cax.spines["left"].set_visible(False)
            utils.outside_ticks(cax)
        ax.contour(flts,
                   regs,
                   Elog,
                   vmin=vmin,
                   vmax=vmax,
                   levels=C.levels,
                   colors=['white'],
                   linestyles=['--'],
                   linewidths=[0.7])
        ax.set_title(param_keys_subtitle_map[key])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Filter time-constant $\\tau$")
        ax.set_ylabel("Regularisation $\\lambda$")

        regsp, fltsp, Ep = interpolate_in_log_space(regs, flts, E)
        iregp, ifltp = np.unravel_index(np.argmin(Ep, axis=None), Ep.shape)
        ax.scatter([fltsp[ifltp]], [regsp[iregp]],
                   marker='+',
                   linewidth=2.5,
                   clip_on=False,
                   zorder=100,
                   s=35,
                   color="white")
        ax.scatter([fltsp[ifltp]], [regsp[iregp]],
                   marker='+',
                   linewidth=1.0,
                   clip_on=False,
                   zorder=100,
                   s=20,
                   color=utils.blues[0])

        ox, oy = 1.5, 2.25
        flipx = 4 * ifltp > 3 * Ep.shape[0]
        flipy = 4 * iregp > 3 * Ep.shape[1]
        utils.annotate(ax,
                       fltsp[ifltp] * (1.0 / 1.2 if flipx else 1.2),
                       regsp[iregp] * (1.0 / 1.2 if flipy else 1.2),
                       fltsp[ifltp] * (1.0 / ox if flipx else ox),
                       regsp[iregp] * (1.0 / oy if flipy else oy),
                       "$E = {:0.2f}\\%$".format(Ep[iregp, ifltp] * 100.0),
                       ha="right" if flipx else "left",
                       va="top" if flipy else "bottom",
                       fontdict={"size": 8},
                       color="white")

        print(key, f"\n\tWith pre-filter: lambda = {regsp[iregp]:0.2e}, tau = {fltsp[ifltp]:0.2e}")

        iregp2 = np.argmin(Ep[:, 0], axis=None)
        ax.scatter([fltsp[0]], [regsp[iregp2]],
                   marker='+',
                   linewidth=2.5,
                   clip_on=False,
                   zorder=100,
                   s=35,
                   color="white")
        ax.scatter([fltsp[0]], [regsp[iregp2]],
                   marker='+',
                   linewidth=1.0,
                   clip_on=False,
                   zorder=100,
                   s=20,
                   color=utils.oranges[1])
        dist = np.sqrt(np.square(iregp2 - iregp) + np.square(ifltp))
        if dist > 10.0:
            flipx = False
            flipy = 4 * iregp2 > 3 * Ep.shape[1]
            ifltp = 0
            iregp = iregp2
            utils.annotate(ax,
                           fltsp[ifltp] * (1.0 / 1.2 if flipx else 1.2),
                           regsp[iregp] * (1.0 / 1.2 if flipy else 1.2),
                           fltsp[ifltp] * (1.0 / ox if flipx else ox),
                           regsp[iregp] * (1.0 / oy if flipy else oy),
                           "$E = {:0.2f}\\%$".format(Ep[iregp, ifltp] * 100.0),
                           ha="right" if flipx else "left",
                           va="top" if flipy else "bottom",
                           fontdict={"size": 8},
                           color="white")

        print(f"\tWithout pre-filter: lambda = {regsp[iregp2]:0.2e}")

        ax.set_xlim(flts[0], flts[-1])
        ax.set_ylim(regs[0], regs[-1])


    #for i, key in enumerate(param_keys):
    #    ax =

    fig = plt.figure(figsize=(6.35, 7.0))

    y0s = np.linspace(0, 1, 4)[1:][::-1]
    y1s = y0s - 0.2

    gss = [
        fig.add_gridspec(1,
                         3,
                         left=0.05 + 0.15,
                         right=0.95 + 0.15,
                         top=y0s[0],
                         bottom=y1s[0],
                         wspace=0.5),
        fig.add_gridspec(1,
                         3,
                         left=0.05,
                         right=0.95,
                         top=y0s[1],
                         bottom=y1s[1],
                         wspace=0.5),
        fig.add_gridspec(1,
                         3,
                         left=0.05,
                         right=0.95,
                         top=y0s[2],
                         bottom=y1s[2],
                         wspace=0.5),
    ]
    cgs = fig.add_gridspec(1, 1, left=0.05, right=0.95, top=0.05, bottom=0.04)
    cax = fig.add_subplot(cgs[0, 0])
    cax.set_xlabel("Network error $\\log_{10}(E_\\mathrm{net})$")

    axs = np.array([
        [fig.add_subplot(gss[0][0, 0]),
         fig.add_subplot(gss[0][0, 1]), None],
        [
            fig.add_subplot(gss[1][0, 0]),
            fig.add_subplot(gss[1][0, 1]),
            fig.add_subplot(gss[1][0, 2])
        ],
        [
            fig.add_subplot(gss[2][0, 0]),
            fig.add_subplot(gss[2][0, 1]),
            fig.add_subplot(gss[2][0, 2])
        ],
    ])

    subth_blurb = "with subthreshold relaxation" if subth else "no subthreshold relaxation"

    axs[0, 1].text(-0.25,
                   1.175,
                   "\\textbf{{Current-based LIF}} ({})".format(subth_blurb),
                   va="bottom",
                   ha="center",
                   transform=axs[0, 1].transAxes)
    axs[1, 1].text(0.5,
                   1.175,
                   "\\textbf{{Two-compartment LIF}} ({}; no noise model)".format(subth_blurb),
                   va="bottom",
                   ha="center",
                   transform=axs[1, 1].transAxes)
    axs[2, 1].text(0.5,
                   1.175,
                   "\\textbf{{Two-compartment LIF}} ({}; with noise model)".format(subth_blurb),
                   va="bottom",
                   ha="center",
                   transform=axs[2, 1].transAxes)

    for i, key in enumerate(param_keys):
        ax = axs[param_keys_idx_map[key]]
        plot_contour(ax, i, key, cax=cax if i == 0 else None)

    utils.save(fig)

