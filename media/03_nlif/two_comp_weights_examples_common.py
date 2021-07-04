import lif_utils
from nonneg_common import mk_ensemble
import bioneuronqp
import scipy.optimize

def forkrng(rng=np.random):
    return np.random.RandomState(rng.randint((1 << 31)))


def single_experiment(ax1,
                      ax2,
                      ax3,
                      cax2,
                      cax3,
                      N_pre=201,
                      N_smpls=40,
                      N_res=201,
                      f=lambda x, y: 0.5 * (x + y),
                      sigma=0.1,
                      w=[0, 1, -1, 1, 0, 0],
                      gmax=200,
                      gscale=1.0,
                      gylabel=None,
                      gtitle=None,
                      gletter="g",
                      vabs=0.2,
                      letter='A',
                      plot_hyper=False,
                      rng=np.random):

    # Create two pre-populations
    gains_pre1, biases_pre1, encoders_pre1 = mk_ensemble(N_pre,
                                                         rng=forkrng(rng))
    gains_pre2, biases_pre2, encoders_pre2 = mk_ensemble(N_pre,
                                                         rng=forkrng(rng))

    # Create the post-neuron. Set the x-intercept to zero.
    gains_post, biases_post, encoders_post = mk_ensemble(
        1,
        rng=forkrng(rng),
        x_intercepts_cback=lambda _, N: np.zeros(N),
        encoders_cback=lambda _, N, d: np.ones((N, d)),
        max_rates=(100, 100))

    # Generate pre-population samples. We implicitly map the range [-1, 1] onto
    # [0, 1]
    xs, ys = np.linspace(-1, 1, N_smpls), np.linspace(-1, 1, N_smpls)
    xss, yss = np.meshgrid(xs, ys)
    smpls = np.array((xss.flatten(), yss.flatten())).T

    # Evaluate the pre-populations at those points
    Js_pre1 = gains_pre1[None, :] * (
        smpls[:, 0].reshape(-1, 1) @ encoders_pre1.T) + biases_pre1[None, :]
    Js_pre2 = gains_pre2[None, :] * (
        smpls[:, 1].reshape(-1, 1) @ encoders_pre2.T) + biases_pre2[None, :]

    # Evaluate the post-neuron over the range [0, 1]
    xs, ys = np.linspace(0, 1, N_smpls), np.linspace(0, 1, N_smpls)
    xss, yss = np.meshgrid(xs, ys)
    zss = f(xss, yss)
    Js_post = gains_post[None, :] * (
        zss.reshape(-1, 1) @ encoders_post.T) + biases_post[None, :]

    As_pre1 = lif_utils.lif_rate(Js_pre1)
    As_pre2 = lif_utils.lif_rate(Js_pre2)

    As_pre = np.concatenate((As_pre1, As_pre2), axis=1)

    # Solve for weights
    WE, WI = bioneuronqp.solve(As_pre,
                               Js_post,
                               np.array(w),
                               None,
                               iTh=1.0,
                               renormalise=True,
                               reg=10.0,
                               n_threads=1,
                               progress_callback=None,
                               warning_callback=None)

    def H(gE, gI):
        b0, b1, b2, a0, a1, a2 = w
        return (b0 + b1 * gE + b2 * gI) / (a0 + a1 * gE + a2 * gI)

    def dec(J):
        return (J - biases_post[0]) / gains_post[0]

    # Now evaluate the pre-populations at higher resolutions in 1D
    xs, ys = np.linspace(-1, 1, N_res), np.linspace(-1, 1, N_res)

    Js_pre1 = gains_pre1[None, :] * (
        xs.reshape(-1, 1) @ encoders_pre1.T) + biases_pre1[None, :]
    Js_pre2 = gains_pre2[None, :] * (
        ys.reshape(-1, 1) @ encoders_pre2.T) + biases_pre2[None, :]

    As_pre1 = lif_utils.lif_rate(Js_pre1)
    As_pre2 = lif_utils.lif_rate(Js_pre2)

    gE1 = As_pre1 @ WE[:N_pre]
    gE2 = As_pre2 @ WE[N_pre:]
    gE = 0.5 * (gE1 + gE2)  # Functions should be symmetric

    gI1 = As_pre1 @ WI[:N_pre]
    gI2 = As_pre2 @ WI[N_pre:]
    gI = 0.5 * (gI1 + gI2)  # Functions should be symmetric

    # Map onto [0, 1]
    xs, ys = np.linspace(0, 1, N_res), np.linspace(0, 1, N_res)
    zss = f(xs[:, None], ys[None, :])
    zss_dec = dec(H(gE + gE.T, gI + gI.T))


    ax1.plot(xs,
             gE * gscale,
             color=utils.blues[0],
             clip_on=False,
             label="${{{}}}_\\mathrm{{E}}$".format(gletter))
    ax1.plot(xs,
             gI * gscale,
             color=utils.reds[0],
             clip_on=False,
             label="${{{}}}_\\mathrm{{I}}$".format(gletter))

    if plot_hyper:
        def hyperbola(xs, alpha, beta):
            return beta / (1.0 + alpha * xs)

        alpha, beta = 21.0, 1300.0
        ax1.plot(xs, hyperbola(xs, alpha, beta), 'k:')

    ax1.set_xlabel("Represented value $x_1$ / $x_2$", labelpad=2)
    if not gylabel is None:
        ax1.set_ylabel(gylabel)
    leg = ax1.legend(loc="upper left",
               bbox_to_anchor=(0.02, 1.2),
               ncol=2,
               handletextpad=0.5,
               columnspacing=1.0,
               handlelength=1.0,
               facecolor='white',
               frameon=True,
               fancybox=False,
               borderpad=0.05,
               framealpha=1)
    leg.get_frame().set_linewidth(0.0)

    if not gtitle is None:
        ax1.set_title(gtitle)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, gmax)
    ax1.set_yticks(np.linspace(0, gmax, 3))
    ax1.set_yticks(np.linspace(0, gmax, 5), minor=True)

    C = ax2.contourf(xs,
                     ys,
                     dec(H(gE + gE.T, gI + gI.T)),
                     levels=np.linspace(-0.3, 1.0, 14))
    ax2.contour(xs,
                ys,
                zss_dec,
                levels=C.levels,
                colors=['k'],
                linestyles=['-'],
                linewidths=[0.7])
    ax2.contour(xs,
                ys,
                zss,
                levels=C.levels,
                colors=['white'],
                linestyles=['--'],
                linewidths=[1.0])
    ax2.set_xlabel("Represented value $x_1$", labelpad=2)
    ax2.set_ylabel("Represented value $x_2$")
    ax2.set_aspect(1)
    ax2.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax2.set_title("Decoded $\\hat y$")

    cb2 = None
    if not cax2 is None:
        cb2 = plt.colorbar(C, cax=cax2, orientation='horizontal')
        cb2.outline.set_visible(False)
        cb2.set_ticks(np.arange(-0.3, 1, 0.3))
        cax2.spines["bottom"].set_visible(True)
        utils.outside_ticks(cax2)

    E = dec(H(gE + gE.T, gI + gI.T)) - zss
    C = ax3.imshow(E,
                   extent=[0, 1, 0, 1],
                   vmin=-vabs,
                   vmax=vabs,
                   cmap='RdBu',
                   origin='lower')
    ax3.set_xlabel("Represented value $x_1$", labelpad=2)
    #ax3.set_ylabel("Represented value $x_2$")
    ax3.set_yticklabels([])
    ax3.set_aspect(1)
    ax3.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax3.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax3.set_title("Error $\\hat y - y$")

    cb3 = None
    if not cax3 is None:
        cb3 = plt.colorbar(C, cax=cax3, orientation='horizontal')
        cb3.outline.set_visible(False)
        cax3.spines["bottom"].set_visible(True)
        utils.outside_ticks(cax3)

    def nrmse(x, x_ref):
        return np.sqrt(np.mean(np.square(x - x_ref))) / np.sqrt(
            np.mean(np.square(x_ref)))

    ax3.text(0.05,
             0.95,
             "$E = {:0.1f}\%$".format(
                 nrmse(zss_dec, zss) * 1e2),
             va="top",
             ha="left",
             bbox={
                 "pad": 1.0,
                 "color": "w",
                 "linewidth": 0.0,
             },
             transform=ax3.transAxes,
             size=8)

    utils.outside_ticks(ax2)
    utils.outside_ticks(ax3)

    fig = ax1.get_figure()
    fig.align_labels([ax1, ax2])
    fig.align_labels([ax2, ax3])

    return cb2, cb3


def do_plot(f, gmax=200, vabs=0.006, tall=False, plot_hyper=False):
    fig = plt.figure(figsize=(6.3, 6.25 if tall else 5.7))

    h = 0.36
    ygap = 0.6
    gs, cgs = [], []
#    for top, bottom in [(0.95, 0.95 - h),
#                    (0.95 - (1 + ygap) * h, 0.95 - (2 + ygap) * h)]:
    for top, bottom in [(0.95, 0.95 - h)]:
        for left, right in [(0.05, 0.45), (0.55, 0.95)]:
            gs.append(
                fig.add_gridspec(2,
                                 2,
                                 wspace=0.24,
                                 hspace=0.725,
                                 left=left,
                                 right=right,
                                 top=top,
                                 bottom=bottom,
                                 height_ratios=[0.5 if tall else 0.35, 1]))
            cgs.append(
                fig.add_gridspec(1,
                                 2,
                                 wspace=0.24,
                                 left=left,
                                 right=right,
                                 top=bottom - 0.07,
                                 bottom=bottom - 0.08))

    gs1, gs2 = gs
    cgs1, cgs2 = cgs

    ax1 = fig.add_subplot(gs1[0, :])
    ax2 = fig.add_subplot(gs1[1, 0])
    ax3 = fig.add_subplot(gs1[1, 1])
    cax1 = fig.add_subplot(cgs1[0, 0])
    cax2 = fig.add_subplot(cgs1[0, 1])

    cb2, cb3 = single_experiment(
        ax1,
        ax2,
        ax3,
        cax1,
        cax2,
        f=f,
        rng=np.random.RandomState(37901),
        w=[0, 1, -1, 1, 0, 0],
        gmax=2,
        gscale=1.0,
        gylabel="Cur. ($\\mathrm{nA}$)",
        gtitle="Decoded current function",
        gletter="J",
        vabs=vabs,
    )
    ax1.text(0.5,
             1.575,
             "\\textbf{Current-based LIF neuron}",
             ha="center",
             va="bottom",
             transform=ax1.transAxes)
    ax1.text(-0.18,
             1.575,
             "\\textbf{A}",
             size=12,
             ha="left",
             va="bottom",
             transform=ax1.transAxes)

    ax1 = fig.add_subplot(gs2[0, :])
    ax2 = fig.add_subplot(gs2[1, 0])
    ax3 = fig.add_subplot(gs2[1, 1])
    cax1 = fig.add_subplot(cgs2[0, 0])
    cax2 = fig.add_subplot(cgs2[0, 1])

    single_experiment(
        ax1,
        ax2,
        ax3,
        cax1,
        cax2,
        f=f,
        rng=np.random.RandomState(37901),
        w=[-19.5e-6, 1000.0, -425.5, 9.0e6, 296.4, 132.2],
        gmax=gmax,
        gscale=1e-3,
        gylabel="Cond. ($\\mathrm{nS}$)",
        gtitle="Decoded conductance function",
        gletter="g",
        vabs=vabs,
        plot_hyper=plot_hyper,
    )
    ax2.text(0.5,
             1.575,
             "\\textbf{Two-compartment LIF neuron}",
             ha="center",
             va="bottom",
             transform=ax1.transAxes)
    ax2.text(-0.18,
             1.575,
             "\\textbf{B}",
             size=12,
             ha="left",
             va="bottom",
             transform=ax1.transAxes)

    utils.save(fig)

