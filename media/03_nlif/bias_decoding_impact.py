import lif_utils
import h5py

np.random.seed(154897)


def mk_ensemble(N,
                x_intercepts_cback=None,
                encoders_cback=None,
                d=1,
                rng=np.random):
    def default_x_intercepts(rng, N):
        return rng.uniform(-0.99, 0.99, N)

    def default_encoders_cback(rng, N, d):
        return rng.normal(0, 1, (N, d))

    x_intercepts_cback = (default_x_intercepts if x_intercepts_cback is None
                          else x_intercepts_cback)
    encoders_cback = (default_encoders_cback
                      if encoders_cback is None else encoders_cback)

    max_rates = rng.uniform(50, 100, N)
    x_intercepts = x_intercepts_cback(rng, N)  #rng.uniform(-0.99, 0.99, N)

    J0s = lif_utils.lif_rate_inv(1e-3)
    J1s = lif_utils.lif_rate_inv(max_rates)

    gains = (J1s - J0s) / (1.0 - x_intercepts)
    biases = (J0s - x_intercepts * J1s) / (1.0 - x_intercepts)

    encoders = encoders_cback(rng, N, d)  #rng.normal(0, 1, (N, d))
    encoders /= np.linalg.norm(encoders, axis=1)[:, None]

    if d == 1:
        idcs = np.argsort(x_intercepts * encoders[:, 0])
    else:
        idcs = np.arange(N)

    return gains[idcs], biases[idcs], encoders[idcs]


def do_analysis(axs,
                N_pres,
                errs,
                x_intercepts_cback=None,
                encoders_cback=None):
    N_pre = 1000
    N_smpls = 101

    # Generate the pre-activities
    xs = np.linspace(-1, 1, N_smpls).reshape(-1, 1)
    gains_pre, biases_pre, encoders_pre = mk_ensemble(N_pre,
                                                      x_intercepts_cback,
                                                      encoders_cback)
    Js_pre = gains_pre[None, :] * (xs @ encoders_pre.T) + biases_pre[None, :]
    As_pre = lif_utils.lif_rate(Js_pre)

    axs[0].plot(xs, As_pre[:, ::20], 'k-', linewidth=0.75)
    axs[0].set_xlim(-1, 1)
    axs[0].set_ylim(0, 100)
    axs[0].set_title("\\textbf{Pre tuning-curves}")
    axs[0].set_xlabel("Represented $x$")
    axs[0].set_ylabel("Rate $a_i$ ($\\mathrm{s}^{-1}$)")
    axs[0].text(-0.275,
                1.042,
                '\\textbf{A}',
                size=12,
                va='bottom',
                ha='right',
                transform=axs[0].transAxes)

    colours = [
        utils.blues[0], utils.oranges[1], utils.greens[0], utils.reds[2],
        utils.purples[2]
    ]

    def offs_and_scale(src, tar):
        Y = np.array((np.ones_like(src), src)).T
        offs, scale = np.linalg.lstsq(Y, tar, rcond=None)[0]
        return offs + scale * src

    U, S, V = np.linalg.svd(As_pre)  # - np.mean(As_pre, axis=0))
    N_svd = 4
    for i in range(N_svd):
        c = colours[i]
        axs[2].plot(xs,
                    U[:, i],
                    zorder=100 - i,
                    color=c,
                    label='$i = {}$'.format(i + 1),
                    linewidth=1.5)
        P = np.polynomial.Legendre([0] * i + [1])(xs[:, 0])
        axs[2].plot(xs, offs_and_scale(P, U[:, i]), 'k:')


#        axs[1].bar(i + 1,
#                   S[i] / np.sum(S),
#                   color=c,
#                   linewidth=0.5,
#                   edgecolor='k')
    axs[2].set_xlabel("Represented $x$")
    axs[2].set_ylabel("Basis functin $f_i(x)$", labelpad=2)
    axs[2].set_title("\\textbf{PCA}")
    axs[2].set_ylim(-0.3, 0.3)
    axs[2].set_xlim(-1, 1)
    axs[2].text(-0.275,
                1.042,
                '\\textbf{C}',
                size=12,
                va='bottom',
                ha='right',
                transform=axs[2].transAxes)

    axs[2].legend(loc='upper left',
                  ncol=2,
                  fontsize=8,
                  columnspacing=1.0,
                  handlelength=0.75,
                  handletextpad=0.5,
                  borderaxespad=0.0,
                  labelspacing=0.25,
                  bbox_to_anchor=(0.05, 1.05))

    #    axs[1].set_xticks(np.arange(N_svd) + 1)
    #    axs[1].set_ylim(0, 0.6)
    #    axs[1].set_xlabel("PC index $i$")
    #    axs[1].set_title("Singular values")

    err_dec_xs, err_dec_ones, err_As_bias, err_As_dec_bias = errs

    def plot_median_with_percentile(ax, xs, ys, color, **kwargs):
        ax.loglog(xs, np.median(ys, axis=-1), '-+', color=color, **kwargs)
        ax.fill_between(xs,
                        np.percentile(ys, 25, axis=-1),
                        np.percentile(ys, 75, axis=-1),
                        color=color,
                        alpha=0.5,
                        linewidth=0)

    plot_median_with_percentile(axs[1],
                                N_pres,
                                err_dec_xs,
                                color=utils.blues[0],
                                label='Identity $\\mathbf{D}$')
    plot_median_with_percentile(axs[1],
                                N_pres,
                                err_dec_ones,
                                color=utils.oranges[1],
                                label='Constant $\\mathbf{D}^1$')

    print(
        "median increase in decoding error between identity and constant decoder: ",
        (np.mean(err_dec_ones / err_dec_xs) - 1.0) * 100.0,
        (np.std(err_dec_ones / err_dec_xs)) * 100.0)

    axs[1].set_ylim(1e-3, 1e-1)
    axs[1].set_title('\\textbf{Decoding error}')
    axs[1].set_xlabel('Pre-neurons $n$')
    axs[1].set_ylabel('RMSE (a.u.)', labelpad=2)
    axs[1].legend(loc='upper right',
                  fontsize=8,
                  handlelength=1.0,
                  handletextpad=0.5,
                  borderaxespad=0.0,
                  labelspacing=0.25,
                  bbox_to_anchor=(1.0, 1.05))
    axs[1].text(-0.275,
                1.042,
                '\\textbf{B}',
                size=12,
                va='bottom',
                ha='right',
                transform=axs[1].transAxes)

    plot_median_with_percentile(axs[3],
                                N_pres,
                                err_As_bias,
                                color=utils.blues[0],
                                label='Intrinsic bias')
    plot_median_with_percentile(axs[3],
                                N_pres,
                                err_As_dec_bias,
                                color=utils.oranges[1],
                                label='Decoded bias')
    axs[3].set_ylim(1e-1, 1e1)
    axs[3].legend(loc='upper right',
                  fontsize=8,
                  handlelength=1.0,
                  handletextpad=0.5,
                  borderaxespad=0.0,
                  labelspacing=0.25,
                  bbox_to_anchor=(1.0, 1.05))
    axs[3].set_xlabel("Pre-neurons $n$")
    axs[3].set_ylabel("RMSE ($s^{-1}$)", labelpad=2)
    axs[3].set_title("\\textbf{Post-tuning error}")
    axs[3].text(-0.275,
                1.042,
                '\\textbf{D}',
                size=12,
                va='bottom',
                ha='right',
                transform=axs[3].transAxes)

fig, axs = plt.subplots(1,
                        4,
                        figsize=(7.45, 1.5),
                        gridspec_kw={
                            "wspace": 0.5,
                            "hspace": 0.9,
                            "width_ratios": [1, 1.1, 1, 1]
                        })

with h5py.File(utils.datafile('bias_decoding_impact.h5'), 'r') as f:
    N_pres = f['N_pres'][()]
    errs = f['errs'][()]

do_analysis(axs, N_pres, errs)
#do_analysis(axs[2], encoders_cback=lambda rng, N, d: np.ones((N, d)))

#fig.text(0.5,
#         0.95,
#         "\\textbf{Balanced encoders}",
#         va="bottom",
#         ha="center",
#         transform=fig.transFigure)
#fig.text(0.0775,
#         0.95,
#         "\\textbf{A}",
#         va="bottom",
#         ha="center",
#         size=12,
#         transform=fig.transFigure)

#fig.text(0.5,
#         0.44,
#         "\\textbf{Biased encoders}",
#         va="bottom",
#         ha="center",
#         transform=fig.transFigure)
#fig.text(0.0775,
#         0.44,
#         "\\textbf{B}",
#         va="bottom",
#         ha="center",
#         size=12,
#         transform=fig.transFigure)

utils.save(fig)

