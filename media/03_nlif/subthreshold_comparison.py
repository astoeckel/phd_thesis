import nonneg_common
import lif_utils
import bioneuronqp


def Jerr(J_tar, J):
    J_tar = J_tar.flatten()
    J = J.flatten()

    is_sup = J_tar > 1.0
    is_inv = np.logical_and(J > 1.0, ~is_sup)

    return np.sqrt(
        np.mean(np.square(is_sup * (J_tar - J) + is_inv * (J - 1.0))))


styles = [
    {
        'color': 'k',
        'linestyle': (0, (1, 1)),
        'linewidth': 0.7,
    },
    {
        'color': utils.blues[0],
        'linestyle': '--',
        'linewidth': 1.0,
    },
    {
        'color': utils.oranges[1],
        'linestyle': (0, (1, 1)),
        'linewidth': 1.0,
    },
    {
        'color': mpl.cm.get_cmap('viridis')(0.7),
        'linestyle': '-',
        'linewidth': 1.0,
    },
]


def plot_decoding(ax,
                  Js_tar,
                  p_exc=0.5,
                  N_pre=50,
                  N_smpls=1000,
                  show_ylabels=True,
                  tx=-0.9):
    # Generate the pre-populatin
    rng = np.random.RandomState(1234)
    gains, biases, encoders = nonneg_common.mk_ensemble(N_pre, rng=rng)
    xs = np.linspace(-1, 1, N_smpls).reshape(-1, 1)

    is_exc = rng.choice([True, False], (N_pre, 1), p=[p_exc, 1.0 - p_exc])
    is_inh = ~is_exc
    conn = np.array((is_exc, is_inh), dtype=bool)

    Js_pre = gains[None, :] * (xs @ encoders.T) + biases[None, :]
    As_pre = lif_utils.lif_rate(Js_pre)

    WE1, WI1 = bioneuronqp.solve(As_pre,
                                 Js_tar.reshape(-1, 1),
                                 np.array([0, 1, -1, 1, 0, 0]),
                                 conn,
                                 iTh=None,
                                 renormalise=False,
                                 tol=1e-6,
                                 reg=0.1)
    WE2, WI2 = bioneuronqp.solve(As_pre,
                                 np.clip(Js_tar, 0, None),
                                 np.array([0, 1, -1, 1, 0, 0]),
                                 conn,
                                 iTh=None,
                                 renormalise=False,
                                 tol=1e-6,
                                 reg=0.1)
    WE3, WI3 = bioneuronqp.solve(As_pre,
                                 Js_tar.reshape(-1, 1),
                                 np.array([0, 1, -1, 1, 0, 0]),
                                 conn,
                                 iTh=1.0,
                                 renormalise=False,
                                 tol=1e-6,
                                 reg=0.1)

    print(np.sqrt(np.mean(np.square(WE1) + np.square(WI1))), np.sqrt(np.mean(np.square(WE2) + np.square(WI2))), np.sqrt(np.mean(np.square(WE3) + np.square(WI3))),)

    ax.axhline(1.0, color='k', linestyle='--', linewidth=0.5)
    ax.axhline(0.0, color='k', linestyle='--', linewidth=0.5)

    ax.plot(xs, As_pre @ (WE1 - WI1), **styles[1])
    ax.plot(xs, As_pre @ (WE3 - WI3), **styles[3])
    ax.plot(xs, As_pre @ (WE2 - WI2), **styles[2])

    ax.plot(xs, Js_tar, 'k', **styles[0])

    yt = 2.55
    yw = 0.25

    ax.text(-0.7,
            yt,
            "${:0.2f}\\,\\mathrm{{pA}}$".format(
                Jerr(Js_tar, As_pre @ (WE1 - WI1)) * 1e3),
            va="top",
            ha="center",
            size=8)
    ax.plot([-0.7 - yw, -0.7 + yw], [yt - 0.3, yt - 0.3], **styles[1], clip_on=False)

    ax.text(0.0,
            yt,
            "${:0.2f}\\,\\mathrm{{pA}}$".format(
                Jerr(Js_tar, As_pre @ (WE2 - WI2)) * 1e3),
            va="top",
            ha="center",
            size=8)
    ax.plot([0.0 - yw, 0.0 + yw], [yt - 0.3, yt - 0.3], **styles[2], clip_on=False)

    ax.text(0.7,
            yt,
            "${:0.2f}\\,\\mathrm{{pA}}$".format(
                Jerr(Js_tar, As_pre @ (WE3 - WI3)) * 1e3),
            va="top",
            ha="center",
            size=8)
    ax.plot([0.7 - yw, 0.7 + yw], [yt - 0.3, yt - 0.3], **styles[3], clip_on=False)

    mpl.rcParams['hatch.linewidth'] = 0.5
    ax.fill_between(xs[:, 0],
                    np.ones_like(xs[:, 0]) * 1.0,
                    np.ones_like(xs[:, 0]) * -5.0,
                    linewidth=0.0,
                    color='#f0f0f0')
    ax.fill_between(xs[:, 0],
                    np.ones_like(xs[:, 0]) * 1.0,
                    np.ones_like(xs[:, 0]) * -5.0,
                    hatch='//',
                    edgecolor='#a0a0a0',
                    linewidth=0.0,
                    facecolor='None')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 2)

    ax.set_xticks(np.linspace(-1, 1, 3), minor=False)
    ax.set_xticks(np.linspace(-1, 1, 5), minor=True)

    ax.set_yticks(np.linspace(-1, 2, 4), minor=False)
    ax.set_yticks(np.linspace(-1, 2, 7), minor=True)

    if not show_ylabels:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel('$J_\\mathrm{dec}(\\xi)$ ($\\mathrm{nA}$)')
    ax.set_xlabel('Encoded $\\xi$')


fig, axs = plt.subplots(1,
                        4,
                        figsize=(7.5, 1.5),
                        gridspec_kw={
                            "wspace": 0.2,
                            "width_ratios": [1, 1, 1, 1]
                        })

xs = np.linspace(-1, 1, 1000).reshape(-1, 1)

plot_decoding(axs[0], 1.5 * xs + 0.5, p_exc=0.5, show_ylabels=True)
utils.annotate(axs[0], -0.25, 1.05, -0.5, 1.5, '$J_\\mathrm{th}$')
axs[0].set_title('\\textbf{Mixed pre-population} ($1\!\!:\!\!1$ ratio)',
                 x=1.05,
                 y=1.25)

plot_decoding(axs[1],
              2.5 * np.square(xs) - 0.5,
              p_exc=0.5,
              tx=-0.75,
              show_ylabels=False)

plot_decoding(axs[2], 1.5 * xs + 0.5, p_exc=1.0, show_ylabels=False)
axs[2].set_title('\\textbf{Excitatory pre-population}  ($1\!\!:\!\!0$ ratio)',
                 x=1.05,
                 y=1.25)

plot_decoding(axs[3],
              2.5 * np.square(xs) - 0.5,
              p_exc=1.0,
              show_ylabels=False,
              tx=-0.75)

for i in range(4):
    axs[i].text(#-0.28 if i == 0 else -0.15,
                #1.22125,
                0.05 if i % 2 == 0 else 0.1,
                0.975,
                '\\textbf{{{}}}'.format(chr(ord('A') + i)),
                va='top',
                ha='left',
                transform=axs[i].transAxes,
                bbox=dict(facecolor='white', edgecolor='None', pad=2.0),
                size=12)

fig.legend([
    mpl.lines.Line2D([], [], **styles[0]),
    mpl.lines.Line2D([], [], **styles[1]),
    mpl.lines.Line2D([], [], **styles[2]),
    mpl.lines.Line2D([], [], **styles[3]),
], [
    "Target current", "NNLS", "NNLS with clamped target current",
    "Subthreshold relaxation"
],
           handlelength=1.5,
           loc='upper center',
           ncol=4,
           bbox_to_anchor=(0.5, 1.4))

utils.save(fig)

