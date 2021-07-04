import os
import h5py

with h5py.File(os.path.join('..', '..', 'data', 'nonnegative_sparsity3.h5'),
               'r') as f:
    p_dropouts = f['p_dropouts'][()]
    p_excs = f['p_excs'][()]
    p_reg_l1s = f['p_reg_l1s'][()]
    errs = f['errs'][()]

sparsity_median = np.nanmedian(errs[0], axis=-1)
errs_median = np.nanmedian(errs[1], axis=-1)

N_smpls = errs.shape[2]


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


@mpl.ticker.FuncFormatter
def ratio_formatter(x, pos):
    p1 = x
    p2 = (1.0 - x)
    base = 100
    a1 = int(np.round(p1 * base / 5)) * 5
    a2 = int(np.round(p2 * base / 5)) * 5
    d = gcd(a1, a2)
    b1 = a1 // d
    b2 = a2 // d
    return f"${b1}\\!\\!:\\!\\!{b2}$"


fig, axsp = plt.subplots(2,
                         3,
                         figsize=(7.3, 2.5),
                         gridspec_kw={
                             "wspace": 0.3,
                             "hspace": 0.75,
                             "height_ratios": [10.0, 0.2],
                         })

axs = axsp[0]
caxs = axsp[1]

modes = [
    "NNLS", "Enforced sparsity", "Elastic net"
]

fi = 0

for i, mode_idx in enumerate([1, 2, 0]):
    mode_styles = [
        {
            'marker': 'o',
            's': 15,
        },
        {
            'marker': 'd',
            's': 15,
        },
        {
            'marker': 's',
            's': 15,
        },
    ]
    mode_cmap = ["RdBu", 'viridis', 'plasma']

    for j, do_bias in enumerate([1]):
        xs = sparsity_median[mode_idx, :, fi, do_bias] * 100
        ys = errs_median[mode_idx, :, fi, do_bias]

        styles = [
            {
                'linestyle': '-',
                'color': 'k',
                'linewidth': 0.70,
            },
            {
                'linestyle': '--',
                'color': '#606060',
                'linewidth': 0.5
            },
        ]

#        for k in range(3):
#            if (i != k) and (j == 0):
#                axs[k].plot(xs,
#                            ys,
#                            linewidth=0.5,
#                            color='#606060',
#                            linestyle=':',
#                            zorder=-1)

        axs[i].plot(xs, ys, **styles[j], clip_on=False)

        for k in np.linspace(0, N_smpls - 1, 9, dtype=int):
            if mode_idx != 2:
                P = p_excs[k]
            else:
                P = p_reg_l1s[k]
            axs[i].scatter(
                [xs[k]], [ys[k]], **{
                    **dict(color=mpl.cm.get_cmap(mode_cmap[mode_idx])(k / (N_smpls - 1)),
                           linewidth=styles[j]['linewidth'],
                           linestyle=styles[j]['linestyle'],
                           edgecolor=styles[j]['color'],
                           zorder=100 - j,
                           clip_on=False),
                    **mode_styles[mode_idx]
                })
            caxs[i].scatter(
                [P], [0.0], **{
                    **dict(color='white',
                           linewidth=styles[j]['linewidth'],
                           linestyle=styles[j]['linestyle'],
                           edgecolor='white',
                           clip_on=False),
                    **mode_styles[mode_idx],
                    **{
                        "s": 1.75 * mode_styles[mode_idx]['s'],
                        "zorder": 1,
                    },
                })
            caxs[i].scatter(
                [P], [0.0], **{
                    **dict(color=mpl.cm.get_cmap(mode_cmap[mode_idx])(k / (N_smpls - 1)),
                           linewidth=styles[j]['linewidth'],
                           linestyle=styles[j]['linestyle'],
                           edgecolor=styles[j]['color'],
                           clip_on=False),
                    **mode_styles[mode_idx],
                    **{
                        "s": 1.0 * mode_styles[mode_idx]['s'],
                        "zorder": 1,
                    },
                })

    axs[i].set_xlim(0, 100)
    axs[i].set_ylim(1e-2, 1e0)
    axs[i].set_yscale('log')
    #    axs[i].set_title("\\textbf{" + modes[mode_idx] + "}", y=1.0675)
    axs[i].set_title("\\textbf{" + modes[mode_idx] + "}", y=1.0675)

    axs[i].set_xticks(np.linspace(0, 100, 5))
    axs[i].set_xticklabels(
        ["${}\\%$".format(int(p)) for p in np.linspace(0, 100, 5)])

    if i == 0:
        axs[i].set_ylabel("NRMSE $E$")
#    else:
#        axs[i].set_yticklabels([])
    if mode_idx == 0:
        axs[i].set_xlabel("Sparsity (fraction of zeros in $\\mat{W}^+$, $\\mat{W}^-$)")
    else:
        axs[i].set_xlabel("Sparsity (fraction of zeros in $\\mat{W}$)")

    axs[i].text(-0.27 if i == 0 else -0.155,
                1.1,
                '\\textbf{{{}}}'.format(chr(ord('A') + i)),
                va='bottom',
                ha='left',
                size=12,
                transform=axs[i].transAxes)

    ys = np.linspace(-1, 1, 10)
    if mode_idx != 2:
        xs = np.linspace(0, 1, 100).reshape(1, -1)
        xss, yss = np.meshgrid(xs, ys)
        zss = xss
    else:
        xs = np.logspace(1, 3, 100)
        xss, yss = np.meshgrid(xs, ys)
        zss = (np.log10(xss) - 1.0) / 2.0
        caxs[i].set_xscale('log')
    utils.remove_frame(caxs[i])
    pcol = caxs[i].pcolormesh(xs,
                              ys,
                              zss,
                              cmap=mpl.cm.get_cmap(mode_cmap[mode_idx]),
                              linewidth=0.1,
                              zorder=-1,
                              shading='auto')
    pcol.set_edgecolor('face')
    caxs[i].set_aspect('auto')
    caxs[i].spines['bottom'].set_visible(True)
    caxs[i].spines['bottom'].set_position(('outward', 10.0))
    caxs[i].set_ylim(-0.1, 0.1)

    if mode_idx == 0:
        caxs[i].set_xticks(np.linspace(0, 1, 5))
        caxs[i].set_xticks(np.linspace(0, 1, 9), minor=True)
        caxs[i].xaxis.set_major_formatter(ratio_formatter)
        caxs[i].set_xlabel("Excitatory to inhibitory ratio $n^+ \!\! : \!\! n^-$")
    elif mode_idx == 1:
        caxs[i].set_xticks(np.linspace(0, 1, 5))
        caxs[i].set_xticks(np.linspace(0, 1, 9), minor=True)
        caxs[i].set_xticklabels(
            ["${}\\%$".format(int(p)) for p in np.linspace(0, 100, 5)])
        caxs[i].set_xlabel("Threshold percentile $P$")
    elif mode_idx == 2:
        caxs[i].set_xticks([1e1, 1e2, 1e3])
        caxs[i].set_xlabel("$L_1$ regularisation factor $\\sqrt{\\lambda_1}$")

utils.save(fig)

