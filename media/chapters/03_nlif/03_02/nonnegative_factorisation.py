import os
import h5py

with h5py.File(utils.datafile('nonnegative_factorisation.h5'), 'r') as f:
    p_excs = f['p_excs'][()]
    errs = f['errs'][()]


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


fig, axs = plt.subplots(1,
                        2,
                        figsize=(7.4, 1.75),
                        gridspec_kw={
                            "wspace": 0.2,
                            "width_ratios": [4, 9]
                        })

cgs = fig.add_gridspec(1,
                       1,
                       left=0.55,
                       right=0.9,
                       top=1.2,
                       bottom=1.1775,
                       wspace=0.2)
cax = fig.add_subplot(cgs[0])

for i, do_bias in enumerate([False, True]):
    ds = [
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5, 6, 7, 10, 20],
    ][i]
    for j in range(2):
        for l, d in enumerate(ds):
            xs = np.linspace(-0.3, 0.3, errs.shape[1]) + (l + 1)
            ys = np.nanmedian(errs[d - 1, :, i, j], axis=-1)
            styles = [
                {
                    'linestyle': '-',
                    'color': 'k',
                    'linewidth': 0.70,
                },
                {
                    'linestyle': ':',
                    'color': 'k',
                    'linewidth': 0.5
                },
            ]
            axs[i].plot(xs, ys, **styles[j])
            for k in np.linspace(0, len(p_excs) - 1, 5, dtype=int):
                axs[i].scatter([xs[k]], [ys[k]],
                               marker='o',
                               color=mpl.cm.get_cmap('RdBu')(p_excs[k]),
                               linewidth=styles[j]['linewidth'],
                               edgecolor='k',
                               linestyle=styles[j]['linestyle'],
                               s=10,
                               zorder=100 - j)
            if l > 0 and ds[l] - ds[l - 1] > 1:
                axs[i].scatter([l + 0.5], [1e-2],
                               marker='s',
                               color='white',
                               clip_on=False,
                               zorder=100,
                               s=9)
                axs[i].scatter([l + 0.55], [1e-2],
                               marker='$/$',
                               color='k',
                               clip_on=False,
                               zorder=101)
                axs[i].scatter([l + 0.45], [1e-2],
                               marker='$/$',
                               color='k',
                               clip_on=False,
                               zorder=101)
    axs[i].set_yscale('log')
    axs[i].set_ylim(1e-2, 1.5e0)

    axs[i].set_xticks(np.arange(len(ds)) + 1)
    axs[i].set_xticklabels(ds)
    axs[i].set_xlabel('Factorised weight matrix rank $k$', labelpad=1.0)
    axs[i].set_ylabel('NRMSE $E$')
    axs[i].set_title(
        ['\\textbf{Intrinsic biases}', '\\textbf{Decoded biases}'][i])
    axs[i].text(-0.27 if i == 0 else -0.12,
                1.034,
                '\\textbf{{{}}}'.format(chr(ord('A') + i)),
                va='bottom',
                ha='left',
                size=12,
                transform=axs[i].transAxes)

cs = np.linspace(0, 1, 1000).reshape(1, -1)
cax.imshow(cs,
           cmap='RdBu',
           interpolation='bilinear',
           extent=[0, 1, -0.01, 0.01])
cax.set_aspect('auto')
utils.remove_frame(cax)
cax.set_xlabel('Excitatory to inhibitory ratio', labelpad=-27.5)

for p_exc in np.linspace(0, 1, 5):
    cax.add_patch(
        mpl.patches.Ellipse((p_exc, 0.5),
                            0.0375 * 0.75,
                            2.5 * 0.75,
                            facecolor=mpl.cm.get_cmap('RdBu')(p_exc),
                            transform=cax.transAxes,
                            edgecolor='k',
                            linewidth=0.7,
                            zorder=100,
                            clip_on=False))
cax.spines['bottom'].set_visible(True)
cax.set_xticks(np.linspace(0, 1, 9))
cax.xaxis.set_major_formatter(ratio_formatter)

fig.legend(
    [
        mpl.lines.Line2D([], [], color='k', linewidth=0.7),
        mpl.lines.Line2D([], [], color='k', linestyle=':', linewidth=0.5),
    ],
    [
        'Identity $\\varphi(x) = x$',
        'Square $\\varphi(x) = 2 x^2 - 1$',
    ],
    ncol=2,
    loc='upper left',
    bbox_to_anchor=(0.075, 1.28),
)

utils.save(fig)

