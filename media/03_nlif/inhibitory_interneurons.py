import os
from nonneg_common import *

with h5py.File(os.path.join('..', '..', 'data', 'inhibitory_interneurons.h5'),
               'r') as f:
    xss = f['xss'][()]
    yss = f['yss'][()]
    ys_decs = f['ys_decs'][()]
    nrmses = f['nrmses'][()]

fig, axs = plt.subplots(1,
                        5,
                        figsize=(7.55, 1.75),
                        gridspec_kw={"width_ratios": [1, 1, 0.25, 1, 1]})
utils.remove_frame(axs[2])

xs = xss[0, 0, 0]
ys_f1 = yss[0, 0, 0]
ys_dec_f1 = ys_decs[0, 0]

ys_f2 = yss[1, 0, 0]
ys_dec_f2 = ys_decs[1, 0]


def plot_fn(ax, xs, ys, ys_decs, nrmses, P=10, ylabels=True):
    ax.plot(xs, ys, 'k-', linewidth=0.5)

    ax.plot(xs, np.median(ys_decs, axis=0), 'k', clip_on=False)
    ax.fill_between(xs,
                    np.percentile(ys_decs, P, axis=0),
                    np.percentile(ys_decs, 100 - P, axis=0),
                    color='#c0c0c0',
                    clip_on=False)

    ax.plot(xs, ys, ':', color='white', linewidth=0.6, zorder=10)

    ax.text(-0.8,
            0.9,
            "$E = {:0.2f} \\pm {:0.2f}$".format(np.mean(nrmses), np.std(nrmses)),
            va="top",
            ha="left")

    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_xticks([-1, -0.5, 0, 0.5, 1], minor=True)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1], minor=True)

    ax.set_xlabel('Represented $x$')
    if ylabels:
        ax.set_ylabel('Decoded $\\varphi(x)$')
    else:
        ax.set_yticklabels([])
    ax.set_aspect(1)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)


plot_fn(axs[0], xss[0, 0, 0], yss[0, 0, 0], ys_decs[0, 0], nrmses[0, 0])
plot_fn(axs[1],
        xss[1, 0, 0],
        yss[1, 0, 0],
        ys_decs[1, 0],
        nrmses[1, 0],
        ylabels=False)

plot_fn(axs[3], xss[0, 1, 0], yss[0, 1, 0], ys_decs[0, 1], nrmses[0, 1])
plot_fn(axs[4],
        xss[1, 1, 0],
        yss[1, 1, 0],
        ys_decs[1, 1],
        nrmses[1, 1],
        ylabels=False)

axs[0].text(-0.3,
            1.05,
            '\\textbf{B}',
            va='bottom',
            ha='left',
            size=12,
            transform=axs[0].transAxes)
axs[3].text(-0.3,
            1.05,
            '\\textbf{B}',
            va='bottom',
            ha='left',
            size=12,
            transform=axs[3].transAxes)
axs[0].set_title('$\\varphi(x) = x$')
axs[1].set_title('$\\varphi(x) = 2x^2 - 1$')
axs[3].set_title('$\\varphi(x) = x$')
axs[4].set_title('$\\varphi(x) = 2x^2 - 1$')

utils.save(fig)

