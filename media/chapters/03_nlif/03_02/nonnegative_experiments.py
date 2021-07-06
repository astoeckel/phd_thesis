import os
import h5py

with h5py.File(utils.datafile('nonnegative_experiment.h5'), 'r') as f:
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


def plot_individual(ax, cax, errs):
    ax.plot(p_excs,
            np.median(errs[:, 1, 1], axis=-1),
            'k',
            label='Decoded bias')
    ax.fill_between(p_excs,
                    np.percentile(errs[:, 1, 1], 25, axis=-1),
                    np.percentile(errs[:, 1, 1], 75, axis=-1),
                    color='#aaaaaa',
                    linewidth=0)

    ax.plot(p_excs,
            np.median(errs[:, 1, 0], axis=-1),
            '--',
            color='#606060',
            linewidth=0.75,
            label='Intrinsic bias')

    ax.axhline(np.median(errs[0, 0, 1], axis=-1), color='k', linestyle=':', linewidth=0.75)
#    ax.fill_between(
#        p_excs,
#        np.ones_like(p_excs) * np.percentile(errs[0, 0, 1], 25, axis=-1),
#        np.ones_like(p_excs) * np.percentile(errs[0, 0, 1], 75, axis=-1),
#        color=utils.greens[0],
#        alpha=0.25,
#        linewidth=0)

#    ax.axhline(np.median(errs[0, 0, 0], axis=-1),
#               color='k',
#               linestyle='--',
#               linewidth=0.75)

    ax.set_yscale('log')
    ax.set_xlabel('Pre-population excitatory to inhibitory ratio', labelpad=25)
    ax.set_ylabel('NRMSE $E$')
    ax.set_xticks(np.linspace(0, 1, 9))

    y_cen = -0.3
    ax.add_patch(
        mpl.patches.Polygon([(0.0, y_cen), (0.03, y_cen - 0.02),
                             (0.03, y_cen + 0.02)],
                            facecolor='k',
                            transform=ax.transAxes,
                            clip_on=False))
    ax.add_patch(
        mpl.patches.Polygon([(1.0, y_cen), (0.97, y_cen - 0.02),
                             (0.97, y_cen + 0.02)],
                            facecolor='k',
                            transform=ax.transAxes,
                            clip_on=False))
    ax.text(0.05,
            y_cen - 0.005,
            'Purely inhibitory',
            va='center',
            ha='left',
            transform=ax.transAxes)
    ax.text(0.95,
            y_cen - 0.005,
            'Purely excitatory',
            va='center',
            ha='right',
            transform=ax.transAxes)
    ax.xaxis.set_major_formatter(ratio_formatter)
    ax.set_xlim(0, 1)
    ax.spines["bottom"].set_position(('outward', 10.0))
    ax.spines["left"].set_position(('outward', 10.0))


    for p_exc in np.linspace(0, 1, 3):
        cax.add_patch(
            mpl.patches.Ellipse((p_exc, 0.5),
                               0.0375, 2.5,
                               facecolor=mpl.cm.get_cmap('RdBu')(p_exc),
                               transform=cax.transAxes,
                               edgecolor='k',
                               linewidth=0.5,
                               zorder=100,
                               clip_on=False))


    cs = np.linspace(0, 1, 1000).reshape(1, -1)
    cax.imshow(cs, cmap='RdBu', interpolation='bilinear', extent=[0, 1, -0.01, 0.01])
    cax.set_aspect('auto')
    utils.remove_frame(cax)

    ax.legend(ncol=2,
              loc='upper center',
              handlelength=1.0,
              handletextpad=0.5,
              columnspacing=1.0)


fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.25), gridspec_kw={"wspace": 0.4})
cgs = fig.add_gridspec(1,
                       2,
                       left=0.125,
                       right=0.9,
                       top=-0.05,
                       bottom=-0.065,
                       wspace=0.4)
plot_individual(axs[0], fig.add_subplot(cgs[0]), errs[4, :, :, :, 0])
plot_individual(axs[1], fig.add_subplot(cgs[1]), errs[4, :, :, :, 1])

axs[0].set_ylim(1e-2, 1e0)
axs[0].set_title('\\textbf{Identity $\\varphi(x) = x$}')
axs[1].set_ylim(1e-2, 1e0)
axs[1].set_title('\\textbf{Square $\\varphi(x) = 2 x^2 - 1$}')

axs[0].text(-0.25,
            1.02,
            '\\textbf{B}',
            size=12,
            ha='left',
            va='bottom',
            transform=axs[0].transAxes)
axs[1].text(-0.25,
            1.02,
            '\\textbf{C}',
            size=12,
            ha='left',
            va='bottom',
            transform=axs[1].transAxes)

#plot_individual(axs[1, 0], errs[1, :, :, :, 1])
#plot_individual(axs[1, 1], errs[2, :, :, :, 1])
#axs[1, 0].set_ylim(5e-3, 2e0)
#axs[1, 1].set_ylim(5e-1, 1e2)

#ax.set_ylim(5e-3, 2e0)
#axs[0].set_ylim(5e-1, 1e2)

utils.save(fig)

