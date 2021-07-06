import nonneg_common
import lif_utils

fig, axs = plt.subplots(1,
                        3,
                        figsize=(7.4, 1.6),
                        gridspec_kw={
                            "wspace": 0.4,
                        })

Js = np.linspace(0, lif_utils.lif_rate_inv(100), 1001)
As = lif_utils.lif_rate(Js)
axs[0].plot(Js, As, color=utils.blues[0], clip_on=False)
axs[0].set_xlim(0, 3)
axs[0].set_ylim(0, 100)
axs[0].set_xlabel('Current $J$ ($\\mathrm{nA}$)')
axs[0].set_ylabel('Rate $G[J]$ ($\\mathrm{s}^{-1}$)')
axs[0].axvline(1, color='k', linestyle='--', linewidth=0.5)
axs[0].set_title('LIF response curve')
axs[0].text(-0.265,
            1.175,
            '\\textbf{A}',
            va='top',
            ha='left',
            transform=axs[0].transAxes,
            size=12)
axs[0].fill_betweenx(np.linspace(0, 100, 2),
                     np.ones(2) * 0.0,
                     np.ones(2) * 1.0,
                     linewidth=0.0,
                     color='#f0f0f0')
axs[0].fill_betweenx(np.linspace(0, 100, 2),
                     np.ones(2) * 0.0,
                     np.ones(2) * 1.0,
                     linewidth=0.0,
                     color='#f0f0f0',
                     hatch='//',
                     edgecolor='#a0a0a0',
                     facecolor='None')
utils.annotate(axs[0], 1.05, 50, 1.5, 70, '$J_\\mathrm{th}$')

rng = np.random.RandomState(1234)
gains, biases, encoders = nonneg_common.mk_ensemble(5, rng=rng)
xs = np.linspace(-1, 1, 1001).reshape(-1, 1)
Js = gains[None, :] * (xs @ encoders.T) + biases[None, :]
As = lif_utils.lif_rate(Js)
axs[1].plot(xs, As, 'k', linewidth=0.75)
axs[1].set_xlim(-1, 1)
axs[1].set_ylim(0, 100)
axs[1].text(-0.265,
            1.175,
            '\\textbf{B}',
            va='top',
            ha='left',
            transform=axs[1].transAxes,
            size=12)
axs[1].set_ylabel('Rate $G[J]$ ($\\mathrm{s}^{-1}$)')
axs[1].set_xlabel('Represented $x$')
axs[1].set_title('Desired tuning')

axs[2].plot(xs, Js, 'k', linewidth=0.75)
axs[2].set_xlim(-1, 1)
axs[2].set_ylim(-5, 3)
axs[2].text(-0.23,
            1.175,
            '\\textbf{C}',
            va='top',
            ha='left',
            transform=axs[2].transAxes,
            size=12)
axs[2].set_ylabel('Current $J_i(\\xi)$')
axs[2].set_xlabel('Encoded $\\xi$')
axs[2].set_title('Target currents')
axs[2].axhline(1, color='k', linestyle='--', linewidth=0.5)

mpl.rcParams['hatch.linewidth'] = 0.5
axs[2].fill_between(xs[:, 0],
                    np.ones_like(xs[:, 0]) * 1.0,
                    np.ones_like(xs[:, 0]) * -5.0,
                    linewidth=0.0,
                    color='#f0f0f0')
axs[2].fill_between(xs[:, 0],
                    np.ones_like(xs[:, 0]) * 1.0,
                    np.ones_like(xs[:, 0]) * -5.0,
                    hatch='//',
                    edgecolor='#a0a0a0',
                    linewidth=0.0,
                    facecolor='None')

utils.annotate(axs[2], -1.03, 1, -1.375, 2.35, '$J_\\mathrm{th}$')

utils.save(fig)

