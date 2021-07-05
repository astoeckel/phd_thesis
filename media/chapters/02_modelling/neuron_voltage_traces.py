import os


# Loads voltages, spikes and currents
def load_ramp_data(filename):
    data = np.load(os.path.join(utils.datadir, filename))
    ts = data["ts"] * 1e-3  # ms to sec
    us = data["us"] * 1e-3  # mV to V
    Js = data["Js"] * 1e-9  # nA to A
    spikes = data["spikes"] * 1e-3  # ms to sec
    dt = (ts[-1] - ts[0]) / (len(ts) - 1)

    return ts, us, Js, spikes, dt


def do_plot(ax0, ax1, fn1, letter, title, inset=False):
    ts, us, Js, spikes, _ = load_ramp_data(fn1)

    for t in spikes:
        ax0.axvline(t, linestyle='--', linewidth=0.5, color='k')
        ax1.axvline(t, linestyle='--', linewidth=0.5, color='k')

    ax0.axhline(-65.0, color='k', linewidth=0.5, linestyle=':')
    ax0.plot(ts, us * 1e3, '#204a87ff', linewidth=1.0)

    ax0.set_xticklabels([])
    ax0.set_ylim(-90.0, 50.0)
    ax0.set_yticks(np.arange(-85, 55, 20))
    ax0.set_yticks(np.arange(-95, 55, 10), minor=True)
    ax0.set_xticks(np.arange(0.0, 0.5, 0.05), minor=True)
    ax0.set_ylabel('$v$ (mV)')
    ax0.set_xlim(0, 0.5)
    ax0.set_title(title)

    #ax1.plot(ts, Js * 1e9, '#204a87ff')
    ax1.plot(ts, Js * 1e9, 'k')
    ax1.set_ylabel('$J$ (nA)')
    ax1.set_ylim(0, 0.25)
    ax1.set_xlim(0, 0.5)
    ax1.set_xticks(np.arange(0.0, 0.5, 0.05), minor=True)
    ax1.set_xlabel('Time $t$ (s)')

    ax0.text(-0.125,
             1.045,
             '\\textbf{' + letter + '}',
             fontdict={'size': 12},
             va='bottom',
             ha='right',
             transform=ax0.transAxes)

#    if inset:
#        fig = ax0.get_figure()
#        gs = fig.add_gridspec(1, 1, top=0.85, bottom=0.6, left=0.175, right=0.45)
#        ax0.add_patch(mpl.patches.Rectangle((0.15, 0.55), 0.325, 0.3, transform=fig.transFigure, color='white', zorder=10))
#        ax0.add_patch(mpl.patches.Rectangle((0.2, -90), 0.08, 45, transform=ax0.transData, linewidth=3.0, color='white', fill=False, zorder=10, clip_on=False))
#        ax0.add_patch(mpl.patches.Rectangle((0.2, -90), 0.08, 45, transform=ax0.transData, linewidth=1.5, color='#f57900ff', fill=False, zorder=10, clip_on=False))

#        ax = fig.add_subplot(gs[0, 0])
#        ax.axhline(-55, linestyle='--', color='k', linewidth=0.5)
#        ax.axhline(-85, linestyle='--', color='k', linewidth=0.5)
#        ax.annotate('$v_\\mathrm{reset}$', (0.25, -85), xytext=(0.26, -75), arrowprops= {"width": 0.05, "headwidth": 2.0, "headlength": 4.0, "shrink": 0.1, "color": 'k'})
#        ax.plot(ts, us * 1e3, '#204a87ff', linewidth=1.0)
#        ax.set_xlim(0.2, 0.28)
#        ax.spines["left"].set_visible(False)
#        ax.set_yticks([])
#        ax.set_xticklabels([])


#fig = plt.figure(figsize=(7.2, 4.0))
#gs1 = fig.add_gridspec(4, 1, top=0.95, bottom=0.60)
#gs2 = fig.add_gridspec(4, 1, top=0.40, bottom=0.05)
#ax00 = fig.add_subplot(gs1[0:3, 0])
#ax01 = fig.add_subplot(gs1[3, 0])
#ax10 = fig.add_subplot(gs2[0:3, 0])
#ax11 = fig.add_subplot(gs2[3, 0])

fig = plt.figure(figsize=(7.2, 2.95))
gs = fig.add_gridspec(2, 2, wspace=0.3, height_ratios=[4, 1])
ax00 = fig.add_subplot(gs[0, 0])
ax01 = fig.add_subplot(gs[1, 0])
ax10 = fig.add_subplot(gs[0, 1])
ax11 = fig.add_subplot(gs[1, 1])

do_plot(ax00, ax01, 'generated/chapters/02_modelling/90fb5555c0fee2d1_lif_ramp_short.npz', 'A', 'LIF neuron')

do_plot(ax10, ax11, 'generated/chapters/02_modelling/037d43b959417bf8_hodgkin_huxley_ramp_short.npz', 'B',
        'Hodgkin-Huxley neuron')

utils.save(fig)

