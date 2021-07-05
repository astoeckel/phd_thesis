def second_order_filter(t_spikes=[(0e-3, 1.0)],
                        T=0.2,
                        tau0=20e-3,
                        tau1=20e-3,
                        dt=1e-4,
                        normalise=True):
    t_spikes = sorted(t_spikes)
    ts = np.arange(0, T, dt)
    hs = np.zeros_like(ts)
    gs = np.zeros_like(ts)
    for i in range(1, len(ts)):
        while len(t_spikes) > 0 and t_spikes[0][0] <= ts[i - 1]:
            hs[i - 1] += t_spikes[0][1]
            t_spikes.pop(0)
        hs[i] = hs[i - 1] - dt * (hs[i - 1] / tau0)
        gs[i] = gs[i - 1] - dt * (gs[i - 1] / tau1 - hs[i - 1])

    return ts, gs / ((np.sum(gs) * dt) if normalise else 1.0)


def first_order_filter(t_spikes=[(0e-3, 1.0)],
                       T=0.2,
                       tau0=20e-3,
                       dt=1e-4,
                       normalise=True):
    t_spikes = sorted(t_spikes)
    ts = np.arange(0, T, dt)
    gs = np.zeros_like(ts)
    for i in range(1, len(ts)):
        while len(t_spikes) > 0 and t_spikes[0][0] <= ts[i - 1]:
            gs[i - 1] += t_spikes[0][1]
            t_spikes.pop(0)
        gs[i] = gs[i - 1] - dt * gs[i - 1] / tau0

    return ts, gs / ((np.sum(gs) * dt) if normalise else 1.0)


fig = plt.figure(figsize=(6.25, 3.0))
gs1 = fig.add_gridspec(nrows=5,
                       ncols=1,
                       left=0.05,
                       right=0.23,
                       height_ratios=[3, 1, 1, 1, 1],
                       hspace=0.1)
gs2 = fig.add_gridspec(nrows=4,
                       ncols=1,
                       left=0.32,
                       right=0.95,
                       height_ratios=[1, 1, 2, 2],
                       hspace=0.5)

ax1 = fig.add_subplot(gs1[0])

ax1.plot([1, 24], [1, 24], '--k')
#ax1.set_aspect(1)
ax1.set_xlabel("$\\tau_1$ (ms)", labelpad=0.0)
ax1.set_ylabel("$\\tau_2$ (ms)")
ax1.set_xlim(1, 29)
ax1.set_ylim(1, 29)
ax1.set_xticks([5, 15, 25])
ax1.set_yticks([5, 15, 25])

trials = [
    (5, 5),
    (15, 15),
    (25, 25),
    (5, 15),
    (15, 25),
    (5, 25),
]
cs = [
    utils.blues[0], utils.blues[1], utils.blues[2], utils.oranges[0],
    utils.oranges[1], utils.reds[0]
]
for i, (tau1, tau2) in enumerate(trials):
    ax1.plot(tau1, tau2, '+', color=cs[i], markersize=5, markeredgewidth=1.5)
    ax1.text(tau1,
             tau2 + 1,
             chr(ord('a') + i),
             va='bottom',
             ha='center',
             color=cs[i])
    if tau1 != tau2:
        ax1.plot(tau2,
                 tau1,
                 '+',
                 color='grey',
                 markersize=5,
                 markeredgewidth=1.5)
        ax1.text(tau2,
                 tau1 + 1,
                 chr(ord('a') + i),
                 va='bottom',
                 ha='center',
                 color='grey')

ax1.text(-0.45,
         1.0,
         '\\textbf{A}',
         transform=ax1.transAxes,
         ha='left',
         va='top',
         fontsize=12)

ax_idx = 1
plot_in_ax = 0
for i, (tau1, tau2) in enumerate(trials):
    if i in [0, 3, 5]:
        ax_idx += 1
        plot_in_ax = 0
        ax = fig.add_subplot(gs1[ax_idx])
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 80)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        if ax_idx < 4:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time $t$ (ms)')
    else:
        plot_in_ax += 1
    ts, gs = second_order_filter(tau0=tau1 * 1e-3, tau1=tau2 * 1e-3)
    ax.plot(ts * 1e3, gs, color=cs[i])
    peak = np.argmax(gs)
    ax.plot(ts[peak] * 1e3,
            gs[peak],
            'o',
            color=cs[i],
            markersize=3,
            clip_on=False)
    #ax.plot([ts[peak] * 1e3, ts[peak] * 1e3], [gs[peak], - 183.5 - 91.5 * (2 - ax_idx)], 'k--', linewidth=0.4, clip_on=False, zorder=-100, color=cs[i])
    utils.annotate(ax, ts[peak] * 1e3 + 5, gs[peak] + 2,
                   ts[peak] * 1e3 + 20 + 10 * plot_in_ax,
                   gs[peak] + 15 * (3 - plot_in_ax - ax_idx + 2),
                   chr(ord('a') + i))

np.random.seed(7899)

times1 = [50] + [*np.random.uniform(300.0, 1000.0, 7)]
times2 = [150] + [*np.random.uniform(300.0, 1000.0, 8)]
w1, w2 = 1.0, 0.5
times = [(t1 * 1e-3, w1) for t1 in times1] + [(t2 * 1e-3, w2) for t2 in times2]

ax1 = fig.add_subplot(gs2[0])
ax1.text(-0.09,
         1.0,
         '\\textbf{B}',
         transform=ax1.transAxes,
         ha='left',
         va='top',
         fontsize=12)
ax1.set_title('Spike times of pre-neuron 1 ($t_{1, j}$)',
              x=0.0,
              y=0.7,
              va='top',
              ha='left')
ax1.spines['left'].set_visible(False)
ax1.set_yticks([])
ax1.set_xticklabels([])
for t in times1:
    ax1.plot([t, t], [0.1, 0.5],
             'k-',
             solid_capstyle='round',
             linewidth=1.5,
             color=utils.blues[0])
    ax1.plot([t, t], [0.1, -7.25],
             '--',
             linewidth=0.25,
             color=utils.blues[0],
             clip_on=False,
             zorder=-100)
ax1.set_ylim(0, 1)
ax1.set_xlim(0, 1000)

ax2 = fig.add_subplot(gs2[1])
ax2.set_title('Spike times of pre-neuron 2 ($t_{2, j}$)',
              x=0.0,
              y=0.95,
              va='top',
              ha='left',
              bbox={
                  "pad": 1.0,
                  "color": "w",
                  "linewidth": 0.0,
              })
ax2.spines['left'].set_visible(False)
ax2.set_yticks([])
ax2.set_xticklabels([])
for t in times2:
    ax2.plot([t, t], [0.1, 0.5],
             'k-',
             solid_capstyle='round',
             linewidth=1.5,
             color=utils.oranges[0])
    ax2.plot([t, t], [0.1, -4.1],
             '--',
             linewidth=0.25,
             color=utils.oranges[0],
             clip_on=False,
             zorder=-100)
ax2.set_ylim(0, 0.75)
ax2.set_xlim(0, 1000)

ax3 = fig.add_subplot(gs2[2])
ts, gs = second_order_filter(times,
                             T=1.0,
                             tau0=5e-3,
                             tau1=25e-3,
                             normalise=False)
ax3.plot(ts, gs / 0.015, 'k')
ax3.set_xlim(0, 1)
ax3.set_title(
    'Second-order low-pass model ($\\tau_1 = 5\\,\\mathrm{ms}$, $\\tau_2 = 25\\,\\mathrm{ms}$)',
    x=0.0,
    y=0.8,
    va='top',
    ha='left',
    bbox={
        "pad": 1.0,
        "color": "w",
        "linewidth": 0.0,
    })
ax3.spines['left'].set_visible(False)
ax3.set_yticks([])
ax3.set_xticklabels([])
ax3.set_ylim(0, 0.75)
ax3.set_ylabel('$g(t)$', y=0.3)
ax3.plot([0.04, 0.1], [0.24, 0.24], '--k', linewidth=0.5)
ax3.text(0.105, 0.24, "$w_1$", ha="left", va="center")
ax3.plot([0.14, 0.2], [0.12, 0.12], '--k', linewidth=0.5)
ax3.text(0.205, 0.13, "$w_2$", ha="left", va="center")

ax4 = fig.add_subplot(gs2[3])
ts, gs = first_order_filter(times, tau0=25e-3, T=1.0, normalise=False)
ax4.plot(ts, gs / 4.16, 'k')
ax4.set_xlim(0, 1)
ax4.set_title('First-order low-pass model ($\\tau=25\\,\\mathrm{ms}$)',
              x=0.0,
              y=0.8,
              va='top',
              ha='left',
              bbox={
                  "pad": 1.0,
                  "color": "w",
                  "linewidth": 0.0,
              })
ax4.spines['left'].set_visible(False)
ax4.set_ylim(0, 0.75)
ax4.set_yticks([])
ax4.set_xlabel('Time $t$ (s)')
ax4.set_ylabel('$g(t)$', y=0.3)
ax4.plot([0.04, 0.1], [0.24, 0.24], '--k', linewidth=0.5)
ax4.text(0.105, 0.24, "$w_1$", ha="left", va="center")
ax4.plot([0.14, 0.2], [0.12, 0.12], '--k', linewidth=0.5)
ax4.text(0.205, 0.13, "$w_2$", ha="left", va="center")

utils.save(fig)

