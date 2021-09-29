import numpy as np
import nengo

np.random.seed(78971)

taus = np.logspace(-2, -1, 4)
tau_interm = 5e-3

T, dt = 1.002, 1.0e-3
ts = np.arange(0, T, dt)
us = nengo.processes.WhiteSignal(period=T, high=10.0).run(T, dt=dt)[:, 0]

ts_flt = np.arange(-T / 2, T / 2, dt)
delta = np.zeros_like(ts_flt)
delta[len(delta) // 2] = 1000.0
flts = np.array([(ts_flt >= 0.0) * (np.exp(-ts_flt / tau) / tau)
                 for tau in taus]).T
fflts = nengo.Lowpass(tau_interm).filt(flts, dt=dt)

tau_1 = 0.02
tau_2 = 0.04
alpha = 1.0
flt_tar = (ts_flt >= 0.0) * (((1.0 / tau_1) * np.exp(-ts_flt / tau_1) -
                              (alpha / tau_2) * np.exp(-ts_flt / tau_2)))
flt_tar /= 0.5 * np.sum(np.abs(flt_tar)) * dt

fus = np.array([nengo.Lowpass(tau).filt(us, dt=dt) for tau in taus]).T
ffus = nengo.Lowpass(tau_interm).filt(fus, dt=dt)

us_tar = np.convolve(us, flt_tar, 'same') * dt
D = np.linalg.lstsq(ffus, us_tar, rcond=None)[0]

colors = [
    utils.blues[0], utils.oranges[1], utils.greens[0], utils.reds[1],
    utils.purples[1]
]

fig, axs = plt.subplots(2,
                        4,
                        figsize=(7.8, 1.6),
                        gridspec_kw={
                            "wspace": 0.2,
                            "hspace": 0.2,
                            "height_ratios": [2.76, 1.5]
                        })

axs[0, 0].plot(ts, us, 'k')
axs[1, 0].plot(ts_flt, delta, 'k')

#axs[0, 1].plot(ts, us, 'k--', linewidth=0.5)
for i, tau in enumerate(taus):
    axs[0, 1].plot(ts,
                   fus[:, i],
                   color=colors[i],
                   label=f"$\\tau_{{{i + 1}}}$")
    axs[1, 1].plot(ts_flt, flts[:, i], color=colors[i])
    print(f"$\\tau_{{{i + 1}}} = {(tau*1e3):0.0f} \\,\\mathrm{{ms}}$")

#axs[0, 2].plot(ts, us, 'k--', linewidth=0.5)
for i, tau in enumerate(taus):
    axs[0, 2].plot(ts,
                   ffus[:, i],
                   color=colors[i],
                   label=f"$\\tau_{{{i + 1}}}$")
    axs[1, 2].plot(ts_flt, fflts[:, i], color=colors[i])
axs[1, 2].plot(ts_flt, fflts[:, i], color=colors[i])


axs[0, 3].plot(ts, ffus @ D, 'k')
axs[0, 3].plot(ts, us_tar, color='k', linewidth=0.5)
axs[0, 3].plot(ts, us_tar, color='white', linestyle=(0, (1, 2)), linewidth=0.5)

axs[1, 3].plot(ts_flt, fflts @ D, 'k-')
axs[1, 3].plot(ts_flt, flt_tar, color='k', linewidth=0.5)
axs[1, 3].plot(ts_flt, flt_tar, color='white', linestyle=(0, (1, 2)), linewidth=0.5)


#axs[0, 1].legend(loc="upper right",
#                 ncol=2,
#                 handlelength=1.0,
#                 handletextpad=0.5,
#                 columnspacing=1.0,
#                 bbox_to_anchor=(0.95, 1.1))
#axs[0, 2].legend(loc="upper right",
#                 ncol=2,
#                 handlelength=1.0,
#                 handletextpad=0.5,
#                 columnspacing=1.0,
#                 bbox_to_anchor=(0.95, 1.1))

for i in range(2):
    for j in range(4):
        if i in {0, 1}:
            axs[i, j].set_xticks(np.linspace(0, 1, 3))
            axs[i, j].set_xticks(np.linspace(0, 1, 5), minor=True)

            axs[i, j].set_yticks([])
            axs[i, j].spines["left"].set_visible(False)


        if i == 0:
            axs[i, j].set_ylim(-1.35, 1.35)
            axs[i, j].set_xlim(0, T)
            axs[i, j].text(0.0,
                           1.0,
                           "\\textbf{{{}}}".format(chr(ord('B') + j)),
                           size=12,
                           va="top",
                           ha="left",
                           transform=axs[i, j].transAxes)

        if i == 1:
            axs[i, j].set_xlim(-0.1, 0.2)
            axs[i, j].set_xticks(np.linspace(-0.1, 0.2, 4))
            axs[i, j].set_xlabel("Time $t$ (s)")


utils.save(fig)

