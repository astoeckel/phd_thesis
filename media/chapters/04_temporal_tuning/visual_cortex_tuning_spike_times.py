# See data/hubel_wiesel_1959_digitization
spike_times = [
    [],
    [0.054],
    [0.03, 0.128],
    [0.00926, 0.09088, 0.14391, 0.29092, 0.37257, 0.48277, 0.68278],
    [
        0.02081, 0.04227, 0.07145, 0.09863, 0.12738, 0.16381, 0.19339, 0.2744,
        0.39203
    ],
    [0.01234, 0.06376],
    [],
    [],
]

# Reverse the list for counter-clockwise rotation
spike_times = spike_times[0:1] + spike_times[::-1]

fig = plt.figure(figsize=(7.25, 2.0))

gs1 = fig.add_gridspec(1,
                       2,
                       left=0.055,
                       right=0.5,
                       top=0.9,
                       bottom=0.1,
                       wspace=-0.1)
axs = np.array([fig.add_subplot(gs1[0, i]) for i in range(2)])

for ax in axs.flat:
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-1, 4)
    ax.set_ylim(1.7, 8.9)

for idx, angle in enumerate(np.arange(0, 180, 22.5)):
    i = idx % 4
    ax = axs[idx // 4]
    phi = np.pi * angle / 180
    x0, y0 = 0.4, 8.4 - i * 2.1
    ax.plot(x0,
            y0,
            '+',
            markersize=10,
            markeredgewidth=0.5,
            color='white',
            clip_on=False,
            zorder=10)

    rect = mpl.patches.Rectangle((x0 - 0.27, y0 - 0.46),
                                 0.54,
                                 0.92,
                                 edgecolor='None',
                                 facecolor='#3d3846',
                                 clip_on=False)
    ax.add_artist(rect)

    ax.plot([x0 - 0.2 * np.cos(phi), x0 + 0.2 * np.cos(phi)],
            [y0 - 0.35 * np.sin(phi), y0 + 0.35 * np.sin(phi)],
            color=utils.oranges[2],
            linewidth=3,
            clip_on=False)

    ax.plot([1.0, 4.0], [y0, y0], 'k-', linewidth=0.5, clip_on=False)

    for j, t in enumerate(spike_times[idx]):
        x0 = 2.0
        ax.plot([x0 + t, x0 + t], [y0 - 0.2, y0 + 0.2],
                'k-',
                solid_capstyle='round',
                linewidth=1.0,
                clip_on=False)

    ax.plot([2.0, 3.0], [y0 - 0.4, y0 - 0.4],
            '-',
            color='#3d3846',
            solid_capstyle='butt',
            linewidth=2.0,
            clip_on=False)

    if idx == 0:
        ax.text(3.1, y0 - 0.4, '$1\\,\\mathrm{s}$', va="center", ha="left")
        utils.annotate(ax,
                       2.5,
                       y0 - 0.2,
                       2.25,
                       y0 + 0.55,
                       '\\textit{Stimulus presentation}',
                       va="bottom")

    if idx == 4:
        utils.annotate(ax,
                       2.5,
                       y0 + 0.25,
                       2.75,
                       y0 + 0.55,
                       '\\textit{Neural activity}',
                       va="bottom")

gs2 = fig.add_gridspec(2,
                       2,
                       left=0.55,
                       right=0.95,
                       top=0.9,
                       bottom=0.25,
                       wspace=0.5,
                       hspace=1.0)
axs = [
    fig.add_subplot(gs2[0, 0]),
    fig.add_subplot(gs2[1, 0:2]),
    fig.add_subplot(gs2[0, 1]),
]

ts = np.linspace(-2, 2, 10001)
dt = (ts[-1] - ts[0]) / (len(ts) - 1)

tau1, tau2 = 0.01, 0.1
flt1 = (ts >= 0.0) * (np.exp(-ts / tau1))
flt1 /= np.sum(np.abs(flt1)) * dt
flt2 = (ts >= 0.0) * (np.exp(-ts / tau2))
flt2 /= np.sum(np.abs(flt2)) * dt

flt = flt1 - flt2
flt /= 0.5 * np.sum(np.abs(flt)) * dt
axs[0].plot(ts * 1e3, flt, color='k')
axs[0].set_xlim(-100, 300)
axs[0].set_xlabel("Time $t$ (ms)")
utils.annotate(axs[0],
               20,
               100,
               70,
               100,
               'Impulse\nresponse',
               ha="left",
               fontdict={"size": 8})

H = np.fft.fftshift(np.fft.fft(flt))
fs = np.fft.fftshift(np.fft.fftfreq(len(ts), dt))

axs[2].plot(fs, 10.0 * np.log10(np.abs(H) / np.max(np.abs(H))), 'k-')
axs[2].set_xlim(0, 20)
axs[2].set_ylim(-5, 1)
axs[2].set_xlabel("Frequency $f$ (Hz)")
axs[2].set_ylabel("Gain (dB)")
utils.annotate(axs[2],
               2.5,
               -0.8,
               6.1,
               -2.75,
               'Frequency\nresponse',
               ha="left",
               fontdict={"size": 8})

us = np.zeros(len(flt))
us[ts > 0.0] = 1.0
us[ts > 0.2] = 0.0

axs[1].plot(ts * 1e3, us, linestyle='--', color='k', linewidth=0.7)
axs[1].plot(ts * 1e3, np.convolve(us, flt, 'same') * dt, color='k')
axs[1].set_xlim(-100, 500)
axs[1].set_ylim(-1.1, 1.1)
axs[1].set_xlabel("Time $t$ (ms)")
utils.annotate(axs[1],
               140,
               0.5,
               220,
               0.5,
               'Step response',
               ha="left",
               fontdict={"size": 8})

for i in range(2):
    #    axs[i].set_xticks([])
    #    axs[i].set_yticks([])
    #    axs[i].spines["bottom"].set_visible(False)
    #    axs[i].spines["left"].set_visible(False)
    axs[i].axhline(0, linestyle=':', color='k', linewidth=0.5)

fig.text(0.107, 1.03, '\\textbf{A}', size=12, ha="left", va="baseline")
fig.text(0.3, 1.03, '\\textbf{Raw spike data}', ha="center", va="baseline")

fig.text(0.52, 1.03, '\\textbf{B}', size=12, ha="left", va="baseline")
fig.text(0.75,
         1.03,
         '\\textbf{High-pass impulse, frequency, and step response}',
         ha="center",
         va="baseline")

utils.save(fig)

