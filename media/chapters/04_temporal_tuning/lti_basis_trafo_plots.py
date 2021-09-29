import nengo
import dlop_ldn_function_bases as bases

fig = plt.figure(figsize=(7.5, 5.5))

np.random.seed(5782781)

T = 4.0
T0 = 2.0
T1 = 3.0
dt = 1e-3
ts = np.arange(0, T, dt)
ts_flt = np.arange(T0, T1, dt)
us = nengo.processes.WhiteSignal(period=T, high=1.0, y0=0.0).run(T, dt=dt)[:, 0]

B = bases.mk_dlop_basis(3, len(ts_flt))
B = B / np.max(B, axis=1)[:, None]


def setup_ax(ax):
    utils.remove_frame(ax)
    ax.set_xlim(0, T)
    return ax


gs = fig.add_gridspec(1, 1, left=0.1, right=0.4, top=0.9, bottom=0.8)
ax = setup_ax(fig.add_subplot(gs[0, 0]))
ax.plot(ts, us, 'k')
ax.plot([0, T], [0, 0], 'k:', linewidth=0.5)
ax.axvline(T0, color='k', linestyle='--', linewidth=0.5)
ax.axvline(T1, color='k', linestyle='--', linewidth=0.5)
utils.annotate(ax,
               0.5,
               us[int(0.6 / dt)],
               0.25,
               1.0,
               "$u(t)$",
               va="baseline",
               ha="center")

ax.text(T0, 1.2, '$t - \\theta$', size=8, va="bottom", ha="center")
ax.text(T1, 1.2, '$t$', size=8, va="bottom", ha="center")

y0, y1 = 0.8, 0.1
gap = 0.05
h = (y0 - y1) / B.shape[0]

Es = np.zeros((len(ts), B.shape[0]))

for j in range(B.shape[0]):
    c = [utils.blues[0], utils.oranges[1], utils.greens[0]][j]
    #c = mpl.cm.get_cmap("viridis")([0.2, 0.65, 0.95][j]) # * np.array((0.9, 0.9, 0.9, 1.0))
    #c = cm.get_cmap('viridis')(j / (B.shape[0] - 1)) * np.array((0.9, 0.9, 0.9, 1.0))

    gs = fig.add_gridspec(2,
                          1,
                          left=0.1,
                          right=0.4,
                          top=y0 - h * j - gap,
                          bottom=y0 - h * (j + 1),
                          hspace=0.5)

    ax = setup_ax(fig.add_subplot(gs[0, 0]))
    ax.plot(ts_flt, B[j], color=c, clip_on=False, lw=1.5)
    ax.plot([0, T0], [0, 0], color=c, clip_on=False, lw=1.5)
    ax.plot([T1, T], [0, 0], color=c, clip_on=False, lw=1.5)
    ax.plot(T0, 0, 'o', color=c, markersize=4, fillstyle='none', clip_on=False)
    ax.plot(T0, B[j, 0], 'o', color=c, markersize=4, clip_on=False)
    ax.plot(T1,
            B[j, -1],
            'o',
            color=c,
            markersize=4,
            fillstyle='none',
            clip_on=False)
    ax.plot(T1, 0, 'o', color=c, markersize=4, clip_on=False)
    ax.set_ylim(-1.0, 1.0)

    ax.plot([0, T], [0, 0], 'k:', linewidth=0.5, zorder=-2)
    ax.axvline(T0, color='k', linestyle='--', linewidth=0.5, zorder=-1)
    ax.axvline(T1, color='k', linestyle='--', linewidth=0.5, zorder=-1)
    ax.text(T0, -1.2, '$0$', size=8, va="top", ha="center")
    ax.text(T1, -1.2, '$\\theta$', size=8, va="top", ha="center")

    utils.annotate(ax,
                   T0 - 0.1,
                   B[j, 0],
                   T0 - 0.25,
                   B[j, 0],
                   '$\\mathfrak{{b}}_{}(t)$'.format(j),
                   ha="right",
                   va="center")

    ax = setup_ax(fig.add_subplot(gs[1, 0]))
    Es[:, j] = np.convolve(B[j], us, 'full')[:len(ts)] * dt
    ax.plot(ts, Es[:, j], color=c, clip_on=False, lw=1.5)
    ax.plot([0, T], [0, 0], 'k:', linewidth=0.5, zorder=-2)
    ax.axvline(T1, color='k', linestyle='--', linewidth=0.5, zorder=-1)
    ax.text(T1, min(-0.55, Es[int(T1 / dt), j] - 0.15), '$t$', size=8, va="top", ha="center")
    ax.plot(T1, Es[int(T1 / dt), j], 's', color=c, markersize=4, clip_on=False)
    ax.set_ylim(-0.5, 0.5)
    utils.annotate(ax,
                   0.8,
                   0.65 + 0.8 * (Es[int(0.8 / dt), j] - 0.65),
                   0.75,
                   0.65,
                   '$m_{}(t) = (u \\ast \mathfrak{{b}}_{})(t)$'.format(j, j),
                   ha="center",
                   va="baseline")

gs = fig.add_gridspec(1,
                      1,
                      left=0.53,
                      right=0.84,
                      top=0.425 * y0 + 0.575 * y1 + 0.5 * h,
                      bottom=0.425 * y0 + 0.575 * y1 - 0.5 * h,
                      hspace=0.5)
ax = setup_ax(fig.add_subplot(gs[0, 0]))

ax.plot(ts, us, 'k', linewidth=0.5)
utils.annotate(ax,
               0.5,
               us[int(0.6 / dt)],
               0.25,
               1.0,
               "$u(t)$",
               va="baseline",
               ha="center")

N_delay = int(0.5 / dt)
us_delayed = np.concatenate((np.zeros(N_delay), us[:-N_delay]))

D = np.linalg.lstsq(Es, us_delayed, rcond=None)[0]
us_decoded = Es @ D
print(D)

ax.plot(ts, us_decoded, lw=1.5, color='k')
utils.annotate(ax,
               2.25,
               1.1 * us_decoded[int(2.25 / dt)],
               2.25,
               1.75,
               "$y(t) \\approx u(t - 0.5 \\theta)$",
               va="baseline",
               ha="center")

fig.text(0.05, 0.95, "\\textbf{C}", size=12, va="baseline")
fig.text(0.25,
         0.95,
         "\\textbf{Convolution over time}",
         size=9,
         ha="center",
         va="baseline")

fig.text(0.464, 0.65, "\\textbf{D}", size=12, va="baseline")
fig.text(0.69,
         0.65,
         "\\textbf{Decoding a delay}",
         size=9,
         ha="center",
         va="baseline")

utils.save(fig)

