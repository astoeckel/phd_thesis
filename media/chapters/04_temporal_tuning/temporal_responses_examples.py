import scipy.interpolate
import matplotlib.colors as colors
import colorsys

fig = plt.figure(figsize=(8.0, 3.0))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.5)

axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[0, 2])

axD = fig.add_subplot(gs[1, 0])
axE = fig.add_subplot(gs[1, 1:3])

axs = [axA, axB, axC, axD, axE]
for ax in axs:
    ax.set_ylim(0, 2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ["left", "bottom", "right", "top"]:
        ax.spines[spine].set_visible(False)


def lighten(color, p=None, l_new=None, s_new=None):
    assert (p is None) != (l_new is None)
    color = colors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*color)
    if l_new is None:
        l = np.clip(l * (1.0 + p), 0.0, 1.0)
    else:
        l = l_new
    if not s_new is None:
        s = s_new
    return colorsys.hls_to_rgb(h, l, s)


def plot_pulse_fun(ax,
                   pnts,
                   y0=0.0,
                   h=0.6,
                   T=1.0,
                   dt=1e-3,
                   color=utils.blues[0],
                   kind='previous',
                   tau=None):
    ts = np.arange(0, T, dt)
    pnts = np.array(pnts)
    us = scipy.interpolate.interp1d(pnts[:, 0],
                                    pnts[:, 1],
                                    kind=kind,
                                    bounds_error=False,
                                    fill_value=0.0)(ts)

    if not tau is None:
        ts_flt = ts - T / 2
        hs = (ts_flt >= 0.0) * np.exp(-ts_flt / tau)
        hs /= np.sum(hs)
        us = np.convolve(us, hs, 'same')

    ax.plot(ts, y0 + h * us, color=color, clip_on=False)
    ax.fill_between(ts,
                    y0 * np.ones_like(us),
                    y0 + h * us,
                    color=lighten(color, l_new=0.8))
    ax.set_xlim(0, T)


plot_pulse_fun(axA, [(0.1, 1.0), (0.5, 0.0)], 1.0)
plot_pulse_fun(axA, [(0.1, 1.0), (0.5, 0.0)],
               0.0,
               color=utils.oranges[0],
               tau=0.01)

plot_pulse_fun(axB, [(0.1, 1.0), (0.5, 0.0)], 1.0)
plot_pulse_fun(axB, [(0.1, 0.0), (0.15, 1.0), (0.2, 0.9), (0.3, 0.4),
                     (0.4, 0.1), (0.5, 0.0)],
               0.0,
               color=utils.oranges[0],
               tau=0.01,
               kind='quadratic')

plot_pulse_fun(axC, [(0.1, 1.0), (0.2, 0.0)], 1.0)
plot_pulse_fun(axC, [(0.1, 0.2), (0.2, 0.2), (0.3, 0.0), (0.4, 0.0),
                     (0.45, 0.0), (0.55, 0.2), (0.65, 1.0), (0.8, 0.0)],
               0.0,
               kind='quadratic',
               color=utils.oranges[0],
               tau=0.01)

plot_pulse_fun(axD, [(0.1, 1.0), (0.5, 0.0)], 1.0)
plot_pulse_fun(axD, [(t, np.sin(1.5 * 2.0 * np.pi * (t - 0.1) / 0.4)**2.0)
                     for t in np.linspace(0.1, 0.5, 101)],
               0.0,
               kind='linear',
               color=utils.oranges[0],
               tau=0.01)

plot_pulse_fun(axE, [(0.08, 0.0), (0.08, 1.0), (0.4025, 1.0), (0.4025, 0.0),
                     (1.28, 0.0), (1.28, 1.0), (1.4, 1.0), (1.4, 0.0),
                     (1.48, 0.0), (1.48, 1.0), (1.6, 1.0), (1.6, 0.0)],
               1.0,
               T=2.0)
plot_pulse_fun(axE, [(0.08, 0.0), (0.08, 0.2), (0.5, 0.3), (0.5, 0.0),
                     (1.28, 0.0), (1.28, 0.2), (1.4, 0.3), (1.6, 1.0),
                     (1.7, 0.0)],
               0.0,
               kind='linear',
               T=2.0,
               color=utils.oranges[0],
               tau=0.01)

axA.arrow(0.75,
          0.8,
          0.2,
          0.0,
          width=0.02,
          head_width=0.1,
          head_length=0.05,
          overhang=0.25,
          linewidth=0,
          color='k',
          clip_on=False)
axA.text(0.86, 0.75, "$t$", ha="center", va="top")

utils.annotate(axA, 0.45, 1.35, 0.6, 1.35, "Stimulus", ha="left")
utils.annotate(axA, 0.45, 0.35, 0.6, 0.35, "Response", ha="left")

utils.timeslice(axC, 0.1, 0.69, 0.8)
utils.annotate(axC, 0.725, 0.8, 0.8, 0.8, '$\\theta$', ha="left")

utils.annotate(axE, 0.35, 1.35, 0.5, 1.35, "Suboptimal\nstimulus", ha="left")
utils.annotate(axE, 1.55, 1.35, 1.7, 1.35, "Preferred\nstimulus", ha="left")

axE.plot(1.02, 1.0, 's', color='white', markersize=2.5)
axE.plot([1.00, 1.02], [0.925, 1.075],
         color=utils.blues[0],
         solid_capstyle='round')
axE.plot([1.02, 1.04], [0.925, 1.075],
         color=utils.blues[0],
         solid_capstyle='round')

axE.plot(1.02, 0.0, 's', color='white', markersize=2.5, clip_on=False)
axE.plot([1.00, 1.02], [-0.075, 0.075],
         color=utils.oranges[0],
         solid_capstyle='round',
         clip_on=False)
axE.plot([1.02, 1.04], [-0.075, 0.075],
         color=utils.oranges[0],
         solid_capstyle='round',
         clip_on=False)

for i, ax in enumerate(axs):
    ax.text(0.0,
            0.95,
            '\\textbf{{{}}}'.format(chr(ord('A') + i)),
            ha="left",
            va="baseline",
            size=12,
            transform=ax.transAxes)
    ax.text(0.15 if i < 4 else 0.06,
            0.95,
            "\\textbf{" + [
                "Tonic response",
                "Phasic response",
                "Delayed response",
                "Oscillatory response",
                "Specific temporal tuning",
            ][i] + "}",
            ha="left",
            va="baseline",
            size=9,
            transform=ax.transAxes)

utils.save(fig)

