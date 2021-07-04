def sim_lif(ts,
            Js,
            vReset=-85e-3,
            vTh=-55e-3,
            tauRef=2e-3,
            EL=-65e-3,
            Cm=200e-12,
            gL=10e-9):
    dt = (ts[-1] - ts[0]) / (len(ts) - 1)
    t_spikes = []
    us = np.ones_like(ts) * EL
    tRefs = np.zeros_like(ts)
    for i in range(1, len(ts)):
        tRefs[i] = np.clip(tRefs[i - 1] - dt, 0.0, None)
        if tRefs[i] <= 0.0:
            us[i] = us[i - 1] + dt * (Js[i] + gL * (EL - us[i - 1])) / Cm
        else:
            us[i] = vReset
        if us[i] >= vTh:
            t_spikes.append(ts[i])
            us[i] = np.nan
            tRefs[i] = tauRef

    return us, tRefs, t_spikes


dt = 1e-5
T = 25e-3
ts = np.arange(0, T, dt)

Js = (np.logical_and(ts >= 1e-3, ts <= 2e-3) * 1e-9 +
      np.logical_and(ts >= 10e-3, ts <= 13e-3) * 1e-9 +
      np.logical_and(ts >= 20e-3, ts <= 23e-3) * 2e-9)

us, tRefs, t_spikes = sim_lif(ts, Js)

fig, (ax1, ax2, ax3) = plt.subplots(3,
                                    1,
                                    figsize=(3.6, 2.0),
                                    sharex=True,
                                    gridspec_kw={"height_ratios": [4, 1, 1], "hspace": 0.25})

ax1.axhline(-65, color='k', linestyle=':', linewidth=0.5)
ax1.axhline(-55, color='k', linestyle=':', linewidth=0.5)
ax1.axhline(-85, color='k', linestyle=':', linewidth=0.5)
ax1.plot(ts * 1e3, us * 1e3, color=utils.blues[0])
ax1.set_yticks(np.arange(-85, -54, 10))

utils.timeslice(ax1, 11.43, 13.43, -80)
ax1.text(12.35, -79, '$\\tau_\\mathrm{ref}$', ha="center", va="bottom")

utils.timeslice(ax1, 22.42, 24.42, -80)
ax1.text(23.34, -79, '$\\tau_\\mathrm{ref}$', ha="center", va="bottom")

for i, u in enumerate(us):
    if np.isnan(u):
        ax1.scatter(ts[i] * 1e3,
                    -55,
                    marker='o',
                    ec=utils.blues[0],
                    fc='white',
                    s=20,
                    zorder=100)
        ax1.scatter(ts[i] * 1e3,
                    -85,
                    marker='o',
                    ec=utils.blues[0],
                    fc=utils.blues[0],
                    s=20,
                    zorder=100)
utils.annotate(ax1, 3, -84.5, 4.25, -80, '$v_\mathrm{reset}$')
utils.annotate(ax1, 5.25, -65.5, 6.5, -70.5, '$E_\mathrm{L}$')
utils.annotate(ax1, 15.0, -55.5, 16.5, -60.5, '$v_\mathrm{th}$')
ax1.set_ylabel('$v$ (mV)', labelpad=3)
ax1.set_title("LIF neuron")

ax2.plot(ts * 1e3, Js * 1e9, color='k', clip_on=False)
ax2.set_ylabel('$J$ (nA)', labelpad=12)
ax2.set_ylim(0, 2)

ax3.plot(ts * 1e3, tRefs * 1e3, color='k', clip_on=False)
ax3.set_xlabel('Time $t$ (ms)', labelpad=2)
ax3.set_ylabel('$t_\\mathrm{ref}$', labelpad=12)
ax3.set_xlim(0, T * 1e3)
ax3.set_ylim(0, 2)

for t in t_spikes:
    ax1.axvline(t * 1e3, linestyle='--', linewidth=0.5, color='k')
    ax2.axvline(t * 1e3, linestyle='--', linewidth=0.5, color='k')
    ax3.axvline(t * 1e3, linestyle='--', linewidth=0.5, color='k')

utils.save(fig)

