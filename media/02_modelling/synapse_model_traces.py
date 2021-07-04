def sim_synapse(t_spikes=[10e-3],
                T=0.2,
                w=100e-9,
                delay=2e-3,
                tau0=5e-3,
                tau1=20e-3,
                dt=1e-4,
                Esyn=0e-3,
                EL=-65e-3,
                Cm=200e-12,
                gL=10e-9):
    t_spikes = sorted(t_spikes)
    ts = np.arange(0, T, dt)
    us = np.ones_like(ts) * EL
    gs0 = np.zeros_like(ts)
    gs1 = np.zeros_like(ts)
    Js = np.zeros_like(ts)
    for i in range(1, len(ts)):
        gs0[i] = gs0[i - 1] - dt * gs0[i - 1] / tau0
        gs1[i] = gs1[i - 1] - dt * (gs1[i - 1] - gs0[i - 1]) / tau1
        Js[i] = gs1[i - 1] * (Esyn - us[i])
        us[i] = us[i - 1] + dt * (Js[i] + gL * (EL - us[i - 1])) / Cm
        while len(t_spikes) > 0 and t_spikes[0] + delay <= ts[i]:
            gs0[i] += w
            t_spikes.pop(0)

    return ts, us, Js, gs1


fig, axs = plt.subplots(3,
                        2,
                        figsize=(3.9, 2.2825),
                        gridspec_kw={
                            "wspace": 0,
                            "hspace": 0.5
                        })

for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.axvline(10, linestyle='--', color='gray', linewidth=0.75)

for i in range(2):
    axs[0, i].set_ylim(-5, 20)
    axs[1, i].set_ylim(-95, 5)
    axs[2, i].set_ylim(-0.5, 1.5)
    axs[1, i].axhline(0, color='k', linewidth=0.75, linestyle=":")
    axs[1, i].axhline(-65, color='k', linewidth=0.5)
    axs[1, i].axhline(-90, color='k', linewidth=0.75, linestyle=":")
    axs[2, i].axhline(0, color='k', linewidth=0.5)
    axs[0, i].axhline(0, color='k', linewidth=0.5)

ts, us, Js, gs = sim_synapse()

axs[0, 0].plot(ts * 1e3, gs * 1e9, color=utils.blues[0])
axs[0, 0].plot([150, 200], [5, 5], 'k-', linewidth=1.5, solid_capstyle='butt')
axs[0, 0].text(175, 6.5, '$50\\,\\mathrm{ms}$', va='bottom', ha='center')
utils.annotate(axs[0, 0], 60.0, 5.0, 110.0, 10.0, '$g_\\mathrm{E}(t)$')

axs[1, 0].plot(ts * 1e3, us * 1e3, color=utils.blues[0])
axs[1, 0].text(0.5,
               1.025,
               "EPSP",
               va="bottom",
               ha="center",
               transform=axs[1, 0].transAxes)
utils.annotate(axs[1, 0], 160.0, 5.0, 180.0, 30.0, "$E_\\mathrm{E}$", va="center")
utils.annotate(axs[1, 0], 85.0, -37.5, 120.0, -20.0, "$v(t)$", va="center")

axs[2, 0].plot(ts * 1e3, Js * 1e9, color=utils.blues[0])
axs[2, 0].text(0.5,
               1.025,
               "EPSC",
               va="bottom",
               ha="center",
               transform=axs[2, 0].transAxes)
utils.annotate(axs[2, 0], 52.0, 0.5, 95.0, 1.0, "$J_\\mathrm{syn}(t)$", va="center")

ts, us, Js, gs = sim_synapse(Esyn=-90e-3)

axs[0, 1].plot(ts * 1e3, gs * 1e9, color=utils.oranges[0])
axs[0, 1].plot([150, 150], [5, 15], 'k-', linewidth=1.5, solid_capstyle='butt')
axs[0, 1].text(155, 10, '$10\\,\\mathrm{nS}$', va='center', ha='left')
utils.annotate(axs[0, 1], 60.0, 5.0, 110.0, 10.0, '$g_\\mathrm{I}(t)$')

axs[1, 1].plot(ts * 1e3, us * 1e3, color=utils.oranges[0])
axs[1, 1].text(0.5,
               1.025,
               "IPSP",
               va="bottom",
               ha="center",
               transform=axs[1, 1].transAxes)
axs[1, 1].plot([150, 150], [-50, -10], 'k-', linewidth=1.5, solid_capstyle='butt')
axs[1, 1].text(155, -30, '$40\\,\\mathrm{mV}$', va='center', ha='left')
utils.annotate(axs[1, 1], 145.0, -97.0, 157.0, -112.0, "$E_\\mathrm{I}$", va="top", ha="left")
utils.annotate(axs[1, 1], 50.0, -52.5, 62.5, -20.0, "$E_\\mathrm{L}$", va="center")

axs[2, 1].plot(ts * 1e3, Js * 1e9, color=utils.oranges[0])
axs[2, 1].text(0.5,
               1.025,
               "IPSC",
               va="bottom",
               ha="center",
               transform=axs[2, 1].transAxes)
axs[2, 1].plot([150, 150], [0.25, 1.25], 'k-', linewidth=1.5, solid_capstyle='butt')
axs[2, 1].text(155, 0.75, '$1\\,\\mathrm{nA}$', va='center', ha='left')
utils.annotate(axs[2, 1], 23.0, -0.2, 70.0, 0.75, "$J_\\mathrm{syn}(t)$", va="center", ha="center")

for ax in axs[:, 1].flat:
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)

axs[0, 0].text(1.0, 1.45, "Post-synaptic potentials and currents", transform=axs[0, 0].transAxes, va="bottom", ha="center")
axs[0, 0].text(0.5, 1.05, "\\textit{Excitatory}", transform=axs[0, 0].transAxes, va="bottom", ha="center")
axs[0, 1].text(0.5, 1.05, "\\textit{Inhibitory}", transform=axs[0, 1].transAxes, va="bottom", ha="center")

utils.save(fig)

