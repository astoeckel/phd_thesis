import nlif

g_c1 = 100e-9
g_c2 = 100e-9
E_E = 0.0e-3
E_I = -80.0e-3
g_L = 50e-9
E_L = -65e-3

with nlif.Neuron() as three_comp_lif:
    with nlif.Soma() as soma:
        nlif.CondChan(E_rev=E_L, g=g_L)
        chan_J = nlif.CurChan()

    with nlif.Compartment() as basal:
        nlif.CondChan(E_rev=E_L, g=g_L)
        chan_I = nlif.CondChan(E_rev=E_I)

    with nlif.Compartment() as apical:
        nlif.CondChan(E_rev=E_L, g=g_L)
        chan_E = nlif.CondChan(E_rev=E_E)

    nlif.Connection(soma, basal, g_c=g_c1)
    nlif.Connection(basal, apical, g_c=g_c2)

T=40e-3

fig, axs = plt.subplots(3, 3, figsize=(4.0, 2.5), gridspec_kw={
    "wspace": 0.6,
    "hspace": 0.2,
})

colors = [
    mpl.cm.get_cmap("viridis")(0.2),
    mpl.cm.get_cmap("viridis")(0.7),
    mpl.cm.get_cmap("viridis")(0.95),
]


for k, v0 in enumerate([-65e-3]):
    ts, xs = three_comp_lif.assemble().impulse_response(T=T, v0=[v0, v0, v0], xs={
        chan_J: 0.1e-9,
        chan_E: 1e-9,
        chan_I: 1e-9,
    })

    for i, idx_i in enumerate([2, 1, 0]):
        data = xs[:, idx_i, :] * 1e3
        for j, idx_j in enumerate([2, 1, 0]):
            ax = axs[j, i]
            ax.plot(ts * 1e3, data[:, idx_j], color=colors[j])
            ax.axhline(E_E * 1e3, linestyle="--", color='k', linewidth=0.5, zorder=-1)
            ax.axhline(E_I * 1e3, linestyle="--", color='k', linewidth=0.5, zorder=-1)
            ax.axhline(E_L * 1e3, linestyle="--", color='k', linewidth=0.5, zorder=-1)

            vmin, vmax = np.min(xs[:, :, idx_j]) * 1e3, np.max(xs[:, :, idx_j]) * 1e3
            ax.set_ylim(vmin - (vmax - vmin) * 0.1, vmax + (vmax - vmin) * 0.1)

            if j != 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Time $t$ ($\\mathrm{ms}$)")

#            if i != 0:
#                ax.set_yticklabels([])
#            else:
            if i == 0:
                ax.set_ylabel(f"$v_{{{idx_j + 1}}}$ ($\\mathrm{{mV}}$)")
            ax.set_xlim(0.0, T * 1e3)
            ax.set_xticks(np.linspace(0.0, T * 1e3, 3))
            ax.set_xticks(np.linspace(0.0, T * 1e3, 5), minor=True)
#        axs[i].set_ylabel("$v_i(t)$ ($\\mathrm{mV}$)")
#        axs[i].set_xlabel("Time $t$ ($\\mathrm{ms}$)")

utils.save(fig)

