import nengo


def run_model(a0=0.0, b0=10.0, tau=100e-3, T=2.0, ff=True):

    with nengo.Network() as model:
        nd_u = nengo.Node(lambda t: 1.0 * np.logical_and(t >= 0.2, t < 1.2))
        ens_x = nengo.Ensemble(n_neurons=1, dimensions=1, neuron_type=nengo.Direct())
        nengo.Connection(nd_u, ens_x, synapse=tau, transform=tau * b0)
        if not ff:
            nengo.Connection(ens_x,
                             ens_x,
                             synapse=tau,
                             transform=tau * a0 + 1.0)

        p_u = nengo.Probe(nd_u, synapse=None)
        p_x = nengo.Probe(ens_x, synapse=None)

    with nengo.Simulator(model) as sim:
        sim.run(T)

    return sim.trange(), sim.data[p_u], sim.data[p_x]


cm = plt.get_cmap('inferno')

fig, axs = plt.subplots(1, 2, figsize=(7.85, 1.0))

ts, us, xs = run_model()
axs[0].plot(ts, us, 'k', lw=0.5, zorder=-100)
axs[1].plot(ts, us, 'k', lw=0.5, zorder=-100)
axs[0].plot(ts, us, ':', color='white', lw=0.5)
axs[1].plot(ts, us, ':', color='white', lw=0.5)

axs[0].plot(ts, xs, color=cm(0.05), zorder=-1)

axs[0].spines["left"].set_visible(False)
axs[0].set_yticks([])
axs[0].set_xlim(0, 2)
axs[0].set_xlabel("Time $t$ (s)")

axs[1].spines["left"].set_visible(False)
axs[1].set_yticks([])
axs[1].set_xlim(0, 2)
axs[1].set_xlabel("Time $t$ (s)")

tau_invs = np.logspace(1, 2.5, 5)
for i, tau_inv in enumerate(tau_invs):
    print(1.0 / tau_inv)
    scale = 1.0 - np.exp(-tau_inv)
    _, _, xs = run_model(-tau_inv, tau_inv / scale, ff=False)
    axs[1].plot(ts,
                xs,
                color=cm(0.0 + 0.8 * i / (len(tau_invs) - 1)),
                zorder=-i,
                label=f"$\\tau' = {1000.0/tau_inv:0.0f}\\,\\mathrm{{ms}}$")

axs[1].legend(loc="upper center",
              ncol=7,
              bbox_to_anchor=(-0.11, 1.35),
              handlelength=1.0,
              handletextpad=0.5,
              columnspacing=1.0)

utils.save(fig)

