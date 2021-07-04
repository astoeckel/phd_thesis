import nengo
from nengo.utils.ensemble import sorted_neurons, response_curves, tuning_curves

np.random.seed(49829)


def rasterplot(ax, ts, As):
    #cmap = cm.get_cmap('viridis')
    for i in range(As.shape[1]):
        color = 'k'  #cmap(i / (As.shape[1] - 1))
        for t in ts[As[:, i] != 0]:
            ax.plot([t, t], [i + 0.5, i + 1.5],
                    color=color,
                    linewidth=0.5,
                    solid_capstyle='butt')


def build_and_run(neuron_type=nengo.neurons.LIF,
                  T=3.0,
                  tau=100e-3,
                  tau_record=100e-3):
    A = np.array((
        (-1.0, -2.0 * np.pi),
        (2.0 * np.pi, -1.0),
    ))
    B = np.array((
        (6.0, ),
        (0.0, ),
    ))

    Ap = tau * A + np.eye(2)
    Bp = tau * B

    with nengo.Network(seed=5781) as model:
        node_u = nengo.Node(lambda t: 1.0 * ((t > 0.2) and (t < 0.4)))
        ens_u = nengo.Ensemble(neuron_type=neuron_type(),
                               n_neurons=100,
                               dimensions=1,
                               max_rates=nengo.dists.Uniform(10, 20),
                               encoders=np.ones((100, 1)),
                               intercepts=nengo.dists.Uniform(-0.1, 1.0))
        ens_x = nengo.Ensemble(neuron_type=neuron_type(),
                               n_neurons=100,
                               dimensions=2,
                               max_rates=nengo.dists.Uniform(10, 50))

        nengo.Connection(node_u, ens_u, synapse=None)
        nengo.Connection(ens_u, ens_x, transform=Bp, synapse=tau)
        nengo.Connection(ens_x, ens_x, transform=Ap, synapse=tau)

        pu = nengo.Probe(ens_u, synapse=tau_record)
        px = nengo.Probe(ens_x, synapse=tau_record)
        if not neuron_type is nengo.neurons.Direct:
            pu_neurons = nengo.Probe(ens_u.neurons)
            px_neurons = nengo.Probe(ens_x.neurons)

    with nengo.Simulator(model, seed=5810) as sim:
        sim.run(T)

    if neuron_type is nengo.neurons.Direct:
        return sim.trange(), sim.data[pu], sim.data[px]
    else:
        u_idcs = np.argsort(sim.data[ens_u].intercepts)
        #u_idcs = sorted_neurons(ens_u, sim)
        x_idcs = sorted_neurons(ens_x, sim)
        return (sim.trange(), sim.data[pu], sim.data[px],
                sim.data[pu_neurons][:, u_idcs],
                sim.data[px_neurons][:, x_idcs], sim, ens_u, ens_x)


ts, Sus, Sxs, Aus, Axs, sim, ens_u, ens_x = build_and_run()
ts, us, xs = build_and_run(neuron_type=nengo.neurons.Direct, tau_record=None)

fig, ((ax01, ax02), (ax11, ax12),
      (ax21, ax22)) = plt.subplots(3,
                                   2,
                                   figsize=(3.5, 3.0),
                                   gridspec_kw={"hspace": 1.0})

ax01.plot(ts, us, color='k', linewidth=1.0)
ax01.text(1.0, 1.0, r"$u(t)$", transform=ax01.transAxes, ha="right", va="top")
ax01.set_xlim(0, 3)
ax01.set_ylim(0, 1)
ax01.set_yticks([])
ax01.spines["left"].set_visible(False)

ax02.plot(ts, xs[:, 0], linewidth=1.0, color='#204a87ff')
ax02.plot(ts, xs[:, 1], linewidth=1.0, color='#f57900ff')
ax02.text(1.0,
          1.0,
          r"$\mathbf{x}(t)$",
          transform=ax02.transAxes,
          ha="right",
          va="top")
ax02.set_xlim(0.0, 3.0)
ax02.set_ylim(-1, 1)
ax02.set_yticks([])
ax02.spines["left"].set_visible(False)

rasterplot(ax11, ts, Aus[:, ::5])
ax11.text(1.0,
          1.2,
          r"$\mathbf{a}_{u}(t)$",
          transform=ax11.transAxes,
          ha="right",
          va="top")
ax11.set_xlim(0.0, 3.0)
ax11.set_yticks([])
ax11.spines["left"].set_visible(False)

rasterplot(ax12, ts, Axs[:, ::5])
ax12.text(1.0,
          1.2,
          r"$\mathbf{a}_{\mathbf{x}}(t)$",
          transform=ax12.transAxes,
          ha="right",
          va="top")
ax12.set_xlim(0.0, 3.0)
ax12.set_yticks([])
ax12.spines["left"].set_visible(False)

ax21.plot(ts, Sus, color='k', linewidth=1.0, clip_on=False)
ax21.text(1.0,
          1.0,
          r"$\hat u(t)$",
          transform=ax21.transAxes,
          ha="right",
          va="top")
ax21.set_ylim(0, 1)
ax21.set_xlim(0.0, 3.0)
ax21.set_yticks([])
ax21.spines["left"].set_visible(False)

ax22.plot(ts, Sxs[:, 0], linewidth=1.0, color='#204a87ff')
ax22.plot(ts, Sxs[:, 1], linewidth=1.0, color='#f57900ff')
ax22.text(1.0,
          1.0,
          r"$\mathbf{\hat x}(t)$",
          transform=ax22.transAxes,
          ha="right",
          va="top")
ax22.set_ylim(-1, 1)
ax22.set_xlim(0.0, 3.0)
ax22.set_yticks([])
ax22.spines["left"].set_visible(False)

ax01.set_xlabel("Time $t$ (s)")
ax02.set_xlabel("Time $t$ (s)")
ax11.set_xlabel("Time $t$ (s)")
ax12.set_xlabel("Time $t$ (s)")
ax21.set_xlabel("Time $t$ (s)")
ax22.set_xlabel("Time $t$ (s)")

utils.save(fig)

# Tuning curves

fig, ax = plt.subplots(figsize=(1.25, 0.75))
u_idcs = np.argsort(sim.data[ens_u].intercepts)
us = np.linspace(-1, 1, 1001).reshape(-1, 1)
eval_points, activities = tuning_curves(ens_u, sim, us)
activities = activities[:, u_idcs]
for i in range(0, activities.shape[1], 10):
    color = 'k' #cm.get_cmap('tab20')(i / (activities.shape[1] - 1))
    ax.plot(eval_points, activities[:, i], linewidth=0.75, color=color)
ax.set_ylabel(r'$a_i(u)$ ($\mathrm{s}^{-1}$)')
ax.set_xlabel('Represented $u$')
ax.set_xlim(-0.5, 1)
ax.set_ylim(0, 20)
utils.save(fig, suffix="_tuning_u")

fig, ax = plt.subplots(figsize=(1.25, 0.75))
u_idcs = np.argsort(
    np.arctan2(sim.data[ens_x].encoders[:, 0], sim.data[ens_x].encoders[:, 1]))
phis = np.linspace(-np.pi, np.pi, 1001)
us = np.array((np.sin(phis), np.cos(phis))).T
eval_points, activities = tuning_curves(ens_x, sim, us)
activities = activities[:, u_idcs]
for i in range(0, activities.shape[1], 10):
    color = 'k' #cm.get_cmap('viridis')(i / (activities.shape[1] - 1))
    ax.plot(phis, activities[:, i], linewidth=0.75, color=color)
ax.set_ylabel('$a_i(\\vec x)$ ($\mathrm{s}^{-1}$)')
ax.set_xlabel(r'Represented $\angle \mathbf{x}$')
ax.set_xlim(-np.pi, np.pi)
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_xticklabels([r"$-\pi$", "$0$", "$\pi$"])
ax.set_ylim(0, 50)
utils.save(fig, suffix="_tuning_x")

# Post-synaptic currents

fig, ax = plt.subplots(figsize=(1.25, 0.75))
ts = np.linspace(0.0, 1.0, 1000)
ax.plot(ts, np.exp(-ts / 0.1) / 0.1, linewidth=0.75, color='k')
ax.text(0.5, 0.5, "$\\tau = 100\\,\\mathrm{ms}$", ha="left", va="bottom", transform=ax.transAxes)
ax.set_yticklabels([])
ax.set_ylabel(r'PSC')
ax.set_xlabel(r'Time $t$ (s)')
ax.set_xlim(0.0, 1.0)
utils.save(fig, suffix="_psc")

