import numpy as np
import nengo
import nengo_bio as bio

np.random.seed(58799)

with nengo.Network() as model:
    nd_input = nengo.Node(nengo.processes.WhiteSignal(high=1.0, period=10.0))

    ens_src = bio.Ensemble(n_neurons=25,
                           dimensions=1,
                           locations=bio.NeuralSheetDist(dimensions=2))
    ens_tar = bio.Ensemble(n_neurons=100,
                           dimensions=1,
                           locations=bio.NeuralSheetDist(dimensions=2))

    nengo.Connection(nd_input, ens_src)
    conn = bio.Connection(ens_src,
                          ens_tar,
                          connectivity=bio.SpatiallyConstrainedConnectivity(
                              convergence=5, sigma=0.25))

    probe_in = nengo.Probe(ens_src, synapse=0.1)
    probe_out = nengo.Probe(ens_tar, synapse=0.1)

fig, axs = plt.subplots(1, 3, figsize=(7.5, 1.5), gridspec_kw={
    "wspace": 0.4,
    "width_ratios": [1, 1, 1.75]
})

with nengo.Simulator(model) as sim:
    loc_src = sim.data[ens_src].locations
    loc_tar = sim.data[ens_tar].locations

    con = np.logical_or(sim.data[conn].connectivity[0],
                        sim.data[conn].connectivity[1])

    ps = bio.SpatiallyConstrainedConnectivity(
                convergence=10,
                sigma=0.5
            ).get_probabilities(25, 100, ens_src, ens_tar, sim.data)

    sim.run(10.0)

    ts = sim.trange()
    xs = sim.data[probe_in]
    ys = sim.data[probe_out]

for i in range(con.shape[0]):
    for j in range(con.shape[1]):
        if con[i, j]:
            #                axs[0].plot([loc_src[i, 0], loc_tar[j, 0]],
            #                            [loc_src[i, 1], loc_tar[j, 1]], 'k-', zorder=0)
            r = max(0.01, np.hypot(loc_tar[j, 0] - loc_src[i, 0],
                         loc_tar[j, 1] - loc_src[i, 1]) - 0.1)
            alpha = np.arctan2(loc_tar[j, 1] - loc_src[i, 1],
                               loc_tar[j, 0] - loc_src[i, 0])
            axs[0].arrow(loc_src[i, 0],
                         loc_src[i, 1],
                         r * np.cos(alpha),
                         r * np.sin(alpha),
                         zorder=0,
                         width=0.01,
                         linewidth=0,
                         head_width=0.04,
                         overhang=0.3,
                         color='k')

axs[0].scatter(loc_src[:, 0], loc_src[:, 1], s=30, zorder=2, color=utils.blues[0], label="Pre")
axs[0].scatter(loc_tar[:, 0], loc_tar[:, 1], s=7, marker='o', zorder=1, color=utils.oranges[1], label="Post")
axs[0].set_xlim(-1, 1)
axs[0].set_ylim(-1, 1)
axs[0].set_xlabel("Location $x_1$")
axs[0].set_ylabel("Location $x_2$")
#axs[0].set_aspect(1)
axs[0].set_xticks(np.linspace(-1, 1, 3))
axs[0].set_xticks(np.linspace(-1, 1, 5), minor=True)
axs[0].set_yticks(np.linspace(-1, 1, 3))
axs[0].set_yticks(np.linspace(-1, 1, 5), minor=True)
axs[0].set_title("\\textbf{Spatial connectivity}", y=1.2)
axs[0].legend(loc="upper center", ncol=2, columnspacing=1.0, handletextpad=0.25, bbox_to_anchor=(0.5, 1.3))
axs[0].text(-0.275, 1.275, '\\textbf{A}', size=12, ha="left", va="baseline", transform=axs[0].transAxes)

I = axs[1].imshow(ps / np.max(ps), interpolation=None, cmap='Blues')
axs[1].set_aspect('auto')
axs[1].set_xlabel("Post-neuron index $j$")
axs[1].set_ylabel("Pre-neuron index $i$")
axs[1].set_title("\\textbf{Normalised probabilities}", y=1.2)
axs[1].text(-0.275, 1.275, '\\textbf{B}', size=12, ha="left", va="baseline", transform=axs[1].transAxes)

cgs = fig.add_gridspec(1, 3, wspace=0.4, top=1.05, bottom=1.0, width_ratios=[1, 1, 1.75])
cax = fig.add_subplot(cgs[0, 1])
cb = plt.colorbar(I, ax=axs[1], cax=cax, orientation='horizontal')
cb.outline.set_visible(False)

axs[2].plot(ts, xs, ':', color=utils.blues[0], label="Pre")
axs[2].plot(ts, ys, '-', color=utils.oranges[1], label="Post")
axs[2].set_xlim(0, 10)
axs[2].set_ylim(-1, 1)
axs[2].legend(loc="upper center", ncol=2, columnspacing=1.0, handlelength=1.0, handletextpad=0.5, bbox_to_anchor=(0.5, 1.3))
axs[2].set_title("\\textbf{Decoded pre- and post-neuron activities}", y=1.2)
axs[2].set_xlabel("Time $t$ (s)")
axs[2].text(-0.15, 1.275, '\\textbf{C}', size=12, ha="left", va="baseline", transform=axs[2].transAxes)

utils.save(fig)

