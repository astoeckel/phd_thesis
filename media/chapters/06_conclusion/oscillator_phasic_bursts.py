import temporal_encoder_common
from temporal_encoder_common import Filters
import nonneg_common
import lif_utils
import nengo


def execute_network(xs,
                    W_in,
                    W_rec,
                    gains,
                    biases,
                    T=10.0,
                    dt=1e-4,
                    tau=100e-3):
    N = int(T / dt + 1e-9)
    n_neurons = len(gains)

    with nengo.Network() as model:
        nd_in = nengo.Node(lambda t: 20.0 * np.logical_and(t >= 0.1, t <= 0.2))

        ens_x = nengo.Ensemble(n_neurons=n_neurons,
                               dimensions=1,
                               bias=biases,
                               gain=gains,
                               encoders=np.ones((n_neurons, 1)))

        nengo.Connection(nd_in,
                         ens_x.neurons,
                         transform=W_in[:, :, 0],
                         synapse=tau)

        nengo.Connection(ens_x.neurons,
                         ens_x.neurons,
                         transform=W_rec[:, :, 0],
                         synapse=tau)

        p_in = nengo.Probe(nd_in, synapse=None)
        p_out = nengo.Probe(ens_x.neurons, synapse=None)
        p_out_v = nengo.Probe(ens_x.neurons, 'voltage', synapse=None)

    with nengo.Simulator(model, dt=dt) as sim:
        sim.run(T)

    return sim.data[p_out], sim.data[p_out_v]


def mk_oscillator(f, T=1.0, dt=1e-3):
    ts = np.arange(0, T, dt)
    return ts, np.array([
        np.sin(2.0 * np.pi * f * ts),
        np.cos(2.0 * np.pi * f * ts),
    ]).T



def run_network():
    np.random.seed(58381)

    n_neurons = 100
    n_temporal_dimensions = 2
    n_dimensions = 1

    T, dt = 10.0, 1e-3
    _, _, TEs = nonneg_common.mk_ensemble(n_neurons, d=n_temporal_dimensions)
    gains = np.ones(n_neurons) * 3
    biases = np.ones(n_neurons)

    idcs = np.argsort(np.arctan2(TEs[:, 1], TEs[:, 0]), axis=0)
    TEs = TEs[idcs]

    G = lif_utils.lif_rate
    Es = np.ones((n_neurons, 1))

    ts, Ms = mk_oscillator(4.0, T=T, dt=dt)

    flts_in = [(100e-3,),]
    flts_rec = [(100e-3,),]

    W_in, W_rec, errs = temporal_encoder_common.solve_for_recurrent_population_weights_with_spatial_encoder(
        G, gains, biases, None, None, TEs, Es,
        [Filters.lowpass(*flt_in) for flt_in in flts_in],
        [Filters.lowpass(*flt_rec) for flt_rec in flts_rec],
        Ms=Ms,
        N_smpls=1000,
        xs_sigma=3.0,
        biased=False,
        dt=dt)

    return execute_network(None, W_in, W_rec, gains, biases)


ts = np.arange(0.0, 10.0, 1e-4)
As, As_v = utils.run_with_cache(run_network)
n_neurons = As.shape[1]

fig = plt.figure(figsize=(7.4, 2.0))
gs = fig.add_gridspec(4, 3)

ax1 = fig.add_subplot(gs[:, 0])
ax1.eventplot([np.where(As[10000:40000, i])[0] * 1e-4 for i in range(n_neurons)], color='k');
ax1.set_ylabel('Neuron index $i$')
ax1.set_xlim(0, 3)
ax1.set_ylim(-0.5, 100.5)
ax1.set_xlabel("Time $t$ (s)")
ax1.set_xticks(np.linspace(0, 3, 13), minor=True)
utils.outside_ticks(ax1)

ax2 = fig.add_subplot(gs[:, 1])
ax2.eventplot([np.where(As[10000:20000, i])[0] * 1e-4 for i in range(n_neurons)], color='k');
ax2.set_xlim(0, 1)
ax2.set_ylim(-0.5, 100.5)
ax2.set_xlabel("Time $t$ (s)")
utils.outside_ticks(ax2)

for i in range(4):
    idx = np.linspace(0, 99, 6, dtype=int)[1:-1][i]
    ax = fig.add_subplot(gs[i, 2])
    ax.plot(ts[10000:40000] - 1.0, np.clip(As_v + As, None, 5)[10000:40000, idx], color='k', lw=0.7)
    ax.set_xticks(np.linspace(0, 3, 13), minor=True)
    ax.set_xlim(0.0, 3.0)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    utils.outside_ticks(ax)

    if i < 3:
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
        ax.set_xticks([], minor=True)

ax.set_xlabel("Time $t$ (s)")

fig.text(0.065, 0.94, "\\textbf{A}", size=12, va="baseline", ha="left")
fig.text(0.24, 0.94, "\\textbf{Spike raster}", va="baseline", ha="center")

fig.text(0.365, 0.94, "\\textbf{B}", size=12, va="baseline", ha="left")
fig.text(0.52, 0.94, "\\textbf{Spike raster (detail)}", va="baseline", ha="center")

fig.text(0.67, 0.94, "\\textbf{C}", size=12, va="baseline", ha="left")
fig.text(0.785, 0.94, "\\textbf{Voltage traces}", va="baseline", ha="center")


utils.save(fig)
