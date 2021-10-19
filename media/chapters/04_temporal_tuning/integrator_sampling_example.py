import nonneg_common
import temporal_encoder_common
from temporal_encoder_common import Filters
import lif_utils
import nengo

def run_experiment(biased=False):
    n_neurons = 100
    n_temporal_dimensions = 1

    gains, biases, Es = nonneg_common.mk_ensemble(n_neurons, d=n_temporal_dimensions)
    G = lif_utils.lif_rate

    xs = np.linspace(-1, 1, 1001)
    A_post = G(gains[None, :] * (xs[:, None] @ Es.T) + biases[None, :])

    # Integrator dynamics
    A = np.array(((0,),))
    B = np.array(((1,)))

    # Generate the reference dynamics
    T, dt = 10.0, 1e-3
    ts = np.arange(0, T, dt)
    Ms = temporal_encoder_common.cached_lti_impulse_response(A, B, ts)

    flts_in = [(100e-3,),]
    flts_rec = [(100e-3,),]

    W_in, W_rec = temporal_encoder_common.solve_for_recurrent_population_weights(
        G, gains, biases, None, None, Es,
        [Filters.lowpass(*flt_in) for flt_in in flts_in],
        [Filters.lowpass(*flt_rec) for flt_rec in flts_rec],
        Ms=Ms,
        N_smpls=100,
        biased=biased,
        xs_scale=0.1, # Exaggerate the effect of bias
    )

    def LP(*args):
        return nengo.LinearFilter(*Filters.lowpass_laplace(*args), analog=True)

    with nengo.Network() as model:
        nd_in = nengo.Node(lambda t: 1.0 * (t > 0.5 and t < 1.5))
        ens_x = nengo.Ensemble(
            n_neurons=n_neurons,
            dimensions=n_temporal_dimensions,
            bias=biases,
            gain=gains,
            encoders=Es,)

        for i, flt_in in enumerate(flts_in):
            nengo.Connection(
                nd_in, ens_x.neurons,
                transform=W_in[:, i].reshape(-1, 1),
                synapse=LP(*flt_in))

        for i, flt_rec in enumerate(flts_rec):
            nengo.Connection(
                ens_x.neurons, ens_x.neurons,
                transform=W_rec[:, :, i],
                synapse=LP(*flt_rec))

        p_in = nengo.Probe(nd_in, synapse=None)
        p_out = nengo.Probe(ens_x, synapse=100e-3)

    with nengo.Simulator(model) as sim:
        sim.run(10.0)

    return xs, A_post, W_in, W_rec, sim.trange(), sim.data[p_in], sim.data[p_out]


def setup_ax0(ax):
    ax.plot([-1, 1], [-1, 1], 'k:', lw=0.5)
    ax.plot([-1, 1], [1, -1], 'k:', lw=0.5)
    ax.set_ylim(-1, 1)
    ax.set_xticks(np.linspace(-1, 1, 3), minor=False)
    ax.set_xticks(np.linspace(-1, 1, 5), minor=True)
    ax.set_yticks(np.linspace(-1, 1, 3), minor=False)
    ax.set_yticks(np.linspace(-1, 1, 5), minor=True)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("State $m(t)$")
    ax.set_ylabel("Feedback $\\xi_i$")

def setup_ax1(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.025, 1.025)
    ax.set_xticks(np.linspace(0, 10, 3))
    ax.set_xticks(np.linspace(0, 10, 5), minor=True)
    ax.set_yticks(np.linspace(0, 1, 3))
    ax.set_yticks(np.linspace(0, 1, 5), minor=True)
    ax.set_xlabel("Time $t$ (s)")
    ax.set_ylabel("State $m(t)$")

fig, axs = plt.subplots(1, 4, figsize=(7.5, 1.4), gridspec_kw={
    "wspace": 0.4,
})

np.random.seed(43289)
xs, A_post, W_in, W_rec, ts, us, ys  = run_experiment(biased=False)
axs[0].plot(xs, A_post @ W_rec[0, :, 0].T, color=utils.blues[0], lw=1.2);
axs[0].plot(xs, A_post @ W_rec[1, :, 0].T, color=utils.oranges[1], lw=1.2);
utils.annotate(axs[0], 0.5, 0.5, 0.0, 0.8, "$e^\\mathrm{t}_i = 1$")
utils.annotate(axs[0], 0.5, -0.5, 0.0, -0.8, "$e^\\mathrm{t}_i = -1$")
setup_ax0(axs[0])

axs[1].plot(ts, us, 'k--', lw=0.5)
axs[1].plot(ts, ys, 'k', lw=0.7)
setup_ax1(axs[1])

np.random.seed(43289)
xs, A_post, W_in, W_rec, ts, us, ys  = run_experiment(biased=True)
axs[2].plot(xs, A_post @ W_rec[0, :, 0].T, color=utils.blues[0], lw=1.2);
axs[2].plot(xs, A_post @ W_rec[1, :, 0].T, color=utils.oranges[1], lw=1.2);
setup_ax0(axs[2])

axs[3].plot(ts, us, 'k--', lw=0.5)
axs[3].plot(ts, ys, 'k', lw=0.7)
setup_ax1(axs[3])

fig.text(0.0775, 0.94, "\\textbf{A}", size=12, ha="left", va="baseline")
fig.text(0.295, 0.94, "\\textbf{Na\\\"ively sampled $\\mathfrak{x}_k$}", ha="center", va="baseline")

fig.text(0.495, 0.94, "\\textbf{B}", size=12, ha="left", va="baseline")
fig.text(0.7, 0.94, "\\textbf{Uniform activation sampling}", ha="center", va="baseline")

utils.save(fig)

