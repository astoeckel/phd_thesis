import dlop_ldn_function_bases as bases
import nengo

np.random.seed(5828)


def double_rows(A):
    res = np.zeros((A.shape[0] * 2, *A.shape[1:]))
    res[0::2] = A
    res[1::2] = A
    return res


def double(A):
    return double_rows(double_rows(A).T).T


def quadruple(A):
    return double(double(A))


def pp(ts, A):  # pre-process
    return nengo_extras.plot_spikes.cluster(ts, A,
                                            filter_width=0.01)[1]


def reduce_spike_train(A, ss):
    res = np.zeros((A.shape[0] // ss, A.shape[1]))
    for i in range(ss):
        res += A[i::ss][:res.shape[0]]
    return res


def mk_cosine_bartlett_basis_with_spread(q,
                                         N,
                                         T=1.0,
                                         dt=1e-3,
                                         phi_max=2.0,
                                         decay_min=0.5,
                                         decay_max=1.5,
                                         rng=np.random):
    phis = np.linspace(0, phi_max, q) * 2.0 * np.pi
    phases = rng.uniform(0, np.pi, q)
    t1 = rng.uniform(decay_min, decay_max, q) * T
    ts = np.arange(N) * dt
    return (
        np.cos(ts[None, :] * phis[:, None] / t1[:, None] + phases[:, None]) *
        (1.0 - ts[None, :] / t1[:, None]) * (ts[None, :] <= t1[:, None]))


def mk_gaussian_basis(q, N, T=1.0, dt=1e-3):
    ts = np.arange(N) * dt
    mus = np.sort(np.random.uniform(-0.1, T, q))
    sigmas = np.power(10.0, np.random.uniform(-1.0, -0.5, q))
    res = np.exp(-np.square(ts[None, :] - mus[:, None]) /
                 np.square(sigmas[:, None]))
    return res / np.sum(res, axis=1)[:, None] / dt


def solve_for_and_run_test_network():
    import matplotlib.pyplot as plt
    import numpy as np
    import temporal_encoder_common
    from temporal_encoder_common import Filters
    import nonneg_common
    import lif_utils
    import nengo
    import dlop_ldn_function_bases as bases

    n_neurons = 200
    n_temporal_dimensions = 200

    T, dt = 10.0, 1e-3
    ts = np.arange(0, T, dt)

    gains, biases, Es = nonneg_common.mk_ensemble(n_neurons,
                                                  d=n_temporal_dimensions)
    G = lif_utils.lif_rate
    TEs = np.diag(np.random.choice([-1, 1], n_temporal_dimensions))
    Ms = mk_gaussian_basis(n_temporal_dimensions, len(ts)).T

    W_in, W_rec = temporal_encoder_common.solve_for_recurrent_population_weights(
        G,
        gains,
        biases,
        None,
        None,
        Es,
        [Filters.lowpass(100e-3)],
        [Filters.lowpass(100e-3)],
        Ms=Ms,
        N_smpls=100,
        xs_sigma=3.0,
        biased=False,
    )

    with nengo.Network() as model:
        nd_in = nengo.Node(
            nengo.processes.WhiteSignal(period=10.0, high=1.0, y0=0.0,
                                        rms=0.5))
        ens_x = nengo.Ensemble(
            n_neurons=n_neurons,
            dimensions=n_temporal_dimensions,
            bias=biases,
            gain=gains,
            encoders=Es,
        )

        nengo.Connection(nd_in,
                         ens_x.neurons,
                         transform=W_in[:, 0:1],
                         synapse=100e-3)

        nengo.Connection(ens_x.neurons,
                         ens_x.neurons,
                         transform=W_rec[:, :, 0],
                         synapse=100e-3)

        p_in = nengo.Probe(nd_in, synapse=None)
        p_out = nengo.Probe(ens_x.neurons, synapse=None)

    with nengo.Simulator(model) as sim:
        sim.run(10.0)

    return sim.data[p_out]


q = 101
dt = 1e-3
N = 2000
T = N * dt

Es = np.diag(np.random.choice([-1, 1], q))
Es[0, 0] = 1.0
H = Es @ mk_gaussian_basis(q, N)

ts = np.arange(0, N) * dt

fig, axs = plt.subplots(1,
                        3,
                        figsize=(7.5, 1.75),
                        gridspec_kw={
                            "wspace": 0.4,
                            "width_ratios": [2, 1, 3]
                        })

axs[0].plot(ts, H[::5].T, 'grey', lw=0.5, linestyle=":")
sel = np.linspace(0, q - 1, 6, dtype=int)
for i in sel:
    axs[0].plot(ts, H[i].T, lw=1.2, zorder=1000 - i)
axs[0].set_xlim(0, T)
axs[0].set_ylim(-5, 5)
axs[0].set_xlabel("Time $t$ (s)")
axs[0].set_ylabel("$\\mathfrak{b}_i(t)$")
axs[0].set_xticks(np.linspace(0, 2, 3))
axs[0].set_xticks(np.linspace(0, 2, 5), minor=True)
axs[0].set_yticks(np.linspace(-6, 6, 5))
axs[0].set_yticks(np.linspace(-6, 6, 13), minor=True)
axs[0].set_title("\\textbf{Temporal encoders $\\mathfrak{e}_i$}")
utils.annotate(axs[0], 0.6, 2.6, 1.0, 3.0, "On-neuron", va="center", ha="left")
utils.annotate(axs[0],
               1.0,
               -3.2,
               1.25,
               -4.0,
               "Off-neuron",
               va="center",
               ha="left")

U, S, V = np.linalg.svd(H)
axs[1].plot(np.arange(1, q + 1), S / np.max(S), 'k-', markersize=5, lw=1.25)
axs[1].set_yscale('log')
axs[1].set_ylim(1e-6, 1)
axs[1].set_xticks([1, 14, 50])
axs[1].set_xticks([1, 25, 50], minor=True)
axs[1].set_xlim(1, 50)
axs[1].set_xlabel("Index $i$")
axs[1].set_ylabel("Singular value $\\sigma_i / \\sigma_1$")
axs[1].axhline(1e-2, linestyle=":", linewidth=0.5, color='grey')
axs[1].axvline(14, linestyle=":", linewidth=0.5, color='grey')
axs[1].set_title("\\textbf{Singular values}")

As = utils.run_with_cache(solve_for_and_run_test_network)
As_flt = nengo.Lowpass(100e-3).filtfilt(As)
As_sum = np.sum(As_flt, axis=0)
sel = np.logical_and(As_sum > np.percentile(As_sum, 25),
                     As_sum < np.percentile(As_sum, 75))
N = np.sum(sel)
As_norm = 1.0 / np.max(As_flt[:, sel], axis=0)
sort_idcs = np.argsort(np.argmax(As_flt[0:2000, sel], axis=0))

axs[2].imshow(quadruple(reduce_spike_train(As[4000:9000, sel][:, sort_idcs], 10)).T, vmin=0.0, vmax=1.0, cmap='Greys', extent=[0, 5, N + 0.5, 0.5])
axs[2].set_xlim(0, 5)

#axs[2].imshow()

#axs[2].imshow((As[::10, sel][:, sort_idcs] * As_norm[None, :]).T,
#              vmin=0,
#              vmax=1,
#              cmap="magma",
#              )
axs[2].set_yticks(np.linspace(1, 100, 6, dtype=int))
axs[2].set_aspect('auto')
axs[2].set_xlabel("Time $t$ (s)")
axs[2].set_ylabel("Neuron index $i$")
axs[2].set_title("\\textbf{Neural activities}")

fig.text(0.0775, 0.929, "\\textbf{A}", size=12, va="baseline", ha="left")
fig.text(0.355, 0.929, "\\textbf{B}", size=12, va="baseline", ha="left")
fig.text(0.54, 0.929, "\\textbf{C}", size=12, va="baseline", ha="left")

utils.save(fig)

