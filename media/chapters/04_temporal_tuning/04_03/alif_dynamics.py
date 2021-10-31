import temporal_encoder_common
from temporal_encoder_common import Filters
import nonneg_common
import lif_utils
import nengo


def LP(*args):
    return nengo.LinearFilter(*Filters.lowpass_laplace(*args), analog=True)


np.random.seed(58381)

# Generate a neuron population tuned to an oscillator
n_neurons = 100
n_temporal_dimensions = 2

gains, biases, TEs = nonneg_common.mk_ensemble(
    n_neurons,
    d=n_temporal_dimensions,
    max_rates=(
        200.0,
        200.0,
    ),
)
G = lif_utils.lif_rate

A = np.array((
    (0.0, 2.0 * np.pi * 2.0),
    (-2.0 * np.pi * 2.0, 0.0),
))
B = np.array((
    1.0,
    0.0,
)).T

T, dt = 3.0, 1e-3
ts = np.arange(0, T, dt)
Ms = temporal_encoder_common.cached_lti_impulse_response(A, B, ts)

flts_in = [(200e-3, )]
flts_rec = [(200e-3, )]

# Solve for weights

Y_nonlin, XS = None, None
W_in, W_rec, Y_nonlin, XS, AS = temporal_encoder_common.solve_for_recurrent_population_weights_nonlin(
    G,
    gains,
    biases,
    None,
    None,
    TEs,
    [Filters.lowpass(*flt_in) for flt_in in flts_in],
    [Filters.lowpass(*flt_rec) for flt_rec in flts_rec],
    Ms=Ms,
    Y_nonlin=Y_nonlin,
    XS=XS,
    N_smpls=101,
    xs_sigma=3.0,
    T=T,
    dt=dt,
    biased=True,
)

# Account for adaptation
tau_adap = 100e-3
n0 = 0.01
h = np.exp(-ts / tau_adap)
h /= np.sum(h)
J_adap = n0 * np.array(
    [[np.convolve(AS[i, j], h, 'valid') for j in range(AS.shape[1])]
     for i in range(AS.shape[0])])
Y_nonlin = J_adap[..., -1] / gains[None, :]

W_in_comp, W_rec_comp, _, _, _ = temporal_encoder_common.solve_for_recurrent_population_weights_nonlin(
    G,
    gains,
    biases,
    None,
    None,
    TEs,
    [Filters.lowpass(*flt_in) for flt_in in flts_in],
    [Filters.lowpass(*flt_rec) for flt_rec in flts_rec],
    Ms=Ms,
    Y_nonlin=Y_nonlin,
    XS=XS,
    N_smpls=101,
    xs_sigma=3.0,
    T=T,
    dt=dt,
    biased=True,
)


def execute_network_ref(XS,
                        AS,
                        TEs,
                        W_in,
                        W_rec,
                        gains,
                        biases,
                        flts_in,
                        flts_rec,
                        dt=1e-3,
                        neuron_type=nengo.LIF()):
    N = len(XS)
    T = N * dt
    n_neurons = len(gains)
    n_dims = TEs.shape[1]

    with nengo.Network() as model:
        nd_ref = nengo.Node(lambda t: AS[:, int(t / dt) % N])
        ens_ref = nengo.Ensemble(n_neurons=n_neurons,
                                 dimensions=1,
                                 bias=np.zeros(n_neurons),
                                 gain=np.ones(n_neurons),
                                 neuron_type=nengo.SpikingRectifiedLinear())

        nengo.Connection(nd_ref, ens_ref.neurons, transform=np.eye(n_neurons))

        nd_in = nengo.Node(lambda t: XS[int(t / dt) % N])

        ens_x = nengo.Ensemble(n_neurons=n_neurons,
                               dimensions=n_dims,
                               bias=biases,
                               gain=gains,
                               encoders=TEs,
                               neuron_type=neuron_type)

        for i, flt_in in enumerate(flts_in):
            nengo.Connection(nd_in,
                             ens_x.neurons,
                             transform=W_in[:, i:(i + 1)],
                             synapse=LP(*flt_in))

        for i, flt_rec in enumerate(flts_rec):
            nengo.Connection(ens_ref.neurons,
                             ens_x.neurons,
                             transform=W_rec[:, :, i],
                             synapse=LP(*flt_rec))

        p_ref_in = nengo.Probe(nd_ref, synapse=100e-3)
        p_ref_as = nengo.Probe(ens_ref.neurons, synapse=None)
        p_out_as = nengo.Probe(ens_x.neurons, synapse=None)

    with nengo.Simulator(model, dt=dt) as sim:
        sim.run(T)

    return sim.trange(
    ), sim.data[p_ref_in], sim.data[p_ref_as], sim.data[p_out_as]


def execute_network(xs,
                    TEs,
                    W_in,
                    W_rec,
                    gains,
                    biases,
                    flts_in,
                    flts_rec,
                    T=10.0,
                    dt=1e-3,
                    neuron_type=nengo.LIF()):
    if xs is None:
        N = int(T / dt + 1e-9)
    else:
        N = len(xs)
        T = N * dt
    n_neurons = len(gains)
    n_dims = TEs.shape[1]

    with nengo.Network() as model:
        if xs is None:
            nd_in = nengo.Node(lambda t: 20.0 * (t >= 0.1) * (t < 0.15))
        else:
            nd_in = nengo.Node(lambda t: xs[int(t / dt) % N])

        ens_x = nengo.Ensemble(n_neurons=n_neurons,
                               dimensions=n_dims,
                               bias=biases,
                               gain=gains,
                               encoders=TEs,
                               neuron_type=neuron_type)

        for i, flt_in in enumerate(flts_in):
            nengo.Connection(nd_in,
                             ens_x.neurons,
                             transform=W_in[:, i:(i + 1)],
                             synapse=LP(*flt_in))

        for i, flt_rec in enumerate(flts_rec):
            nengo.Connection(ens_x.neurons,
                             ens_x.neurons,
                             transform=W_rec[:, :, i],
                             synapse=LP(*flt_rec))

        p_in = nengo.Probe(nd_in, synapse=None)
        p_out = nengo.Probe(ens_x.neurons, synapse=None)
        p_out_dec = nengo.Probe(ens_x, synapse=100e-3)

    with nengo.Simulator(model, dt=dt) as sim:
        sim.run(T)

    return sim.trange(), sim.data[p_in], sim.data[p_out], sim.data[p_out_dec]


def execute_network2(W_in, W_rec, neuron_type):
    ts, us, _, xs = execute_network(None,
                                    TEs,
                                    W_in,
                                    W_rec,
                                    gains,
                                    biases,
                                    flts_in,
                                    flts_rec,
                                    neuron_type=neuron_type)

    ts2, _, ref_as, out_as = execute_network_ref(XS[48],
                                                 AS[48],
                                                 TEs,
                                                 W_in,
                                                 W_rec,
                                                 gains,
                                                 biases,
                                                 flts_in,
                                                 flts_rec,
                                                 neuron_type=neuron_type)

    return ts, ts2, us, xs, ref_as, out_as


ts, ts2, us, xs_lif, ref_as, out_as_lif = execute_network2(
    W_in, W_rec, nengo.LIF())
_, _, _, xs_alif, _, out_as_alif = execute_network2(
    W_in, W_rec, nengo.AdaptiveLIF(tau_n=tau_adap, inc_n=n0))
_, _, _, xs_alif_comp, _, out_as_alif_comp = execute_network2(
    W_in_comp, W_rec_comp, nengo.AdaptiveLIF(tau_n=tau_adap, inc_n=n0))

fig, axs = plt.subplots(2,
                        3,
                        figsize=(7.3, 2.0),
                        gridspec_kw={
                            "hspace": 0.7,
                        })


def plot_as(ax, ts, as_tar, as_ref, as_net, i=7):
    as_tar_flt = nengo.Lowpass(100e-3).filt(as_tar[i])
    as_ref_flt = nengo.Lowpass(100e-3).filt(as_ref[:, i], dt=1e-3)
    as_net_flt = nengo.Lowpass(100e-3).filt(as_net[:, i], dt=1e-3)
    ax.set_xlim(-1, 0)
    ax.plot(ts - 3.0, as_ref_flt, color='k')
    ax.plot(ts - 3.0, as_net_flt, color=utils.purples[1])
    ax.plot(ts - 3.0, as_tar_flt, '--', color="white", lw=0.5)
    ax.plot(0.0,
            as_tar_flt[-1],
            'o',
            markersize=6,
            markerfacecolor='none',
            markeredgecolor='k',
            clip_on=False)
    ax.plot(0.0,
            as_net_flt[-1],
            'o',
            markersize=6,
            markerfacecolor='none',
            markeredgecolor=utils.purples[1],
            clip_on=False)
    ax.set_ylim(0, 160)
    ax.set_yticks(np.linspace(0, 150, 4), minor=True)

    utils.annotate(
        ax, 0.0, as_net_flt[-1], -0.5, 60,
        "$\\varepsilon_{{ik}} = {:0.2f} \\,\\mathrm{{s}}^{{-1}}$".format(
            as_net_flt[-1] - as_tar_flt[-1]), ha="right", va="top")


def plot_xs(ax, ts, xs):
    ut0, ut1 = np.min(ts[us[:, 0] != 0.0]), np.max(ts[us[:, 0] != 0.0])
    ax.plot(ts, xs[:, 0], color=utils.blues[0])
    ax.plot(ts, xs[:, 1], color=utils.oranges[1])
    ax.set_xlim(0, ts[-1] + 1e-3)
    ax.set_xticks(np.linspace(0, 10, 5), minor=True)
    ax.set_xlabel("Time $t$ (s)")
    ax.set_ylim(-0.75, 0.75)
    ax.plot([ut0, ut1], [0.7, 0.7],
            solid_capstyle="butt",
            color='k',
            linewidth=2,
            clip_on=False)


plot_as(axs[0, 0], ts2, AS[48], ref_as, out_as_lif)
axs[0, 0].set_ylabel("Rate ($s^{-1}$)")
plot_as(axs[0, 1], ts2, AS[48], ref_as, out_as_alif)
plot_as(axs[0, 2], ts2, AS[48], ref_as, out_as_alif_comp)

plot_xs(axs[1, 0], ts, xs_lif)
axs[1, 0].set_ylabel("Decoded $x_i$")
plot_xs(axs[1, 1], ts, xs_alif)
plot_xs(axs[1, 2], ts, xs_alif_comp)

axs[0, 2].legend([
    mpl.lines.Line2D([], [], color='k'),
    mpl.lines.Line2D([], [], color=utils.purples[1]),
], [
    "Expected $\\mathfrak{a}_i(\\mathfrak{x}_k)$",
    "Actual $\\hat{\\mathfrak{a}}_i(\\mathfrak{x}_k)$"
],
                 loc="upper right",
                 bbox_to_anchor=(1.05, 1.49),
                 ncol=2,
                 handlelength=1.0,
                 handletextpad=0.5,
                 columnspacing=1.0)

axs[1, 2].legend([
    mpl.lines.Line2D([], [], color=utils.blues[0]),
    mpl.lines.Line2D([], [], color=utils.oranges[1]),
], ["$x_1(t)$", "$x_2(t)$"],
                 loc="upper right",
                 bbox_to_anchor=(1.05, 1.49),
                 ncol=2,
                 handlelength=1.0,
                 handletextpad=0.5,
                 columnspacing=1.0)

for i in range(2):
    for j in range(3):
        axs[i, j].text(1.0,
                       0.05, ["LIF", "ALIF", "ALIF (comp.)"][j],
                       bbox={
                           "color": "white",
                           "pad": 0.2
                       },
                       ha="right",
                       va="bottom",
                       transform=axs[i, j].transAxes)

fig.text(0.065, 0.92, "\\textbf{A}", size=12, va="baseline", ha="left")
fig.text(0.09,
         0.92,
         "\\textbf{Single neuron response} (synthetic spiking input)",
         va="baseline",
         ha="left")
fig.text(0.065, 0.43, "\\textbf{B}", size=12, va="baseline", ha="left")
fig.text(0.09, 0.43, "\\textbf{Decoded dynamics}", va="baseline", ha="left")

fig.align_labels(axs)

utils.save(fig)

