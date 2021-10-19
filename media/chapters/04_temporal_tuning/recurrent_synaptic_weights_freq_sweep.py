import os, sys

sys.path.append(
    os.path.join(utils.libdir, '..', 'chapters', '04_temporal_tuning'))

import evaluate_synaptic_weight_computation_freq_sweep as exp
import nonneg_common
import lif_utils

exp.N_TRAIN_SMPLS = 100


def run_experiment(solver_mode):
    idcs = [
        exp.SOLVER_MODES.index(solver_mode),
        exp.MODES.index("mod_fourier_erasure"),
        list(exp.QS).index(7),
        list(exp.NEURONS).index(100), 0
    ]

    Ws_rec = exp.execute_single(idcs, return_weights=True)

    return Ws_rec


mode_colors = {
    "mod_fourier_bartlett": mpl.cm.get_cmap("viridis")(0.1),
    "mod_fourier_erasure": mpl.cm.get_cmap("viridis")(0.4),
    "legendre_erasure": mpl.cm.get_cmap("viridis")(0.95),
}

mode_markers = {
    "mod_fourier_bartlett": "o",
    "mod_fourier_erasure": "h",
    "legendre_erasure": "^",
}

files = [
    #    utils.datafile(f"evaluate_synaptic_weight_computation_{i}.h5")
    #    for i in range(10)
    utils.datafile("evaluate_synaptic_weight_computation_freq_sweep.h5")
]

errs_tuning, errs_delay = [None] * 2
for i, fn in enumerate(files):
    print(f"Loading {fn}...")
    with h5py.File(fn, "r") as f:
        if i == 0:
            solver_modes = json.loads(f.attrs["solver_modes"])
            modes = json.loads(f.attrs["modes"])
            qs = json.loads(f.attrs["qs"])
            neurons = json.loads(f.attrs["neurons"])
            xs_sigma_test = json.loads(f.attrs["xs_sigma_test"])
            errs_delay = f["errs_delay"][()]
        else:
            errs_delay_new = f["errs_delay"][()]

            invalid = np.isnan(errs_delay)
            errs_delay[invalid] = errs_delay_new[invalid]

styles = {
    "nef": {
        "color": "k",
        "linestyle": "--",
        "marker": '+',
        "markersize": 5,
        "markevery": (0.05, 0.2),
    },
    "biased_xs": {
        "color": utils.blues[0],
        "linestyle": '-',
        "marker": 'x',
        "markersize": 5,
        "markevery": (0.1, 0.2),
    },
    "unbiased_xs": {
        "color": utils.oranges[1],
        "linestyle": '-',
        "marker": '2',
        "markersize": 7,
        "markevery": (0.0, 0.2),
    }
}

titles = {
    "mod_fourier_bartlett":
    "\\textbf{Modified Fourier}\n(Bartlett window; $q = 7$)",
    "mod_fourier_erasure":
    "\\textbf{Modified Fourier}\n(Rectangle window; $q = 7$)",
    "legendre_erasure":
    "\\textbf{Legendre (LDN)}\n(Rectangle window; $q = 7$)",
}

np.random.seed(133)
G = lif_utils.lif_rate
gains, biases, encoders = nonneg_common.mk_ensemble(100,
                                                    d=7,
                                                    max_rates=(100, 200))

colors = ['tab:blue', 'tab:orange', 'tab:green']
fig = plt.figure(figsize=(7.55, 4.0))
gs1 = fig.add_gridspec(1, 3, wspace=0.4, bottom=0.55)
axs = [fig.add_subplot(gs1[0, i]) for i in range(3)]

Ss = []

for i, solver_mode in enumerate(solver_modes):
    j = modes.index("mod_fourier_erasure")
    k = qs.index(7)
    i_neurons = neurons.index(100)

    Es = np.nanmean(errs_delay[i, j, k, i_neurons, :, :, :], axis=-1) * 100.0
    Es = Es.transpose(1, 2, 0).reshape(len(xs_sigma_test), -1)
    E25 = np.nanpercentile(Es, 25, axis=1)
    E50 = np.nanpercentile(Es, 50, axis=1)
    E75 = np.nanpercentile(Es, 75, axis=1)
    color = colors[i]
    axs[0].plot(xs_sigma_test,
                E50,
                **styles[solver_mode],
                linewidth=1.25,
                zorder=100 + i)
    axs[0].fill_between(xs_sigma_test,
                        E25,
                        E75,
                        color=styles[solver_mode]["color"],
                        linewidth=0.0,
                        alpha=0.4)

    W = utils.run_with_cache(run_experiment, solver_mode)
    if W.ndim == 3:
        W = W[:, :, 0]
    print(np.sum(np.square(W)))
    U, S, V = np.linalg.svd(W)
    Ss.append(S / np.max(S))
    if i == 0:
        U, V = -U, -V
    style_all_markers = dict(styles[solver_mode])
    style_all_markers["linestyle"] = ""
    del style_all_markers["markevery"]
    axs[1].plot(np.arange(1, 101), S / np.max(S), **style_all_markers)
    axs[1].set_ylim(1e-4, 2)
    axs[1].set_xlim(0.5, 10.5)
    axs[1].set_xticks(np.arange(1, 11, 1))
    axs[1].set_yscale("log")

    style_no_markers = dict(styles[solver_mode])
    style_no_markers["marker"] = None

    xs_lin = np.linspace(-1, 1, 1001)
    xs1 = np.zeros((len(xs_lin), 7))
    xs1[:, 1] = xs_lin  #r * np.sin(np.pi * xs_lin)
    As1 = G(gains[None, :] * (xs1 @ encoders.T) + biases[None, :])
    axs[2].plot(xs_lin, ((As1 @ W.T) @ np.linalg.pinv(encoders).T)[:, :3],
                **style_no_markers)

    if i == 0:
        axs[2].set_xlim(-1, 1)
        axs[2].set_ylim(-1, 1)
        axs[2].plot(0,
                    0,
                    '+',
                    color='white',
                    zorder=10,
                    markeredgewidth=2,
                    markersize=7)
        axs[2].plot(0, 0, '+', color='k', zorder=11)


axs[0].arrow(3, 20, 0, -5, width=0.1, head_width=0.4, head_length=4, overhang=0.1, color='k', clip_on=False)

axs[0].set_xticks(np.linspace(1, 10, 3, dtype=int))
axs[0].set_xticks(np.linspace(1, 10, 10, dtype=int), minor=True)
axs[0].set_xlabel("Bandwidth $\\rho$ (Hz)")
axs[0].set_ylabel("Mean delay NRMSE (\\%)")

axs[1].plot(np.arange(1, 11), np.mean(Ss, axis=0)[:10], 'k:', lw=0.5, zorder=0)
axs[1].set_xlabel("Singular value index $i$")
axs[1].set_ylabel("Singular value $\\sigma_i$")

utils.annotate(axs[2], 0.5, 0.1, 0.0, 0.5, "$j = 1$")
utils.annotate(axs[2], 0.5, 0.525, 0.0, 0.75, "$j = 2$")
utils.annotate(axs[2], 0.5, -0.25, 0.0, -0.5, "$j = 3$")

axs[2].set_xticks(np.linspace(-1, 1, 3))
axs[2].set_xticks(np.linspace(-1, 1, 5), minor=True)
axs[2].set_yticks(np.linspace(-1, 1, 3))
axs[2].set_yticks(np.linspace(-1, 1, 5), minor=True)
axs[2].set_xlabel("Represented value $m_1(t)$")
axs[2].set_ylabel("Decoded feedback $m_j'(t)$")

axs[0].set_title("\\textbf{Bandwidth sweep}")
axs[0].text(-0.2175, 1.06475, "\\textbf{A}", size=12, transform=axs[0].transAxes)

axs[1].set_title("\\textbf{Feedback weights SVD}")
axs[1].text(-0.26, 1.06475, "\\textbf{B}", size=12, transform=axs[1].transAxes)

axs[2].set_title("\\textbf{Decoded feedback}")
axs[2].text(-0.235, 1.06475, "\\textbf{C}", size=12, transform=axs[2].transAxes)

fig.legend([
    mpl.lines.Line2D([0], [0], **styles["nef"]),
    mpl.lines.Line2D([0], [0], **styles["unbiased_xs"]),
    mpl.lines.Line2D([0], [0], **styles["biased_xs"]),
], ["Standard NEF", "Na\\\"ive sampling", "Uniform activation sampling"],
           loc="upper center",
           bbox_to_anchor=(0.5, 1.02),
           ncol=3)

utils.save(fig)

