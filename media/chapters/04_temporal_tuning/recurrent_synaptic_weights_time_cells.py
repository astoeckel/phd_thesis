import nengo

import os, sys

sys.path.append(
    os.path.join(utils.libdir, '..', 'chapters', '04_temporal_tuning'))

import evaluate_synaptic_weight_computation as exp

exp.N_TRAIN_SMPLS = 100
exp.N_REPEAT_TEST = 1
exp.XS_SIGMA = 3.0
exp.XS_SIGMA_TEST = [3.0]
exp.NEURONS = [120]

titles = {
    "mod_fourier_bartlett":
    "\\textbf{Modified Fourier}\n(Bartlett window; $q = 7$)",
    "mod_fourier_erasure":
    "\\textbf{Modified Fourier}\n(Rectangle window; $q = 7$)",
    "legendre_erasure": "\\textbf{Legendre (LDN)}\n(Rectangle window; $q = 7$)",
}


def run_experiment(mode):
    ts_test = np.arange(0, 10.0, 1e-3)
    xs_test = np.zeros_like(ts_test)
    xs_test[ts_test > 1.0] = 10.0
    xs_test[ts_test > 1.1] = 0.0

    idcs = [
        exp.SOLVER_MODES.index("nef"),
        exp.MODES.index(mode),
        list(exp.QS).index(5), 0, 0
    ]

    ts, xs_test, xs_test_flt, xs_test_decs, As_test, As_test_flt, Es_tuning, Es_delay = exp.execute_single(
        idcs, return_test_data=True, inject_xs_test=xs_test)
    xs_test_decs = np.array(xs_test_decs)

    return ts, xs_test, xs_test_flt, xs_test_decs, As_test, As_test_flt, Es_tuning, Es_delay


fig, axs = plt.subplots(2, 3, figsize=(7.4, 3.8), gridspec_kw={"wspace": 0.7})

for j, mode in enumerate(
    ["mod_fourier_bartlett", "mod_fourier_erasure", "legendre_erasure"]):

    ts, _, _, _, As_test, _, Es_tuning, Es_delay = utils.run_with_cache(
        run_experiment, mode)
    As_test_flt = nengo.Lowpass(tau=0.2).filtfilt(As_test, dt=1e-3)
    As_test_flt = As_test_flt[1100:2100]

    valid_neuron_idcs = np.where(np.max(As_test_flt, axis=0) > 1e-3)[0]

    N_neurons = 73
    rng = np.random.RandomState(38181)
    neuron_idcs = rng.choice(valid_neuron_idcs, N_neurons, replace=False)

    As = As_test_flt[:, neuron_idcs]
    As_max = np.max(As, axis=0)
    As_max_idcs = np.argmax(As, axis=0)
    idcs = np.argsort(As_max_idcs)
    As_max_ts = ts[As_max_idcs[idcs]]

    axs[0, j].imshow((As[:, idcs] / As_max[idcs]).T,
                     extent=(0, 1.0, As.shape[1] + 0.5, 0.5),
#                     cmap='inferno',
                     cmap='inferno',
                     interpolation='nearest')
    axs[0, j].set_aspect('auto')
    axs[0, j].set_xticklabels([])
    axs[0, j].plot(As_max_ts,
                np.arange(0.5, N_neurons + 0.5),
                linewidth=3,
                color='white',
                alpha=1)
    axs[0, j].plot(As_max_ts,
                np.arange(0.5, N_neurons + 0.5),
                linestyle=':',
                linewidth=1,
                color='k',
                alpha=1)
    axs[0, j].set_ylabel("Neuron index $i$")
    axs[0, j].set_xticks(np.linspace(0, 1, 3))
    axs[0, j].set_xticks(np.linspace(0, 1, 5), minor=True)
    axs[0, j].set_yticks([x if (i == 0) or (i == 4) else (x // 5) * 5 for i, x in enumerate(np.linspace(1, 73, 5, dtype=int))])
    utils.outside_ticks(axs[0, j])

    As_len = np.linalg.norm(As, axis=1)
    As_norm = As / As_len[:, None]
    axs[1, j].imshow(As_norm @ As_norm.T,
                     cmap='inferno',
                     origin='upper',
                     extent=(0, 1, 1, 0))
    ts = np.linspace(0, 1, 2)
    axs[1, j].plot(ts, ts, linewidth=3, color='white', alpha=1)
    axs[1, j].plot(ts, ts, linestyle=':', linewidth=1, color='k', alpha=1)
    axs[1, j].set_aspect('auto')
    axs[1, j].set_xticks(np.linspace(0, 1, 3))
    axs[1, j].set_xticks(np.linspace(0, 1, 5), minor=True)
    axs[1, j].set_yticks(np.linspace(0, 1, 3))
    axs[1, j].set_yticks(np.linspace(0, 1, 5), minor=True)
    axs[1, j].set_xlabel("Time $t - t_0$ (s)")
    axs[1, j].set_ylabel("Time $t - t_0$ (s)")
    utils.outside_ticks(axs[1, j])


    axs[0, j].set_title(titles[mode])
    axs[0, j].text(-0.3,
                   1.17,
                   "\\textbf{{{}}}".format(chr(ord('A') + j)),
                   size=12,
                   transform=axs[0, j].transAxes)

fig.align_labels(axs)

utils.save(fig)

