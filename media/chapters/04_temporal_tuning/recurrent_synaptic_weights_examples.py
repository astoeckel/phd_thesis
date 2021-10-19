import os, sys

sys.path.append(
    os.path.join(utils.libdir, '..', 'chapters', '04_temporal_tuning'))

import evaluate_synaptic_weight_computation as exp

exp.N_TRAIN_SMPLS = 100
exp.N_REPEAT_TEST = 1
exp.XS_SIGMA = 3.0
exp.XS_SIGMA_TEST = [3.0]

titles = {
    "mod_fourier_bartlett":
    "\\textbf{Modified Fourier}\n(Bartlett window; $q = 7$)",
    "mod_fourier_erasure":
    "\\textbf{Modified Fourier}\n(Rectangle window; $q = 7$)",
    "legendre_erasure": "\\textbf{Legendre (LDN)}\n(Rectangle window; $q = 7$)",
}


def run_experiment(mode):
    idcs = [
        exp.SOLVER_MODES.index("nef"),
        exp.MODES.index(mode),
        list(exp.QS).index(7),
        list(exp.NEURONS).index(1000), 0
    ]

    ts, xs_test, xs_test_flt, xs_test_decs, _, _, Es_tuning, Es_delay = exp.execute_single(
        idcs, return_test_data=True)
    xs_test_decs = np.array(xs_test_decs)

    return ts, xs_test, xs_test_flt, xs_test_decs, Es_tuning, Es_delay


fig, axs = plt.subplots(1, 3, figsize=(7.3, 1.5), gridspec_kw={"wspace": 0.2})

for j, mode in enumerate(
    ["mod_fourier_bartlett", "mod_fourier_erasure", "legendre_erasure"]):

    ts, xs_test, xs_test_flt, xs_test_decs, Es_tuning, Es_delay = utils.run_with_cache(
        run_experiment, mode)

    for i in range(exp.N_DELAYS_TEST):
        color = mpl.cm.get_cmap('inferno')(1.0 - i / (exp.N_DELAYS_TEST - 1))
        axs[j].plot(ts, xs_test_decs[i], color=color, zorder=-i)
    axs[j].plot(ts, xs_test_flt, 'k--', lw=0.7)
    axs[j].text(1.0,
                1.0,
                "$E = {:0.1f}\\%$".format(np.mean(Es_delay) * 100.0, ),
                bbox={"color": "white"},
                ha="right",
                va="top",
                transform=axs[j].transAxes)
    axs[j].set_xlim(0, 10)
    axs[j].set_xlabel("Time $t$ (s)")
    axs[j].set_title(titles[mode])
    axs[j].set_ylim(-1.25, 1.25)
    axs[j].set_yticks(np.linspace(-1, 1, 5))
    axs[j].set_yticks(np.linspace(-1, 1, 9), minor=True)

    axs[j].plot([1.65, 2.65], [1.0, 1.0], 'k-', linewidth=2, solid_capstyle='butt')
    axs[j].text(2.15, 1.05, '$\\theta$', size=8, ha="center", va="bottom")

    if j == 0:
        axs[j].set_ylabel("Decoded value $x(t - \\theta')$")
    else:
        axs[j].set_yticklabels([])

    axs[j].text(0.0 if j > 0 else -0.276,
                1.19,
                "\\textbf{{{}}}".format(chr(ord('A') + j)),
                size=12,
                transform=axs[j].transAxes)

utils.save(fig)

