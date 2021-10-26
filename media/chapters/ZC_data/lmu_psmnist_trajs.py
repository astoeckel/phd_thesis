import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as patches
import scipy.stats

LABEL_MAP = {
    "ldn": "LDN",
    "mod_fourier": "Mod.~Fourier",
    "fourier": "Fourier",
    "cosine": "Cosine",
    "haar": "Haar",
    "dlop": "DLOP",
    "random": "Random",
    "eye": "Identity",
}

BASIS_SORT_ORDER = {name: i for i, name in enumerate(LABEL_MAP.keys())}

STYLE_MAP = {
    "ldn": {
        "color": "tab:blue"
    },
    "dlop": {
        "color": "tab:purple"
    },
    "fourier": {
        "color": "tab:red"
    },
    "cosine": {
        "color": "tab:orange"
    },
    "haar": {
        "color": "tab:green"
    },
    "random": {
        "color": "black"
    }
}

BASIS_TYPES = {
    "ldn": "lti",
    "mod_fourier": "lti",
    "fourier": "sdt",
    "cosine": "sdt",
    "haar": "sdt",
    "dlop": "fir",
    "random": "fir",
}

MARKER_STYLES = {
    "fir": dict(
        marker='^',
        color=utils.yellows[1],
        markeredgecolor='k',
        markeredgewidth=0.7,
    ),
    "lti": dict(
        marker='o',
        color=utils.blues[0],
        markeredgecolor='k',
        markeredgewidth=0.7,
    ),
    "sdt": dict(
        marker='h',
        color=utils.oranges[1],
        markeredgecolor='k',
        markeredgewidth=0.7,
    ),
}

#
# Convoluted code for loading the data. Handles merging of multiple results
# files.
#

errs, trajs, qs, basis_names = None, None, None, None
for fn in [
    utils.datafile("manual/chapters/04_temporal_tuning/psmnist_results_single_2021_01_26_04_08_28.npz"),
    utils.datafile("manual/chapters/04_temporal_tuning/psmnist_results_single_2021_01_26_04_23_24.npz"),
    utils.datafile("manual/chapters/04_temporal_tuning/psmnist_results_single_2021_01_26_04_25_49.npz"),
    utils.datafile("manual/chapters/04_temporal_tuning/psmnist_results_single_2021_01_26_06_48_40.npz"),
	utils.datafile("lmu_psmnist.npz"),
]:
    data = np.load(fn)

    basis_names_loc, qs_loc = list(data['basis_names']), data['qs']
    print(fn, basis_names_loc)

    # Copy the number of qs
    if qs is None:
        N_QS = len(qs_loc)
        N_SEEDS = data["errs"].shape[-1]
        N_EPOCHS = data["trajs"].shape[-2]
        qs = qs_loc
        print("qs =", qs)
    else:
        assert qs_loc == qs
        assert N_SEEDS == data["errs"].shape[-1]
        assert N_EPOCHS == data["trajs"].shape[-2]

    if basis_names is None:
        basis_names = basis_names_loc
        basis_map = np.arange(len(basis_names_loc), dtype=int)
    else:
        basis_map = np.zeros(len(basis_names_loc), dtype=int)
        for i, name in enumerate(basis_names_loc):
            if name in basis_names:
                basis_map[i] = basis_names.index(name)
            else:
                basis_map[i] = len(basis_names)
                basis_names.append(name)
                errs = np.concatenate((errs, np.zeros((1, N_QS, 2, N_SEEDS))),
                                      axis=0)
                trajs = np.concatenate(
                    (trajs, np.zeros((1, N_QS, 2, N_SEEDS, N_EPOCHS, 2))),
                    axis=0)

    # Copy the data
    errs_loc, trajs_loc = data['errs'], data['trajs']
    if errs is None:
        errs, trajs = errs_loc, trajs_loc
    else:
        for i in range(len(basis_names_loc)):
            valid_errs = errs_loc[i] != 0.0
            valid_trajs = trajs_loc[i] != 0.0
            errs[basis_map[i], valid_errs] = errs_loc[i][valid_errs]
            trajs[basis_map[i], valid_trajs] = trajs_loc[i][valid_trajs]

# Select the first Q (there can be only one)
errs, trajs = errs[:, 0], trajs[:, 0]

# Sort the data
errs_sorted = np.zeros_like(errs)
trajs_sorted = np.zeros_like(trajs)
basis_names_sorted = sorted(basis_names, key=lambda x: BASIS_SORT_ORDER[x])
for i, name in enumerate(basis_names_sorted):
    j = basis_names.index(name)
    errs_sorted[i] = errs[j]
    trajs_sorted[i] = trajs[j]

errs, trajs, basis_names = errs_sorted, trajs_sorted, basis_names_sorted

#
# Actual plotting code
#

fig, axs = plt.subplots(7, 2, figsize=(7.85, 3.7))

for i in range(7):
    for k in range(2):
        errs_val = trajs[i, k, :, :, 1].T
        errs_trn = trajs[i, k, :, :, 0].T

        errs_val_25 = np.percentile(errs_val, 10, axis=-1)
        errs_trn_25 = np.percentile(errs_trn, 10, axis=-1)
        errs_val_50 = np.median(errs_val, axis=-1)
        errs_trn_50 = np.median(errs_trn, axis=-1)
        errs_val_75 = np.percentile(errs_val, 90, axis=-1)
        errs_trn_75 = np.percentile(errs_trn, 90, axis=-1)

        epochs = np.arange(1, 101)

        axs[i, k].plot(epochs, errs_val_50, '-', lw=1.0, color=utils.blues[0])
        axs[i, k].fill_between(epochs, errs_val_25, errs_val_75, color=utils.blues[0], alpha=0.5, lw=0.0)

        axs[i, k].plot(np.arange(1, 101), errs_trn_50, '--', color=utils.oranges[1], lw=0.7)
        axs[i, k].fill_between(epochs, errs_trn_25, errs_trn_75, color=utils.oranges[1], alpha=0.5, lw=0.0)

        m = np.argmin(errs_val_50)
        axs[i, k].plot(m + 1, errs_val_50[m], 'o', color=utils.blues[0], markersize=4)

        axs[i, k].set_ylim(0, 0.2)
        axs[i, k].spines["left"].set_visible(False)
        axs[i, k].set_yticklabels([])
        axs[i, k].set_xlim(0, 100)
        axs[i, k].set_yticks([0, 0.05, 0.1, 0.15, 0.2])
        for ytick in axs[i, k].get_yticks():
            axs[i, k].axhline(ytick, linestyle=':', lw=0.5, color=(0.8, 0.8, 0.8), zorder=-100, clip_on=False)

        axs[i, k].text(1.0, 1.0, f"{LABEL_MAP[basis_names[i]]}", ha="right", va="top", transform=axs[i, k].transAxes)

        if i == 0:
            axs[i, k].set_title(["\\textbf{Fixed FIR filters}", "\\textbf{Learned FIR filters}"][k], y=0.5)

        if i < 6:
            axs[i, k].set_xticklabels([])
        else:
            axs[i, k].set_xlabel("Training epoch")

fig.legend([
    mpl.lines.Line2D([0], [0], color=utils.blues[0], lw=1.0),
    mpl.lines.Line2D([0], [0], color=utils.oranges[1], lw=0.7, linestyle='--'),
], ["Validation error", "Training error"], loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=2)

utils.save(fig)

