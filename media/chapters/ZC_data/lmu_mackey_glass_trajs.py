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
    "eye": "fir",
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

errs, trajs, basis_names = None, None, None
for fn in [
        utils.datafile("lmu_mackey_glass_0.npz"),
        utils.datafile("lmu_mackey_glass_1.npz"),
        utils.datafile("lmu_mackey_glass_2.npz"),
#        utils.datafile("lmu_mackey_glass_3.npz"),
#        utils.datafile("lmu_mackey_glass_4.npz"),
#        utils.datafile("lmu_mackey_glass_5.npz"),
#        utils.datafile("lmu_mackey_glass_6.npz"),
#        utils.datafile("lmu_mackey_glass_7.npz"),
#        utils.datafile("lmu_mackey_glass_8.npz"),
#        utils.datafile("lmu_mackey_glass_9.npz"),
]:
    data = np.load(fn)

    basis_names_loc = list(data['basis_names'])
    print(fn, basis_names_loc)

    # Copy some information
    if errs is None:
        N_SEEDS = data["errs"].shape[-1]
        N_EPOCHS = data["trajs"].shape[-2]
    else:
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
                errs = np.concatenate((errs, np.zeros((1, 2, 2, N_SEEDS))),
                                      axis=0)
                trajs = np.concatenate(
                    (trajs, np.zeros((1, 2, 2, N_SEEDS, N_EPOCHS, 2))), axis=0)

    # Copy the data
    errs_loc, trajs_loc = data['errs'], data['trajs']
    if errs is None:
        errs, trajs = errs_loc, trajs_loc
    else:
        for i in range(len(basis_names_loc)):
            valid_errs = ~np.isnan(errs_loc[i])
            valid_trajs = ~np.isnan(trajs_loc[i])
            errs[basis_map[i], valid_errs] = errs_loc[i][valid_errs]
            trajs[basis_map[i], valid_trajs] = trajs_loc[i][valid_trajs]

# Only select the results for the extended window
errs_p, trajs_p = errs[:, 1], trajs[:, 1]

# Do not use the extended window for the learned and random initialisations
i_random = basis_names.index("random")
errs_p[:, 1] = errs[:, 0, 1] # Override the data for "learn = True"
errs_p[i_random] = errs[i_random, 0]
trajs_p[:, 1] = trajs[:, 0, 1] # Override the data for "learn = True"
trajs_p[i_random] = trajs[i_random, 0]

errs, trajs = errs_p, trajs_p

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

fig, axs = plt.subplots(8, 2, figsize=(7.85, 3.7))

for i in range(8):
    for k in range(2):
        errs_val = np.log10(trajs[i, k, :, :, 1].T)
        errs_trn = np.log10(trajs[i, k, :, :, 0].T)

        errs_val_25 = np.nanpercentile(errs_val, 10, axis=-1)
        errs_trn_25 = np.nanpercentile(errs_trn, 10, axis=-1)
        errs_val_50 = np.nanmedian(errs_val, axis=-1)
        errs_trn_50 = np.nanmedian(errs_trn, axis=-1)
        errs_val_75 = np.nanpercentile(errs_val, 90, axis=-1)
        errs_trn_75 = np.nanpercentile(errs_trn, 90, axis=-1)

        epochs = np.arange(1, N_EPOCHS + 1)

        axs[i, k].plot(epochs, errs_val_50, '-', lw=1.0, color=utils.blues[0])
        axs[i, k].fill_between(epochs, errs_val_25, errs_val_75, color=utils.blues[0], alpha=0.5, lw=0.0)

        axs[i, k].plot(np.arange(1, N_EPOCHS + 1), errs_trn_50, '--', color=utils.oranges[1], lw=0.7)
        axs[i, k].fill_between(epochs, errs_trn_25, errs_trn_75, color=utils.oranges[1], alpha=0.5, lw=0.0)

        m = np.argmin(errs_val_50)
        axs[i, k].plot(m + 1, errs_val_50[m], 'o', color=utils.blues[0], markersize=4)

        axs[i, k].set_ylim(-3.5, -1.5)
        axs[i, k].spines["left"].set_visible(False)
        axs[i, k].set_yticklabels([])
        axs[i, k].set_xlim(0, 200)
        axs[i, k].set_yticks([-3.5, -3, -2.5, -2, -1.5])
        for ytick in axs[i, k].get_yticks():
            axs[i, k].axhline(ytick, linestyle=':', lw=0.5, color=(0.8, 0.8, 0.8), zorder=-100, clip_on=False)

        axs[i, k].text(1.0, 1.0, f"{LABEL_MAP[basis_names[i]]}", ha="right", va="top", transform=axs[i, k].transAxes)

        if i == 0:
            axs[i, k].set_title(["\\textbf{Fixed FIR filters}", "\\textbf{Learned FIR filters}"][k], y=0.5)

        if i < 7:
            axs[i, k].set_xticklabels([])
        else:
            axs[i, k].set_xlabel("Training epoch")

fig.legend([
    mpl.lines.Line2D([0], [0], color=utils.blues[0], lw=1.0),
    mpl.lines.Line2D([0], [0], color=utils.oranges[1], lw=0.7, linestyle='--'),
], ["Validation error", "Training error"], loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=2)

utils.save(fig)

