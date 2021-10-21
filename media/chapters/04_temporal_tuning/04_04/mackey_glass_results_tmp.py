import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as patches
import scipy.stats

import lmu_utils

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
    "fir":
    dict(
        marker='^',
        color=utils.yellows[1],
        markeredgecolor='k',
        markeredgewidth=0.7,
    ),
    "lti":
    dict(
        marker='o',
        color=utils.blues[0],
        markeredgecolor='k',
        markeredgewidth=0.7,
    ),
    "sdt":
    dict(
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

fig, ax = plt.subplots(figsize=(7.5, 2.5))

N, m, b = errs.shape[0], 2.0, 0.8
for k in range(N):
    errs_fixed = errs[k, 0, ~np.isnan(errs[k, 0])]
    errs_learned = errs[k, 1, ~np.isnan(errs[k, 1])]
    bp1 = ax.boxplot(errs_fixed,
                     showfliers=True,
                     positions=[k * m],
                     widths=[0.5],
                     notch=True,
                     bootstrap=10000,
                     patch_artist=True)
    bp2 = ax.boxplot(errs_learned,
                     showfliers=True,
                     positions=[k * m + b],
                     widths=[0.5],
                     notch=True,
                     bootstrap=10000,
                     patch_artist=True)

    for a in bp1['means'] + bp2['means']:
        a.set_color('black')
        a.set_linestyle((0, (1, 1)))
    for a in bp1['medians'] + bp2['medians']:
        a.set_color('k')
        a.set_linestyle('--')
        a.set_linewidth(0.7)
    for a in bp1['boxes']:
        #a.set_color('tab:blue')
        a.set_color(utils.blues[2])
        a.set_edgecolor('k')
        a.set_linewidth(0.75)
    for a in bp2['boxes']:
        #a.set_color('tab:orange')
        a.set_color(utils.grays[5])
        a.set_edgecolor('k')
        a.set_linewidth(0.75)
    ax.tick_params(axis='x', which='both', length=0)

for i, name in enumerate(basis_names):
    x = (i + 0.45) / (len(basis_names) - 0.1)
    y = 0.0
    ax.plot(x,
            y,
            **MARKER_STYLES[BASIS_TYPES[name]],
            clip_on=False,
            transform=ax.transAxes)



ax.set_xticks(np.arange(N) * m + 0.5 * b)
ax.set_xticklabels([LABEL_MAP[basis_names[i]] for i in range(errs.shape[0])])
ax.set_ylabel('Test error (NRMSE)', y=0.6)
ax.set_title(
    '\\textbf{{Mackey-Glass}} ($\\tau = 30; n_\\mathrm{{pred}} = 15; n_\\mathrm{{trials}} = {}$; $n_\\mathrm{{epochs}} = {}$)'
    .format(errs.shape[-1], N_EPOCHS),
    y=1.1)

for spine in ["left", "bottom"]:
    ax.spines[spine].set_position(("outward", 2.5))
ax.spines["bottom"].set_visible(False)

patch1 = patches.Patch(fc=utils.blues[2], ec='k', linewidth=0.7)
patch2 = patches.Patch(fc=utils.grays[5], ec='k', linewidth=0.7)
ax.legend([patch1, patch2], ["Fixed", "Learned"],
          ncol=2,
          handlelength=0.75,
          handletextpad=0.5,
          bbox_to_anchor=(0.5, 1.15),
          loc='upper center')

ax.set_ylim(0.0, 0.11)

ax.text(-0.06,
        0.0,
        "$\\blacktriangleleft$ Better",
        va="baseline",
        ha="right",
        rotation=90,
        transform=ax.transAxes)

utils.save(fig)

# Print the table

tbl = np.zeros((len(basis_names), 8))
tbl[:, (0, 4)] = np.nanmean(100 * errs[:, :], axis=-1)
tbl[:, (1, 5)] = np.nanmedian(100 * errs[:, :], axis=-1)
tbl[:, (2, 6)] = np.nanpercentile(100 * errs[:, :], 25, axis=-1)
tbl[:, (3, 7)] = np.nanpercentile(100 * errs[:, :], 75, axis=-1)
tbl = np.round(tbl, 2)

s = ""
for i in range(tbl.shape[0]):
    s += "\t\\sym" + BASIS_TYPES[basis_names[i]].upper() + "~" + LABEL_MAP[
        basis_names[i]] + " &\n"
    for j in range(tbl.shape[1]):
        c = -np.sort(-tbl[:, j])[::-1]
        idx = np.argmin(np.abs(c - tbl[i, j]))
        s += "\t"
        if idx < 3:
            s += " \\cellcolor{{CornflowerBlue!{}}}{{".format(75 - 25 * idx)
        s += "{:0.2f}\\%".format(tbl[i, j])
        if idx < 3:
            s += "}"
        if j + 1 < tbl.shape[1]:
            s += " &\n"
        else:
            s += " \\\\\n"
print(s + "\n\n")

# Produce the significance table

n_basis = len(basis_names)
tbl = np.zeros((n_basis, 2 * n_basis))
for i in range(n_basis):
    for j in range(n_basis):
        for k in range(2):
            D1, D2 = errs[i, k], errs[j, k]
            D1, D2 = D1[~np.isnan(D1)], D2[~np.isnan(D2)]
            _, tbl[i, k * n_basis + j] = scipy.stats.kstest(D1, D2)

s = ""
for i in range(tbl.shape[0]):
    s += "\t\\sym" + BASIS_TYPES[basis_names[i]].upper() + "~" + LABEL_MAP[
        basis_names[i]] + " & (" + str(i + 1) + ") &\n"
    for j in range(tbl.shape[1]):
        p = tbl[i, j]
        stars = "\\sigC"
        if p > 0.001:
            stars = "\\sigB"
        if p > 0.01:
            stars = "\\sigA"
        if p > 0.05:
            stars = ""
        s += "\t"
        s += stars
        if j + 1 < tbl.shape[1]:
            s += " &\n"
        else:
            s += " \\\\\n"

print(s)

