import h5py

colors = [
    '#000000',
    mpl.cm.get_cmap('viridis')(0.3),
    mpl.cm.get_cmap('viridis')(0.6),
    mpl.cm.get_cmap('viridis')(0.9), utils.oranges[1]
]

styles = {
    "linear": {
        "label": "$H_\\mathrm{cur}$",
        "color": "k",
        "linewidth": 2.0,
        "marker": 'o',
        "markersize": 4,
        "markevery": (0, 5),
        "markeredgewidth": 0.5,
    },
    "linear_2d": {
        "label": "$H_\\mathrm{cur}$ (two layers)",
        "color": colors[4],
        "linewidth": 2.0,
        "marker": 'o',
        "markersize": 4,
        "markevery": (0, 5),
        "markeredgewidth": 0.5,
    },
    "gc50_no_noise": {
        "label": "$H_\\mathrm{cond}$ ($c_{12} = 50\,\mathrm{nS}$)",
        "color": colors[1],
        "linewidth": 1.5,
        "marker": 's',
        "markersize": 4,
        "markevery": (0, 6),
        "markeredgewidth": 0.5,
    },
    "gc100_no_noise": {
        "label": "$H_\\mathrm{cond}$ ($c_{12} = 100\,\mathrm{nS}$)",
        "color": colors[2],
        "linewidth": 1.5,
        "marker": 'd',
        "markersize": 4,
        "markevery": (2, 6),
        "markeredgewidth": 0.5,
    },
    "gc200_no_noise": {
        "label": "$H_\\mathrm{cond}$ ($c_{12} = 200\,\mathrm{nS}$)",
        "color": colors[3],
        "linewidth": 1.5,
        "marker": 'v',
        "markersize": 4,
        "markevery": (4, 6),
        "markeredgewidth": 0.5,
    },
    "gc50_noisy": {
        "label":
        "with noise model",
        "color": colors[1],
        "linewidth": 1.0,
        "linestyle": "--",
    },
    "gc100_noisy": {
        "label":
        "with noise model",
        "color": colors[2],
        "linewidth": 1.0,
        "linestyle": "--",
    },
    "gc200_noisy": {
        "label":
        "with noise model",
        "color": colors[3],
        "linewidth": 1.0,
        "linestyle": "--",
    },
}

with h5py.File(utils.datafile("two_comp_2d_regularisation_sweep.h5"),
               "r") as f:
    regs = f["regs"][()]
    params_keys = str(f["params_keys"][()], "utf-8").split("\n")
    errs = f["errs"][()]

fig, axs = plt.subplots(1, 2, figsize=(7.35, 3.0), gridspec_kw={
    "wspace": 0.3,
})

optimal_points = {}

for k, ax in enumerate(axs.flat):
    j = 1 - k
    for i, key in enumerate(params_keys):
        style = dict(styles[key])
        if "linestyle" in style:
            del style["linestyle"]
        style_1 = dict(style)
        style_2 = dict(style)
        style_2["label"] = None
        style_2["zorder"] = 100

        Es = np.median(errs[i, :, :, j], axis=-1)

        ax.loglog(regs, Es, markeredgecolor="k", clip_on=False, linestyle='-', **style_1)
        ax.loglog(regs, Es, markeredgecolor="k", clip_on=False, linestyle='', **style_2)

        imin = np.argmin(Es)
        ax.scatter(regs[imin], Es[imin], marker='+', s=40, color='white', zorder=101, linewidth=3.0)
        ax.scatter(regs[imin], Es[imin], marker='+', s=20, color='k', zorder=102, linewidth=1.5)
        ax.plot([regs[imin], regs[imin]], [1e-2, Es[imin]], 'k--', linewidth=0.5)
        optimal_points[(key, [True, False][j])] = regs[imin]
    if k == 0:
        ax.legend(ncol=(len(styles) + 1) // 2,
                  loc="upper center",
                  bbox_to_anchor=(1.15, 1.35))
    ax.set_ylabel('Median NRMSE $E$')
    ax.set_xlabel('Regularisation factor $\\lambda$')
    ax.set_xlim(1e-3, 1e3)
    ax.set_ylim(1e-2, 5e-1)
    ax.set_title("\\textbf{{{}}}".format(
        ["With subthreshold relaxation",
         "Without subthreshold relaxation"][j]))
    ax.text(-0.175, 1.019, "\\textbf{{{}}}".format(chr(ord('A') + k)), size=12, va="bottom", ha="left", transform=ax.transAxes)

import pprint
pprint.PrettyPrinter(indent=4).pprint(optimal_points)

utils.save(fig)

