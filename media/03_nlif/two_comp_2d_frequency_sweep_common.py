import h5py

def plot_files(files, letter=None, mark_sigmas=[]):
    params_keys = []
    errs = None
    for file in files:
        with h5py.File(utils.datafile(file), 'r') as f:
            network = ("network" in f) and bool(f["network"][()])
            sigmas = f['sigmas'][()]
            params_keys = params_keys + f['params_keys'][()].split(b'\n')
            if errs is None:
                errs = np.array(f['errs'][()])
            else:
                errs = np.concatenate((errs, f['errs'][()]))

        print(params_keys)

    def lighten(c, f):
        f = 1.0 + f;
        return np.clip(c * np.array((f, f, f, 1.0)), 0.0, 1.0)

    colors = ['#000000', mpl.cm.get_cmap('viridis')(0.3), mpl.cm.get_cmap('viridis')(0.6), mpl.cm.get_cmap('viridis')(0.9), utils.oranges[1]]

    styles = {
        "linear": {
            "label": "$H_\\mathrm{cur}$",
            "color": "k",
            "linewidth": 2.0,
            "marker": 'o',
            "markersize": 4,
            "markevery": (0, 5),
            "markeredgewidth": 0.5,
            "zorder": 9,
        },
        "linear_2d": {
            "label": "$H_\\mathrm{cur}$ (two layers)",
            "color": colors[4],
            "linewidth": 2.0,
            "marker": 'o',
            "markersize": 4,
            "markevery": (0, 5),
            "markeredgewidth": 0.5,
            "zorder": 9,
        },
        "gc50_no_noise": {
            "label": "$H_\\mathrm{cond}$ (for $c_{12} = 50\,\mathrm{nS}$)",
            "color": colors[1],
            "linewidth": 1.5,
            "marker": 's',
            "markersize": 4,
            "markevery": (0, 6),
            "markeredgewidth": 0.5,
        },
        "gc100_no_noise": {
            "label": "$H_\\mathrm{cond}$ (for $c_{12} = 100\,\mathrm{nS}$)",
            "color": colors[2],
            "linewidth": 1.5,
            "marker": 'd',
            "markersize": 4,
            "markevery": (2, 6),
            "markeredgewidth": 0.5,
        },
        "gc200_no_noise": {
            "label": "$H_\\mathrm{cond}$ (for $c_{12} = 200\,\mathrm{nS}$)",
            "color": colors[3],
            "linewidth": 1.5,
            "marker": 'v',
            "markersize": 4,
            "markevery": (4, 6),
            "markeredgewidth": 0.5,
        },
    #    "gc50_noisy": {
    #        "label": "$H_\\mathrm{cond}$ (with noise, $g_\\mathrm{C} = 50\,\mathrm{nS}$)",
    #        "color": colors[1],
    #        "linewidth": 1.0,
    #    },
    #    "gc100_noisy": {
    #        "label": "$H_\\mathrm{cond}$ (with noise, $g_\\mathrm{C} = 100\,\mathrm{nS}$)",
    #        "color": colors[2],
    #        "linewidth": 1.0,
    #    },
    #    "gc200_noisy": {
    #        "label": "$H_\\mathrm{cond}$ (with noise, $g_\\mathrm{C} = 200\,\mathrm{nS}$)",
    #        "color": colors[3],
    #        "linewidth": 1.0,
    #    },
    }
    lbls = 1 / sigmas

    def do_plot(col, labels=True, title=None, ax=None, linewidth=1.0, linestyle='-'):
        for i_param in range(0, len(params_keys)):
            key = str(params_keys[i_param], 'utf-8')
            if key in styles:
                style = dict(**styles[key])
                if (not labels) and ("label" in style):
                    del style["label"]
                    del style["marker"]
                if "linewidth" in style:
                    style["linewidth"] *= linewidth
                data = np.median(errs[i_param, :, :, col], axis=1)
                style_1 = dict(style)
                style_2 = dict(style)
                style_2["label"] = None
                style_2["zorder"] = 10
                ax.plot(lbls, data, markeredgecolor="k", linestyle=linestyle, **style_1)
                ax.plot(lbls, data, markeredgecolor="k", linestyle='', **style_2)
                if (key == 'linear' or key == 'gc50_no_noise' or key == 'linear_2d') and linestyle == '-':
                    perc_25 = np.percentile(errs[i_param, :, :, col], 25, axis=1)
                    perc_75 = np.percentile(errs[i_param, :, :, col], 75, axis=1)
                    ax.fill_between(lbls, perc_25, perc_75, color=style["color"], alpha=0.4, lw=0)
    #    ax.set_title(title)
        ax.set_xlabel('Spatial lowpass filter coefficient $1/\\mathrm{\\sigma}$')
        ax.set_ylabel('NRMSE')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(np.min(lbls), np.max(lbls))
        if network:
            ax.set_ylim(3e-2, 1)
        else:
            ax.set_ylim(5e-3, 1)

    fig, ax = plt.subplots(figsize=(7.45, 3))
    #ax.set_title("\\textbf{Current function $f_\\sigma(x_1, x_2)$ superthreshold decoding error}")
    do_plot(1, labels=False, linewidth=0.5, ax=ax, linestyle='--')
    do_plot(0, ax=ax)
    ax.legend(loc="lower right")
    if letter:
        ax.text(-0.075, 1.03, "\\textbf{{{}}}".format(letter), va="top", ha="left", size=12, transform=ax.transAxes)

    for x in mark_sigmas:
        ax.axvline(x, color='grey', linewidth=0.5, linestyle=':', zorder=-1)

    utils.save(fig)

