with h5py.File(utils.datafile("evaluate_bases_delays_freq_sweep.h5"),
               "r") as f:
    try:
        BASES = json.loads(f.attrs["bases"])
        WINDOWS = json.loads(f.attrs["windows"])
    except TypeError:
        BASES = list(f.attrs["bases"])
        WINDOWS = list(f.attrs["windows"])
    FREQS = list(f.attrs["freqs"])
    THETAS = list(f.attrs["thetas"])

    errs = f["errs"][()]

fig, axs = plt.subplots(1,
                        3,
                        figsize=(7.375, 1.5),
                        gridspec_kw={"wspace": 0.5})

colors = [
    utils.blues[0],
    utils.oranges[1],
    utils.blues[0],
    utils.reds[0],
]

STYLES = {
    "fourier": {
        "color": colors[0],
        "linestyle": "--",
        "marker": "+",
        "markersize": 4,
        "markeredgecolor": "k",
        "markevery": (np.linspace(0.0, 0.2, 5)[0], 0.2),
    },
    "cosine": {
        "color": colors[1],
        "marker": "o",
        "markersize": 4,
        "markeredgecolor": "k",
        "markeredgewidth": 0.7,
        "markevery": (np.linspace(0.0, 0.2, 5)[1], 0.2),
    },
    "mod_fourier": {
        "color": colors[2],
        "marker": "x",
        "markersize": 4,
        "markeredgecolor": "k",
        "markevery": (np.linspace(0.0, 0.2, 5)[2], 0.2),
    },
    "legendre": {
        "color": colors[3],
        "marker": "d",
        "markersize": 4,
        "markeredgecolor": "k",
        "markeredgewidth": 0.7,
        "markevery": (np.linspace(0.0, 0.2, 5)[3], 0.2),
    },
}

for i_wnd, window in enumerate(["optimal", "bartlett", "erasure"]):
    y_max_total = 0.0
    ax = axs[i_wnd]
    for i_basis, basis in enumerate(
        ["fourier", "cosine", "mod_fourier", "legendre"]):
        if not ((basis in BASES) and (window in WINDOWS)):
            continue

        j_wnd = WINDOWS.index(window)
        j_basis = BASES.index(basis)

        mean_errs = np.mean(errs[j_basis, j_wnd], axis=1)

        Es = np.median(mean_errs, axis=-1)
        Es_25 = np.percentile(mean_errs, 25, axis=-1)
        Es_75 = np.percentile(mean_errs, 75, axis=-1)
        ax.plot(FREQS,
                Es,
                label=["Fourier", "Cosine", "Modified Fourier",
                       "Legendre"][i_basis],
                **STYLES[basis])
        ax.fill_between(FREQS,
                        Es_25,
                        Es_75,
                        alpha=0.5,
                        color=STYLES[basis]["color"],
                        linewidth=0.0)

        style_no_line = dict(STYLES[basis])
        style_no_line["linestyle"] = "none"
        ax.plot(FREQS, Es, **style_no_line, zorder=100)

    ax.set_xlim(FREQS[0], FREQS[-1])
    ax.set_xscale("log")
    ax.set_ylim(1e-3, 1)
    ax.set_yscale("log")
    ax.set_xlabel("Cutoff frequency $f$ (Hz)")
    ax.set_ylabel("Mean NRMSE $E$")

    ax.set_title("\\textbf{{{}}}".format([
        "Optimal rectangle window", "Approx.~Bartlett window",
        "Approx.~rectangle window"
    ][i_wnd]))
    ax.text(-0.295,
            1.08,
            "\\textbf{{{}}}".format(chr(ord("A") + i_wnd)),
            size=12,
            transform=ax.transAxes)

    if i_wnd == 0:
        ax.legend(loc="upper center", bbox_to_anchor=(1.9, 1.45), ncol=4)

utils.save(fig)


