files = [
    utils.datafile(f"evaluate_synaptic_weight_computation_{i}.h5")
    for i in range(10)
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
            errs_tuning = f["errs_tuning"][()]
        else:
            invalid = np.isnan(errs_tuning)
            errs_tuning[invalid] = f["errs_tuning"][invalid]

styles = {
    "nef": {
        "color": "k",
        "linestyle": "--",
        "marker": '+',
        "markersize": 5,
    },
    "biased_xs": {
        "color": utils.blues[0],
        "linestyle": '-',
        "marker": 'x',
        "markersize": 5,
    },
    "unbiased_xs": {
        "color": utils.oranges[1],
        "linestyle": '-',
        "marker": '2',
        "markersize": 7,
    }
}

titles = {
    "mod_fourier_bartlett":
    "\\textbf{Modified Fourier}\n(Bartlett window; $q = 7$)",
    "mod_fourier_erasure":
    "\\textbf{Modified Fourier}\n(Rectangle window; $q = 7$)",
    "legendre_erasure": "\\textbf{Legendre (LDN)}\n(Rectangle window; $q = 7$)",
}

colors = ['tab:blue', 'tab:orange', 'tab:green']
fig, axs = plt.subplots(1, 3, figsize=(7.45, 1.75), gridspec_kw={"wspace": 0.1})
for i, solver_mode in enumerate(solver_modes):
    for j, mode in enumerate(modes):
        if solver_mode == "nef" and mode == "non_lindep_cosine":
            continue
        for k, q in enumerate(qs):
            if q != 7:
                continue
            Es = errs_tuning[i, j, k, :, :, 0, :]
            Es = Es.reshape(len(neurons), -1)
            E25 = np.nanpercentile(Es, 25, axis=1)
            E50 = np.nanpercentile(Es, 50, axis=1)
            E75 = np.nanpercentile(Es, 75, axis=1)
            color = colors[i]
            axs[j].errorbar(neurons,
                            E50, ((E50 - E25), (E75 - E50)),
                            capsize=2,
                            **styles[solver_mode],
                            linewidth=1.25,
                            clip_on=False,
                            zorder=100 + i)

        if i == 0:
            axs[j].set_ylim(1e-2, 2)
            axs[j].set_xscale('log')
            axs[j].set_yscale('log')
            axs[j].set_xlim(8, 1200)
            axs[j].set_ylim(1e-2, 1.1)

            if j == 0:
                axs[j].set_ylabel("Tuning error ($\\mathrm{s}^{-1}$)")
            else:
                axs[j].set_yticklabels([])
            axs[j].set_xlabel("Number of neurons $n$")
            axs[j].set_title(titles[mode])

            axs[j].text(0.0 if j > 0 else -0.2375,
                        1.14,
                        "\\textbf{{{}}}".format(chr(ord('A') + j)),
                        size=12,
                        transform=axs[j].transAxes)

fig.legend([
    mpl.lines.Line2D([0], [0], **styles["nef"]),
    mpl.lines.Line2D([0], [0], **styles["unbiased_xs"]),
    mpl.lines.Line2D([0], [0], **styles["biased_xs"]),
], ["Standard NEF", "Na\\\"ive sampling", "Uniform activation sampling"],
           loc="upper center",
           bbox_to_anchor=(0.5, 1.225), ncol=3)

utils.save(fig)

