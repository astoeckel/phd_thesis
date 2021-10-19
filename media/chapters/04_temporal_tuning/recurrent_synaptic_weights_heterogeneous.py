files = [
    utils.datafile(
        f"evaluate_synaptic_weight_computation_heterogeneous_{i}.h5")
    for i in range(10)
]

errs_tuning, errs_delay = [None] * 2
for i, fn in enumerate(files):
    print(f"Loading {fn}...")
    with h5py.File(fn, "r") as f:
        if i == 0:
            modes = json.loads(f.attrs["modes"])
            qs = json.loads(f.attrs["qs"])
            tau_sigmas = json.loads(f.attrs["tau_sigmas"])
            n_tau_sigmas = len(tau_sigmas)
            tau_sigmas = np.linspace(
                0.0, 0.1,
                n_tau_sigmas)  # I accidentially saved tau_sigmas as integers

            errs_solver = f["errs_solver"][()]
            errs_solver_ref = f["errs_solver_ref"][()]

            errs_tuning = f["errs_tuning"][()]
            errs_tuning_ref = f["errs_tuning_ref"][()]
        else:
            errs_solver_new = f["errs_solver"][()]
            errs_solver_ref_new = f["errs_solver_ref"][()]
            errs_tuning_new = f["errs_tuning"][()]
            errs_tuning_ref_new = f["errs_tuning_ref"][()]

            invalid = np.isnan(errs_solver)
            errs_solver[invalid] = errs_solver_new[invalid]

            invalid = np.isnan(errs_solver_ref)
            errs_solver_ref[invalid] = errs_solver_ref_new[invalid]

            invalid = np.isnan(errs_tuning)
            errs_tuning[invalid] = errs_tuning_new[invalid]

            invalid = np.isnan(errs_tuning_ref)
            errs_tuning_ref[invalid] = errs_tuning_ref_new[invalid]


# N_MODES, N_QS, N_TAU_SIGMAS, N_REPEAT, N_NEURONS (solver)
# N_MODES, N_QS, N_TAU_SIGMAS, N_REPEAT, N_REPEAT_TEST (tuning)

print(errs_solver.shape)

colors = [utils.blues[0], utils.oranges[1], utils.greens[0]]
styles = [
    {
        "color": colors[0],
        "marker": '+',
        "markersize": 5,
    },
    {
        "color": colors[1],
        "marker": 'x',
        "markersize": 5,
    },
    {
        "color": colors[2],
        "marker": '2',
        "markersize": 7,
    },
]

fig, axs = plt.subplots(1,
                        5,
                        figsize=(7.525, 2.0),
                        gridspec_kw={
                            "hspace": 0.7,
                            "wspace": 0.2,
                            "width_ratios": [4, 0.25, 4, 4, 4],
                        },
                        squeeze=False)
utils.remove_frame(axs[0, 1])

for idx, i_mode in enumerate([modes.index("mod_fourier_erasure")
                              ]):  #, modes.index("non_lindep_cosine")]):
    ax_solv = axs[idx, 0]
    axs_tune = axs[idx, 2:]

    for i_q, q in enumerate(qs):
        E_solv = errs_solver[i_mode, i_q].reshape(n_tau_sigmas, -1)
        E_solv_ref = errs_solver_ref[i_mode, i_q].reshape(n_tau_sigmas, -1)
        E_tune = errs_tuning[i_mode, i_q].reshape(n_tau_sigmas, -1)
        E_tune_ref = errs_tuning_ref[i_mode, i_q].reshape(n_tau_sigmas, -1)

        E_solv25 = np.nanpercentile(E_solv, 25, axis=-1)
        E_solv_ref25 = np.nanpercentile(E_solv_ref, 25, axis=-1)
        E_tune25 = np.nanpercentile(E_tune, 25, axis=-1)
        E_tune_ref25 = np.nanpercentile(E_tune_ref, 25, axis=-1)

        E_solv50 = np.nanmedian(E_solv, axis=-1)
        E_solv_ref50 = np.nanmedian(E_solv_ref, axis=-1)
        E_tune50 = np.nanmedian(E_tune, axis=-1)
        E_tune_ref50 = np.nanmedian(E_tune_ref, axis=-1)

        E_solv75 = np.nanpercentile(E_solv, 75, axis=-1)
        E_solv_ref75 = np.nanpercentile(E_solv_ref, 75, axis=-1)
        E_tune75 = np.nanpercentile(E_tune, 75, axis=-1)
        E_tune_ref75 = np.nanpercentile(E_tune_ref, 75, axis=-1)

        #        print(E_solv25, E_solv50, E_solv75)
        #        print(E_tune25, E_tune50, E_tune75)


        def plot_single_line(ax, Es, style):
            style_white_marker = dict(style)
            style_white_marker["color"] = "white"
            style_white_marker["markeredgewidth"] = 2.5
            style_white_marker["markersize"] *= 1.2
            style_white_marker["linewidth"] = 0.0

            ax.plot(tau_sigmas * 1e3,
                    Es,
                    **style,
                    lw=1.2,
                    clip_on=False,
                    zorder=1)
#            ax.plot(tau_sigmas * 1e3,
#                    Es,
#                    #**style_white_marker,
#                    clip_on=False,
#                    zorder=12)

            style_no_line = dict(style)
            style_no_line["linewidth"] = 0.0
            ax.plot(tau_sigmas * 1e3,
                    Es,
                    **style_no_line,
                    clip_on=False,
                    zorder=13)

        plot_single_line(ax_solv, E_solv50 * 1e3, styles[i_q])
        ax_solv.fill_between(tau_sigmas * 1e3,
                             E_solv25 * 1e3,
                             E_solv75 * 1e3,
                             color=colors[i_q],
                             lw=0.0,
                             alpha=0.4)
        ax_solv.plot(tau_sigmas * 1e3,
                     E_solv_ref50 * 1e3,
                     '--',
                     color=colors[i_q],
                     lw=0.7,
                     clip_on=False)

        ax_solv.set_xlabel(
            "$\\sigma_\\tau$ (ms)")

        plot_single_line(axs_tune[i_q], E_tune50 * 1e2, styles[i_q])
        axs_tune[i_q].fill_between(tau_sigmas * 1e3,
                             E_tune25 * 1e2,
                             E_tune75 * 1e2,
                             color=colors[i_q],
                             lw=0.0,
                             alpha=0.4)

        axs_tune[i_q].plot(tau_sigmas * 1e3,
                     E_tune_ref50 * 1e2,
                     '--',
                     color=colors[i_q],
                     lw=0.7,
                     clip_on=False)
        axs_tune[i_q].fill_between(tau_sigmas * 1e3,
                             E_tune_ref25 * 1e2,
                             E_tune_ref75 * 1e2,
                             color=colors[i_q],
                             lw=0.0,
                             alpha=0.4)

        axs_tune[i_q].set_xlabel(
            "$\\sigma_\\tau$ (ms)")

        axs_tune[i_q].set_xlim(0, 100)
        axs_tune[i_q].set_xticks(np.arange(0, 101, 50))
        axs_tune[i_q].set_xticks(np.arange(0, 101, 25), minor=True)
        axs_tune[i_q].set_ylabel("NRMSE (\\%)")
        axs_tune[i_q].set_ylim(0, 20)
        if i_q != 0:
            axs_tune[i_q].set_yticklabels([])
            axs_tune[i_q].set_ylabel("")

        axs_tune[i_q].text(0.05, 0.975, f"$q = {q}$", transform=axs_tune[i_q].transAxes, ha="left", va="top")

        ax_solv.set_xlim(0, 100)

    ax_solv.set_xticks(np.arange(0, 101, 50))
    ax_solv.set_xticks(np.arange(0, 101, 25), minor=True)
    ax_solv.set_ylabel("RMSE")
    ax_solv.set_ylim(0, None)



axs[0, 0].set_title("\\textbf{Solver loss}", y=1.02)
axs[0, 0].text(-0.235,
               1.075,
               "\\textbf{A}",
               size=12,
               ha="left",
               va="baseline",
               transform=axs[0, 0].transAxes)
axs[0, 3].set_title("\\textbf{Tuning error}", y=1.02)
axs[0, 2].text(-0.9,
               1.075,
               "\\textbf{B}",
               size=12,
               ha="left",
               va="baseline",
               transform=axs[0, 1].transAxes)

fig.legend([
    mpl.lines.Line2D([0], [0], **styles[0]),
    mpl.lines.Line2D([0], [0], **styles[1]),
    mpl.lines.Line2D([0], [0], **styles[2]),
    mpl.lines.Line2D([0], [0], color='k', linestyle='--', lw=0.7),
    mpl.patches.Patch(color='grey', linewidth=0.0),
], [
    "$q = 3$",
    "$q = 5$",
    "$q = 7$",
    "Baseline",
    "25th/75th percentile",
], ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15))


fig.align_labels(axs)
utils.save(fig)

