import h5py

with h5py.File(utils.datafile('nlif_parameter_optimisation_comparison.h5'),
               'r') as f:
    params = f['params'][()]
    errs = f['errs'][()]
    smpls = f['smpls'][()]

c1 = utils.blues[0]
c2 = utils.oranges[1]
c3 = utils.greens[0]

titles = [
    "Two-compartment LIF",
    "Three-compartment LIF",
    "Four-compartment LIF",
]


def do_plot(axs, errs, y1=1e-1):
    epochs = np.arange(errs.shape[-1])

    for j in range(2):
        for i in range(3):
            ax = axs[j, i]

            def plot_single(o, color):
                ax.semilogy(epochs, np.nanmedian(errs[i, o, :, 1], axis=0), color=color)
                ax.fill_between(epochs,
                                       np.nanpercentile(errs[i, o, :, 1], 25, axis=0),
                                       np.nanpercentile(errs[i, o, :, 1], 75, axis=0),
                                       color=color,
                                       linewidth=0,
                                       alpha=0.5)
#                ax.fill_between(epochs,
#                                       np.nanpercentile(errs[i, o, :, 1], 1, axis=0),
#                                       np.nanpercentile(errs[i, o, :, 1], 99, axis=0),
#                                       color=color,
#                                       linewidth=0,
#                                       alpha=0.25)

                last_idx = np.argmax(np.all(np.isnan(errs[i, o, :, 1]),
                                            axis=0))
                if last_idx > 0:
                    for k in range(2):
                        xs = [epochs[last_idx - 1], epochs[-1]]
                        last_dist = errs[i, o, :, k, last_idx - 1]
                        ax.semilogy(
                            xs, [np.nanmedian(last_dist),
                                 np.nanmedian(last_dist)],
                            linestyle=(0, (1, 2)) if k == 0 else (0, (1, 1)),
                            color=color)
                        if k == 1:
                            ax.fill_between(xs,
                                                   np.nanpercentile(last_dist,
                                                                    25,
                                                                    axis=0),
                                                   np.nanpercentile(last_dist,
                                                                    75,
                                                                    axis=0),
                                                   color=color,
                                                   linewidth=0,
                                                   alpha=0.5)

                ax.semilogy(epochs,
                                   np.nanmedian(errs[i, o, :, 0], axis=0),
                                   '--',
                                   linewidth=1.0,
                                   color=color)



            plot_single(2, c3)
            plot_single(1, c2)
            plot_single(0, c1)

            ax.scatter([0],
                              np.nanmedian(errs[i, :, :, 0, 0]),
                              marker='o',
                              clip_on=False,
                              s=10,
                              color='none',
                              edgecolor='k',
                              linewidth=1.0,
                              zorder=10)

            deco_style = dict(color='k', linewidth=0.7, linestyle=':')

            ax.set_ylim(5e-3, y1)
            if j == 1:
                ax.set_xlim(0, 40)
                ax.set_xticks(np.arange(0, 41, 20))
                ax.set_xticks(np.arange(0, 41, 10), minor=True)
                ax.set_xlabel('Number of epochs')
                #ax.spines["right"].set_visible(True)
                #ax.spines["top"].set_visible(True)
            else:
                #                ax.set_title("\\textbf{" + titles[i] + "}")
                ax.set_title(titles[i])
                ax.set_xlim(0, 400)
                ax.set_xticks(np.arange(0, 400 + 1, 200))
                ax.set_xticks(np.arange(0, 400 + 1, 100), minor=True)
                ax.axvline(40, **deco_style)

                ax.plot([0.0, 0.0], [-0.13, -0.3],
                               **deco_style,
                               clip_on=False,
                               transform=ax.transAxes)

                ax.plot([0.1, 0.1], [0.0, -0.13],
                               **deco_style,
                               clip_on=False,
                               transform=ax.transAxes)
                ax.plot([0.1, 1.0], [-0.13, -0.175],
                               **deco_style,
                               clip_on=False,
                               transform=ax.transAxes)
                ax.plot([1.0, 1.0], [-0.175, -0.666],
                               **deco_style,
                               clip_on=False,
                               transform=ax.transAxes)

            if i == 0:
                ax.set_ylabel('NRMSE')


fig = plt.figure(figsize=(7.45, 6.0))
gs1 = fig.add_gridspec(2,
                       3,
                       wspace=0.25,
                       hspace=0.25,
                       height_ratios=[1.0, 0.5],
                       top=0.95,
                       bottom=0.575)
gs2 = fig.add_gridspec(2,
                       3,
                       wspace=0.25,
                       hspace=0.25,
                       height_ratios=[1.0, 0.5],
                       top=0.425,
                       bottom=0.05)

axs = np.array([[fig.add_subplot(gs1[i, j]) for j in range(3)]
                for i in range(2)])
do_plot(axs, errs[:, :, 0,
                  0, :, :])  # first parameter set, no random initialisation

axs = np.array([[fig.add_subplot(gs2[i, j]) for j in range(3)]
                for i in range(2)])
do_plot(axs, errs[:, :, 0, 1, :, :],
        y1=1.0)  # first parameter set, no random initialisation

fig.text(0.0685, 1.0, '\\textbf{A}', va='baseline', ha='left', size=12)
fig.text(0.5125,
         1.0,
         '\\textbf{Initialisation with original estimate}',
         va='baseline',
         ha='center')

fig.legend(
    [
        mpl.lines.Line2D([0], [0], color=c1),
        mpl.lines.Line2D([0], [0], color=c3),
        mpl.lines.Line2D([0], [0], color=c2),
        mpl.lines.Line2D([0], [0], color='k', linestyle="--", linewidth=1.0),
    ],
    ["SQP with soft trust-region", "L-BFGS-B", "SGD (Adam)", "Training error"],
    loc="upper center",
    ncol=4,
    bbox_to_anchor=(0.5, 1.075),
)

fig.text(0.0685, 0.475, '\\textbf{B}', va='baseline', ha='left', size=12)
fig.text(0.5125,
         0.475,
         '\\textbf{Random initialisation}',
         va='baseline',
         ha='center')

utils.save(fig)

