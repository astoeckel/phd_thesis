import h5py

with h5py.File(utils.datafile('nlif_parameter_optimisation_comparison.h5'),
               'r') as f:
    params = f['params'][()]
    errs = f['errs'][()]
    smpls = f['smpls'][()]

c1 = utils.blues[0]
c2 = utils.oranges[1]

titles = [
    "Two-compartment LIF",
    "Three-compartment LIF",
    "Four-compartment LIF",
]
fig, axs = plt.subplots(2,
                        3,
                        figsize=(7.45, 3.0),
                        gridspec_kw={
                            "wspace": 0.25,
                            "hspace": 0.25,
                            "height_ratios": [1.0, 0.5],
                        })
epochs = np.arange(errs.shape[-1])
for j in range(2):
    for i in range(3):
        data = errs[i, 0, 0, :, 1]
        axs[j, i].semilogy(epochs, np.nanmedian(data, axis=0), color=c1)
        axs[j, i].fill_between(epochs,
                               np.percentile(data, 25, axis=0),
                               np.percentile(data, 75, axis=0),
                               color=c1,
                               linewidth=0,
                               alpha=0.5)
        axs[j, i].fill_between(epochs,
                               np.percentile(data, 1, axis=0),
                               np.percentile(data, 99, axis=0),
                               color=c1,
                               linewidth=0,
                               alpha=0.25)

        data = errs[i, 1, 0, :, 1]
        axs[j, i].semilogy(epochs, np.nanmedian(data, axis=0), color=c2)
        axs[j, i].fill_between(epochs,
                               np.percentile(data, 25, axis=0),
                               np.percentile(data, 75, axis=0),
                               color=c2,
                               linewidth=0,
                               alpha=0.5)
        axs[j, i].fill_between(epochs,
                               np.percentile(data, 1, axis=0),
                               np.percentile(data, 99, axis=0),
                               color=c2,
                               linewidth=0,
                               alpha=0.25)


        axs[j, i].semilogy(epochs,
                           np.nanmedian(errs[i, 0, 0, :, 0], axis=0),
                           '--', linewidth=0.5,
                           color=c1)
        axs[j, i].semilogy(epochs,
                           np.nanmedian(errs[i, 1, 0, :, 0], axis=0),
                           '--', linewidth=0.5, color=c2)


        axs[j, i].scatter([0], np.nanmedian(errs[i, 1, 0, :, 0, 0], axis=0), marker='o', clip_on=False, s=7, color='none', edgecolor='k', zorder=10)

        deco_style = dict(color='k', linewidth=0.5, linestyle=':')

        axs[j, i].set_ylim(2e-3, 1e-1)
        if j == 1:
            axs[j, i].set_xlim(0, 50)
            axs[j, i].set_xticks(np.arange(0, 51, 20))
            axs[j, i].set_xticks(np.arange(0, 51, 10), minor=True)
            axs[j, i].set_xlabel('Number of epochs')
            #axs[j, i].spines["right"].set_visible(True)
            #axs[j, i].spines["top"].set_visible(True)
        else:
            axs[j, i].set_title("\\textbf{" + titles[i] + "}")
            axs[j, i].set_xlim(0, epochs[-1])
            axs[j, i].set_xticks(np.arange(0, epochs[-1] + 1, 200))
            axs[j, i].set_xticks(np.arange(0, epochs[-1] + 1, 100), minor=True)
            axs[j, i].axvline(50, **deco_style)

            axs[j, i].plot([0.0, 0.0], [-0.13, -0.3],
                           **deco_style,
                           clip_on=False,
                           transform=axs[j, i].transAxes)

            axs[j, i].plot([0.1, 0.1], [0.0, -0.13],
                           **deco_style,
                           clip_on=False,
                           transform=axs[j, i].transAxes)
            axs[j, i].plot([0.1, 1.0], [-0.13, -0.175],
                           **deco_style,
                           clip_on=False,
                           transform=axs[j, i].transAxes)
            axs[j, i].plot([1.0, 1.0], [-0.175, -0.666],
                           **deco_style,
                           clip_on=False,
                           transform=axs[j, i].transAxes)

        if i == 0:
            axs[j, i].set_ylabel('NRMSE')

fig.legend(
    [
     mpl.lines.Line2D([0], [0], color=c1),
     mpl.lines.Line2D([0], [0], color=c2),
     mpl.lines.Line2D([0], [0], color='k', linestyle="--", linewidth=0.5),
    ],
    ["Soft trust-region", "SGD (Adam)", "Training error"],
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 1.05),
)

utils.save(fig)

