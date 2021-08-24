import h5py

with h5py.File(utils.datafile('nlif_weight_optimisation_comparison.h5'),
               'r') as f:
    errs = f['errs'][()]

c1 = utils.blues[0]
c2 = utils.greens[0]
c3 = utils.oranges[1]

titles = [
    "Two-compartment LIF",
    "Three-compartment LIF",
    "Four-compartment LIF",
]

# Compute correction factors for the NRMSE; use the test RMS for the training
# samples

def mk_smpls(res):
    xs1 = np.linspace(-1, 1, res)
    xs2 = np.linspace(-1, 1, res)
    xss1, xss2 = np.meshgrid(xs1, xs2)
    return xss1, xss2

def rms(x):
    return np.sqrt(np.mean(np.square(x)))

functions = [
    lambda x1, x2: 0.5 * (x1 + x2),
    lambda x1, x2: 0.5 * (x1 * x2 + 1.0),
]

for i, f in enumerate(functions):
    rms_train = rms(f(*mk_smpls(16)))
    rms_test = rms(f(*mk_smpls(100)))
    corr = rms_train / rms_test
    print(f"Function {i + 1} NRMSE correction factor = {corr}")
    errs[i, :, :, 0] *= corr

def do_plot(axs, errs, y1=1e-1):
    epochs = np.arange(errs.shape[-1])

    for j in range(2):
        for i in range(3):
            ax = axs[j, i]

            def plot_single(o, color):
                ax.semilogy(epochs,
                                   np.nanmedian(errs[i, o, :, 1], axis=0),
                                   color=color)
                ax.fill_between(epochs,
                                       np.nanpercentile(errs[i, o, :, 1],
                                                        25,
                                                        axis=0),
                                       np.nanpercentile(errs[i, o, :, 1],
                                                        75,
                                                        axis=0),
                                       color=color,
                                       linewidth=0,
                                       alpha=0.5)

                last_idx = np.argmax(np.all(np.isnan(errs[i, o, :, 1]),
                                            axis=0))
                if last_idx > 0:
                    for k in range(2):
                        xs = [epochs[last_idx - 1], epochs[-1]]
                        last_dist = errs[i, o, :, k, last_idx - 1]
                        ax.semilogy(
                            xs, [np.nanmedian(last_dist),
                                 np.nanmedian(last_dist)],
                            linestyle=(0, (2, 4)) if k == 0 else (0, (1, 1)),
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

            deco_style = dict(color='k', linewidth=0.7, linestyle=':')

            if j == 1:
                ax.set_xlim(0, 100)
                ax.set_xticks(np.arange(0, 101, 25))
                ax.set_xticks(np.arange(0, 101, 12.5), minor=True)
            else:
                ax.set_xlim(0, 1500)
                ax.set_xticks(np.arange(0, 1501, 500))
                ax.set_xticks(np.arange(0, 1501, 250), minor=True)
                ax.set_title(titles[i])

                ax.axvline(100, **deco_style)

                ax.plot([0.0, 0.0], [-0.13, -0.3],
                               **deco_style,
                               clip_on=False,
                               transform=ax.transAxes)

                ax.plot([0.066, 0.066], [0.0, -0.13],
                               **deco_style,
                               clip_on=False,
                               transform=ax.transAxes)
                ax.plot([0.066, 1.0], [-0.13, -0.175],
                               **deco_style,
                               clip_on=False,
                               transform=ax.transAxes)
                ax.plot([1.0, 1.0], [-0.175, -0.666],
                               **deco_style,
                               clip_on=False,
                               transform=ax.transAxes)

            ax.set_ylim(y1, 2)

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
                  0, :, :], y1=4e-3)


axs = np.array([[fig.add_subplot(gs2[i, j]) for j in range(3)]
                for i in range(2)])
do_plot(axs, errs[:, :, 1,
                  0, :, :], y1=1e-2)


fig.text(0.0685, 1.0, '\\textbf{A}', va='baseline', ha='left', size=12)
fig.text(0.5125,
         1.0,
         '\\textbf{Addition}; $J(x_1, x_2) = \\frac{1}2 (x_1 + x_2)$, $\\lambda = 10^{-1}$',
         va='baseline',
         ha='center')

fig.legend(
    [
        mpl.lines.Line2D([0], [0], color=c1),
        mpl.lines.Line2D([0], [0], color=c2),
        mpl.lines.Line2D([0], [0], color=c3),
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
         '\\textbf{Multiplication}; $J(x_1, x_2) = \\frac{1}2 (x_1 x_2 + 1)$, $\\lambda = 10^{-3}$',
         va='baseline',
         ha='center')

utils.save(fig)

