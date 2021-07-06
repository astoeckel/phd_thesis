import os
import h5py


def i_th_to_color(i_th):
    if i_th < 0:
        i_th = len(ths) + i_th
    if i_th == 0:
        return 'gray'
    elif i_th == 1:
        return utils.oranges[1]
    else:
        return mpl.cm.get_cmap('viridis')(0.7 * (i_th - 2) / (len(ths) - 1))


def plot_and_setup_ax(ax, data, y0, y1, letter=None):
    medianprops = dict(linestyle='-', linewidth=1.0, color=utils.oranges[1])
    meanprops = dict(linestyle='--', linewidth=1.0, color=utils.greens[0])
    boxprops = dict(linewidth=0.75)
    whiskerprops = dict(linewidth=0.75)

    x_labels = ["Clamp"] + ["${:0.2f}$".format(th) for th in ths]

    ax.axhline(0, linestyle=':', color='gray', linewidth=0.5)
    ax.boxplot(
        data.T,
        showmeans=True,
        showfliers=False,
        meanline=True,
        boxprops=boxprops,
        medianprops=medianprops,
        meanprops=meanprops,
        whiskerprops=whiskerprops,
    )
    ax.set_xticks(np.arange(1, len(ths) + 2))
    ax.set_xticklabels(x_labels)
    ax.set_ylim(y0, y1)
    ax.set_yticks(np.arange(y0, y1 + 0.01, 0.2))
    ax.set_yticks(np.arange(y0, y1 + 0.01, 0.1), minor=True)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ys = -0.0875
    ax.plot([0.166],
            ys,
            's',
            color='white',
            clip_on=False,
            zorder=100,
            transform=ax.transAxes)
    for i, i_th in enumerate(np.arange(1, len(ths) + 2)):
        ax.scatter([0.166 * i + 0.5 * 0.166],
                   0,
                   marker='o',
                   color=i_th_to_color(i_th),
                   clip_on=False,
                   zorder=100,
                   transform=ax.transAxes,
                   edgecolor='k',
                   linewidth=0.5)
    ax.set_xlabel("Target threshold current $J_\mathrm{th}$ (nA)", x=0.58)
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    if letter:
        ax.text(-0.265,
                1.15,
                '\\textbf{{{}}}'.format(letter),
                va='top',
                ha='left',
                transform=ax.transAxes,
                size=12)


with h5py.File(
        os.path.join(utils.datafile('subthreshold_relaxation_experiment_auto_reg.h5')),
        'r') as f:
    ths = f["ths"][()]
    ratios = f["ratios"][()]
    errs = f["es_dec"][()]
    errs_cur = f["es_cur"][()]
    sigmas = f["sigmas"][()]

print(errs.shape)

i_ratio_even = np.where(ratios == 0.5)[0][0]
errs_rel = (1.0 - (errs / errs[:, 0][:, None]))
errs_cur_rel = (1.0 - (errs_cur / errs_cur[:, 0][:, None]))
errs_rel_all_funs = errs_rel.transpose(
    (1, 2, 0, 3, 4, 5)).reshape(len(ths) + 2, len(ratios), -1)
errs_cur_rel_all_funs = errs_cur_rel.transpose(
    (1, 2, 0, 3, 4, 5)).reshape(len(ths) + 2, len(ratios), -1)

fig = plt.figure(figsize=(6.1, 4.0))
gs1 = fig.add_gridspec(1,
                       2,
                       left=0.05,
                       right=0.95,
                       top=0.95,
                       bottom=0.55,
                       wspace=0.3)
axs = [fig.add_subplot(gs1[0, i]) for i in range(2)]

plot_and_setup_ax(axs[0],
                  errs_cur_rel_all_funs[1:, i_ratio_even],
                  -0.6,
                  1.0,
                  letter="A")
axs[0].set_title('\\textbf{Post-synaptic current error}', y=1.025)
axs[0].set_ylabel('Reduction in PSC error')

plot_and_setup_ax(axs[1],
                  errs_rel_all_funs[1:, i_ratio_even],
                  -0.4,
                  0.6,
                  letter="B")
axs[1].set_title('\\textbf{Post-population representation error}', y=1.025)
axs[1].set_ylabel('Reduction in representation error')

gs2 = fig.add_gridspec(1,
                       3,
                       left=0.027,
                       right=0.95,
                       top=0.3,
                       bottom=0.05,
                       wspace=0.2)
axs = [fig.add_subplot(gs2[0, i]) for i in range(3)]

for i_fun in range(3):
    for i_th in range(len(ths) + 2):
        data = errs_rel[i_fun, i_th, i_ratio_even].reshape(len(sigmas), -1)
        linestyle = (':' if i_th == 0 else (':' if i_th == 1 else '-'))
        linewidth = 0.5 if i_th == 0 else 1.0
        axs[i_fun].semilogx(sigmas,
                            np.median(data, axis=-1),
                            linestyle=linestyle,
                            color=i_th_to_color(i_th),
                            linewidth=linewidth)
        if i_th > 0:
            axs[i_fun].fill_between(sigmas,
                                    np.percentile(data, 25, axis=-1),
                                    np.percentile(data, 75, axis=-1),
                                    color=i_th_to_color(i_th),
                                    linewidth=0.0,
                                    alpha=0.4)
    axs[i_fun].set_ylim(-0.05, 0.4)
    axs[i_fun].set_xlim(1e-3, 1)
    axs[i_fun].set_title([
        "$\\varphi(x) = x$",
        "$\\varphi(x) = 2x^2 - 1$",
        "$\\varphi(x) = 3x^3 - 1.7x$",
    ][i_fun])
    axs[i_fun].set_yticks(np.arange(0, 0.41, 0.2), minor=False)
    axs[i_fun].set_yticks(np.arange(0, 0.41, 0.1), minor=True)
    axs[i_fun].set_yticks(np.arange(0, 0.41, 0.1), minor=True)

    if i_fun == 0:
        axs[i_fun].set_yticklabels(
            ["{:d}\\%".format(int(x * 100)) for x in np.arange(0, 0.41, 0.2)])
        axs[i_fun].set_ylabel("Repr. error reduction", labelpad=8.5)
    else:
        axs[i_fun].set_yticklabels([])

    axs[i_fun].set_xlabel(
        "Noise standard deviation~$\sigma / a^\\mathrm{max}$")

axs[1].text(0.5,
            1.25,
            "\\textbf{Solution robustness}",
            va="bottom",
            ha="center",
            transform=axs[1].transAxes)
axs[0].text(-0.3,
            1.24,
            '\\textbf{C}',
            va='bottom',
            ha='left',
            transform=axs[0].transAxes,
            size=12)

utils.save(fig)

