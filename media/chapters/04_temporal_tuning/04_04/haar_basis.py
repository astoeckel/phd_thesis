from dlop_ldn_function_bases import *


def plot_basis(B, varname='T', bottom_labels=True, n_plot_bases=6):
    import matplotlib.pyplot as plt
    plt.rcParams['figure.dpi'] = 200

    fig = plt.figure(figsize=(7.5, 2.0))
    gs = fig.add_gridspec(4, 3, wspace=0.25)
    ax = fig.add_subplot(gs[:, 0])

    s = np.percentile(np.abs(B), 95)
    ax.imshow(B,
              interpolation='nearest',
              extent=(0.5, B.shape[1] + 0.5, B.shape[0] + 0.5, 0.5),
              vmin=-s,
              vmax=s,
              cmap='RdBu')
    ax.set_xticks(np.linspace(1, B.shape[1], 5, dtype=np.int))
    ax.set_yticks(np.linspace(1, B.shape[0], 5, dtype=np.int))
#    ax.set_title("$\\mathbf{{{}}}$".format(varname))
    ax.set_ylabel("Row $i$")
    ax.set_aspect(B.shape[1] / B.shape[0])
    if bottom_labels:
        ax.set_xlabel("Column $j$")
    else:
        ax.set_xticklabels([])
    utils.outside_ticks(ax)

    qp = min(n_plot_bases, B.shape[0])
    for i in range(qp):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col + 1])
        ax.axhline(0.0, linestyle=':', color='k', lw=0.5)
        ax.plot(np.arange(1, B.shape[1] + 1),
                B[i].T,
                'o',
                markersize=3,
                markerfacecolor='white',
                markeredgecolor=mpl.cm.get_cmap('inferno')(i / 8),
                markeredgewidth=0.7)
        for j in range(1, B.shape[1] + 1):
            ax.plot([j, j], [0.0, B[i, j - 1]], 'k-', lw=0.5)
        ax.text(1.0,
                1.1,
                f'$W_{{{i}}}$',
                ha="right",
                va="top",
                transform=ax.transAxes)
        ax.set_xlim(0.5, B.shape[1] + 0.5)
        ax.set_ylim(-0.5, 0.5)
        utils.remove_frame(ax)
        if row == 3:
            ax.spines["bottom"].set_visible(True)
            ax.set_xticks(np.linspace(1, B.shape[1], 5, dtype=int))
            ax.set_xlabel("Column $j$")

    return fig


H = mk_haar_basis(32, 32)
fig = plot_basis(H, 'W', n_plot_bases=8)

utils.save(fig)

