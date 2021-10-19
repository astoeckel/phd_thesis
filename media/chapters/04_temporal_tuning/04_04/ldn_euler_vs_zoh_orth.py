import dlop_ldn_function_bases as bases

fig, axs = plt.subplots(2,
                        6,
                        figsize=(7.45, 3.25),
                        gridspec_kw={
                            "wspace": 0.3,
                            "hspace": 0.4,
                            "height_ratios": [3, 3],
                        })


def plot_HHT(ax, H):
    ax.imshow(H @ H.T,
              vmin=-1,
              vmax=1,
              cmap='RdBu',
              extent=[0.5, q + 0.5, q + 0.5, 0.5])
    ax.set_xticks([1, 10, 20])
    ax.set_yticks([1, 10, 20])
    utils.outside_ticks(ax)


q = 20
for i, N in enumerate(np.geomspace(20, 1000, axs.shape[1], dtype=int)):
    H = bases.mk_ldn_basis(q, N)
    Hp = bases.mk_ldn_basis_euler(q, N)

    _, S, _ = np.linalg.svd(H)
    _, Sp, _ = np.linalg.svd(Hp)
    S /= np.max(S)
    Sp /= np.max(Sp)

    plot_HHT(axs[0, i], H)
    plot_HHT(axs[1, i], Hp)
    if i > 0:
        axs[0, i].set_yticklabels([])
        axs[1, i].set_yticklabels([])
    else:
        axs[0, i].set_ylabel("Row $j$")
        axs[1, i].set_ylabel("Row $j$")
    axs[0, i].set_xlabel("Column $j$")
    axs[1, i].set_xlabel("Column $j$")

    axs[0, i].set_title(f"$N = {N}$")

    axs[0, i].plot(0.1,
                   0.9,
                   'o',
                   color=utils.grays[0],
                   markersize=8,
                   transform=axs[0, i].transAxes)
    axs[0, i].text(0.1,
                   0.89,
                   f"\\textbf{{{i + 1}}}",
                   color="white",
                   transform=axs[0, i].transAxes,
                   ha="center",
                   va="center")
    axs[0, i].text(0.95,
                   0.85,
                   "$\\Sigma = {:0.1f}$".format(np.sum(S)),
                   ha="right",
                   va="baseline",
                   color="black",
                   transform=axs[0, i].transAxes,
                   bbox={
                       "pad": 0.25,
                       "color": "white",
                   })

    axs[1, i].plot(0.1,
                   0.9,
                   'o',
                   color=utils.grays[0],
                   markersize=8,
                   transform=axs[1, i].transAxes)
    axs[1, i].text(0.1,
                   0.89,
                   f"\\textbf{{{i + 1}}}",
                   color="white",
                   transform=axs[1, i].transAxes,
                   ha="center",
                   va="center")
    axs[1, i].text(0.95,
                   0.85,
                   "$\\Sigma = {:0.1f}$".format(np.sum(Sp)),
                   ha="right",
                   va="baseline",
                   color="black",
                   transform=axs[1, i].transAxes,
                   bbox={
                       "pad": 0.25,
                       "color": "white",
                   })

#    s = np.sqrt(N)
#    axs[2, i].plot(s * H[:4].T, 'k-', lw=0.5)
#    axs[2, i].plot(s * Hp[:4].T, clip_on=False)
#    axs[2, i].plot(s * H[:4].T, ':', color='white', lw=0.5)
#    axs[2, i].set_ylim(-3, 3)
#    axs[2, i].set_xticks([])
#    axs[2, i].set_yticks([])
#    axs[2, i].spines["left"].set_visible(False)
#    axs[2, i].spines["bottom"].set_visible(False)

fig.text(
    0.5,
    0.925,
    "$\mat H \mat H^T$ \\textbf{with zero-order hold discretisation} ($q = 20$)",
    va="baseline",
    ha="center")
fig.text(0.08, 0.925, "\\textbf{A}", size=12, va="baseline", ha="center")

fig.text(
    0.5,
    0.415,
    "$\mat H' (\mat H')^T$ \\textbf{with Euler discretisation} ($q = 20$)",
    va="baseline",
    ha="center")
fig.text(0.08, 0.415, "\\textbf{B}", size=12, va="baseline", ha="center")

utils.save(fig)

