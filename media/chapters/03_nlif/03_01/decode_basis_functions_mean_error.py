import h5py

Es = []
with h5py.File(utils.datafile("decode_basis_functions.h5")) as f:
    N_NEURONS = f["N_NEURONS"][()]
    N_DIMS = f["N_DIMS"][()]
    for i in range(N_DIMS):
        Es.append(f[f"E{i}"][()])

fig, axs = plt.subplots(1, 3, figsize=(7.4, 2), gridspec_kw={
    "wspace": 0.2
})
CMAP = 'viridis'
for i in range(N_DIMS):
    # Select the 10 best basis functions
    Es_median = np.median(Es[i].reshape(Es[i].shape[0], -1), axis=-1)
    best_idcs = np.argsort(Es_median)
    N_BESTS = np.unique(np.geomspace(1, Es[i].shape[0], 20, dtype=int))[::-1]
    print(N_BESTS)
    for j, N_BEST in enumerate(N_BESTS):
        color = mpl.cm.get_cmap(CMAP)(j / (len(N_BESTS) - 1))
        #data = np.percentile(Es[i][best_idcs[:N_BEST]].transpose(1, 0, 2).reshape(Es[i].shape[1], -1), 50, axis=-1)
        data = np.mean(Es[i][best_idcs[:N_BEST]].transpose(1, 0, 2).reshape(Es[i].shape[1], -1), axis=-1)
        axs[i].fill_between(N_NEURONS, data, np.zeros_like(N_NEURONS), linewidth=0.0, color=color)
        axs[i].semilogx(N_NEURONS, data, color='k')
        axs[i].set_xlim(N_NEURONS[0], N_NEURONS[-1])

    axs[i].set_ylim(0, 1)
    #if i == 0:
    #    axs[i].set_ylabel("Decodable basis functions $d$")
    #axs[i].set_yticks(np.linspace(0, 10, 11), minor=True)
    axs[i].set_title(f"\\textbf{{{(i + 1)}D population}} ($\\ell = {i + 1}$)")
    axs[i].set_xlabel("Number of neurons $n$")
    utils.outside_ticks(axs[i])

#axs[1].legend(
#    [
#        mpl.patches.Rectangle((0.0, 0.0), 1.0, 1.0, edgecolor='k', linewidth=0.5, facecolor=mpl.cm.get_cmap(CMAP)(i / (N_THS - 1))) for i in range(N_THS)
#    ],
#    [f"${(th * 100):0.1f}\\%$" for th in THS],
#    ncol=N_THS,
#    loc="upper center",
#    bbox_to_anchor=(0.5, 1.3),
#)

#fig.text(0.5, 1.075, "NRMSE threshold", va="bottom", ha="center")

utils.save(fig)
