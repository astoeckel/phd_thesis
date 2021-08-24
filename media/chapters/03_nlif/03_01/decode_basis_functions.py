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
CMAP = 'Purples'
N_THS = 7
THS = np.logspace(-3, 0, N_THS)[::-1]
for i in range(N_DIMS):
    for j, th in enumerate(THS):
        color = mpl.cm.get_cmap(CMAP)(j / (N_THS - 1))
        data = np.mean(np.sum(Es[i] < th, axis=0), axis=1)
        axs[i].fill_between(N_NEURONS, data, np.zeros_like(N_NEURONS), linewidth=0.0, color=color)
        axs[i].semilogx(N_NEURONS, data, linewidth=0.7, color='k')
    axs[i].set_xlim(N_NEURONS[0], N_NEURONS[-1])
    axs[i].set_ylim(0, 10)
    if i == 0:
        axs[i].set_ylabel("Decodable basis functions $d$")
    axs[i].set_yticks(np.linspace(0, 10, 11), minor=True)
    axs[i].set_title(f"\\textbf{{{(i + 1)}D population}} ($\\ell = {i + 1}$)")
    axs[i].set_xlabel("Number of neurons $n$")
    utils.outside_ticks(axs[i])

axs[1].legend(
    [
        mpl.patches.Rectangle((0.0, 0.0), 1.0, 1.0, edgecolor='k', linewidth=0.5, facecolor=mpl.cm.get_cmap(CMAP)(i / (N_THS - 1))) for i in range(N_THS)
    ],
    [f"${(th * 100):0.1f}\\%$" for th in THS],
    ncol=N_THS,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.3),
)

fig.text(0.5, 1.075, "NRMSE threshold", va="bottom", ha="center")

utils.save(fig)
