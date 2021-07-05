import h5py

with h5py.File(utils.datafile('nef_noise_visualisation.h5'), 'r') as f:
    NS = f["NS"][()]
    DIMS = f["DIMS"][()]
    errs_id = f["errs_id"][()]
    errs_prod = f["errs_prod"][()]

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.0))

def plot_errs(ax1, ax2, errs, err_tar):
    for j, d in enumerate(DIMS):
        ys = np.median(errs[:, j], axis=-1)
        ys25 = np.percentile(errs[:, j], 25, axis=-1)
        ys75 = np.percentile(errs[:, j], 75, axis=-1)

        axs[0].fill_between(NS, ys25, ys75, alpha=0.25)

        xs_log = np.log10(NS)
        ys_log = np.log10(ys)
        p = np.polyfit(xs_log[4*len(NS)//5:], ys_log[4*len(NS)//5:], 1)
        print(d, p)
        axs[0].plot(NS, np.power(10, p[1] + p[0] * xs_log), ':k', linewidth=0.5)
        axs[0].plot(NS, ys)

        N_tar = (err_tar - p[1]) / p[0]
        axs[0].plot([np.power(10.0, N_tar)], [np.power(10.0, err_tar)], 'k+')
        axs[1].plot([d], [np.power(10.0, N_tar)], '+')

#plot_errs(axs[0], axs[1], errs_id, -2)
plot_errs(axs[0], axs[1], errs_prod, -1)

axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_xlabel('Number of neurons $n$')
axs[0].set_ylabel('RMSE $E$')

axs[1].set_yscale('log')

utils.save(fig)

