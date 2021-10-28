import h5py
import scipy.signal


def learning_error(xs_tar, xs_out, bin_width=50.0, dt=1e-3):
    ts = np.arange(xs_tar.shape[0]) * dt
    N_bin = int(bin_width / dt + 1e-9)
    N_bins = int(xs_tar.shape[0] / N_bin)
    N_smpls = N_bin * N_bins
    bins_tar = xs_tar[:N_smpls].reshape(N_bins, N_bin)
    bins_out = xs_out[:N_smpls].reshape(N_bins, N_bin)
    bins_ts = ts[:N_smpls].reshape(N_bins, N_bin)
    rmss = np.sqrt(np.mean(np.square(bins_tar), axis=1))
    rmses = np.sqrt(np.mean(np.square(bins_tar - bins_out), axis=1))
    return np.mean(bins_ts, axis=1), rmses / rmss


with h5py.File(utils.datafile("pendulum_adaptive_filter.h5"), "r") as f:
    xs_in = f["xs_in"][()]
    xs_tar = f["xs_tar"][()]
    xs_out_both = f["xs_out_both"][()]
    xs_err_both = f["xs_err_both"][()]
    xs_out_no_taus = f["xs_out_no_taus"][()]
    xs_err_no_taus = f["xs_err_no_taus"][()]
    xs_out_no_phis = f["xs_out_no_phis"][()]
    xs_err_no_phis = f["xs_err_no_phis"][()]

N = xs_in.shape[0]
dt = 1e-3
ts = np.arange(N) * dt
T = N * dt
T1 = np.power(10.0, np.ceil(np.log10(T - 1.0)))
SS = int(N / 1000)

fig, axs = plt.subplots(1,
                        4,
                        figsize=(7.35, 2.0),
                        gridspec_kw={
                            "wspace": 0.5,
                            "width_ratios": [2, 1, 1, 2]
                        })


def plot_filtered(ax, errs, w=20.0):
    ts_gauss = np.linspace(-w,  w, int(np.ceil(2.0 * w / dt)))
    gauss = np.exp(-np.square(ts_gauss) / np.square(0.5 * w))
    gauss /= np.sum(gauss)
    errs_flt = scipy.signal.fftconvolve(np.abs(errs[:, 0]), gauss,
                                        'valid')[::SS]

    ts_flt = np.linspace(w, T - w, len(errs_flt))
    ax.plot(ts_flt, errs_flt, color=utils.greens[0], lw=1.5)


axs[0].plot(ts[::SS],
            np.abs(xs_err_both[::SS]),
            color="#c0e8a1ff",
            lw=1.0)
plot_filtered(axs[0], xs_err_both)

axs[1].plot(ts[::SS],
            np.abs(xs_err_no_taus[::SS]),
            color="#c0e8a1ff",
            lw=1.0)
plot_filtered(axs[1], xs_err_no_taus)

axs[2].plot(ts[::SS],
            np.abs(xs_err_no_phis[::SS]),
            color="#c0e8a1ff",
            lw=1.0)
plot_filtered(axs[2], xs_err_no_phis)

for i in range(3):
    axs[i].set_xlim(0, T1)
    axs[i].set_ylim(0, 1)
    axs[i].set_xlabel("Time $t$ (s)")
    axs[i].set_ylabel("Error $|\\varepsilon(t)|$")
    axs[i].axvline(T * 0.9, color='k', lw=0.5, linestyle='--')
    axs[i].fill_betweenx([0.0, 1.0], [T * 0.9, T * 0.9], [T1, T1],
                         color=utils.grays[5],
                         lw=0.0)

axs[3].plot(*learning_error(xs_tar, xs_out_both),
            color=utils.greens[0],
            lw=1.5,
            marker='+',
            label="both",
            zorder=10)

axs[3].plot(*learning_error(xs_tar, xs_out_no_taus),
            color=utils.greens[1],
            linestyle=':',
            marker='x',
            lw=0.7,
            label="only $\\varphi$")

axs[3].plot(*learning_error(xs_tar, xs_out_no_phis),
            color=utils.greens[1],
            linestyle='--',
            marker='3',
            lw=0.7,
            label="only $\\tau$")

axs[3].legend(ncol=3,
              loc="upper center",
              bbox_to_anchor=(0.5, 1.22),
              handlelength=1.0,
              handletextpad=0.5,
              columnspacing=0.75)

axs[3].set_xlabel("Time $t$ (s)")
axs[3].set_ylabel("NRMSE $E$")
axs[3].set_ylim(0, 1)
axs[3].set_xlim(0, T1)
axs[3].axvline(T * 0.9, color='k', lw=0.5, linestyle='--')
axs[3].fill_betweenx([0.0, 1.0], [T * 0.9, T * 0.9], [T1, T1],
                     color=utils.grays[5],
                     lw=0.0)

fig.text(0.075, 1.05, "\\textbf{A}", size=12, va="baseline", ha="left")
fig.text(0.22,
         1.05,
         "\\textbf{Error signal over time}",
         va="baseline",
         ha="center")
fig.text(0.22,
         0.95,
         "both $\\varphi(t)$ and $\\tau(t)$",
         va="baseline",
         ha="center")

fig.text(0.335, 1.05, "\\textbf{B}", size=12, va="baseline", ha="left")
fig.text(0.52,
         1.05,
         "\\textbf{Control experiments}",
         va="baseline",
         ha="center")
fig.text(0.43, 0.95, "only $\\varphi(t)$", va="baseline", ha="center")
fig.text(0.595, 0.95, "only $\\tau(t)$", va="baseline", ha="center")

fig.text(0.6625, 1.05, "\\textbf{C}", size=12, va="baseline", ha="left")
fig.text(0.8075, 1.05, "\\textbf{NRMSE over time}", va="baseline", ha="center")

utils.save(fig)
