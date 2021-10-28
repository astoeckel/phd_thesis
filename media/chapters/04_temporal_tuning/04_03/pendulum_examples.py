import h5py
import scipy.signal


with h5py.File(utils.datafile("pendulum_adaptive_filter.h5"), "r") as f:
    xs_in = f["xs_in"][()]
    xs_tar = f["xs_tar"][()]
    xs_out_both = f["xs_out_both"][()]

def round_to_10(x):
    return np.round(x / 10) * 10

N = xs_in.shape[0]
dt = 1e-3
ts = np.arange(N) * dt
T = N * dt
T1 = np.power(10.0, np.ceil(np.log10(T - 1.0)))
SS = 50

fig, axs = plt.subplots(1,
                        4,
                        figsize=(7.8, 1.2),
                        gridspec_kw={
                            "wspace": 0.2,
                        })

for ax in axs:
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    ax.set_ylim(-2, 2)

Ts = np.linspace(0, T, 4)
for i, T0 in enumerate(Ts):
    if i > 0:
        axs[i].set_xlim(round_to_10(T0 - 10.0), round_to_10(T0))
    else:
        axs[i].set_xlim(0.0, 10.0, round_to_10(T0))
    axs[i].plot(ts, xs_in[:, 0], color=utils.oranges[1], label="Torque $\\tau(t)$")
    axs[i].plot(ts, xs_in[:, 1], color='k', linestyle=(0, (1.25, 1)), lw=0.7, zorder=0, label="Delayed angle $\\varphi(t - \\theta')$")
    axs[i].plot(ts[::SS], xs_out_both[::SS, 0], color=utils.blues[0], label="Prediction $\\hat\\varphi(t)$")
#    axs[i].plot(ts[::SS], xs_tar[::SS, 0], color='white', linestyle=':', lw=0.7)
    axs[i].set_xlabel("Time $t$ (s)")

axs[-1].legend(ncol=3,
              loc="upper right",
              bbox_to_anchor=(1.15, 1.22),
              handlelength=1.0,
              handletextpad=0.7,
              columnspacing=1.5)

fig.text(0.12, 0.875, "\\textbf{C}", size=12, va="baseline", ha="left")
fig.text(0.145, 0.875, "\\textbf{Exemplary trajectories}", va="baseline", ha="left")


utils.save(fig)
