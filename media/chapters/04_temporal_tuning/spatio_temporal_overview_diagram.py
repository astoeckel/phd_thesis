import matplotlib.pyplot as plt
import nengo
import h5py
import tqdm
import nengo_extras.plot_spikes

np.random.seed(5829)

def double_rows(A):
    res = np.zeros((A.shape[0] * 2, *A.shape[1:]))
    res[0::2] = A
    res[1::2] = A
    return res

def double(A):
    return double_rows(double_rows(A).T).T

def quadruple(A):
    return double(double(A))

def pp(td, A):  # pre-process
    return nengo_extras.plot_spikes.cluster(ts, A[:, ::2],
                                            filter_width=0.01)[1]


def reduce_spike_train(A, ss):
    res = np.zeros((A.shape[0] // ss, A.shape[1]))
    for i in range(ss):
        res += A[i::ss][:res.shape[0]]
    return res


def shift(xs, t, dt=1e-3):
    N = xs.shape[0]
    N_shift = int(t / dt)
    return np.concatenate((np.zeros(N_shift), xs))[:N]


def unpack(As, As_shape):
    return np.unpackbits(
        As, count=np.prod(As_shape)).reshape(*As_shape).astype(np.float64)


with h5py.File(utils.datafile("spatio_temporal_network.h5"), "r") as f:
    W_in = f["W_in"][()]
    W_rec = f["W_rec"][()]
    gains = f["gains"][()]
    biases = f["biases"][()]
    Es = f["Es"][()]
    TES = f["TEs"][()]

    As_shape = f["As_shape"][()]

    xs_train = f["xs_train"][()]
    As_train = unpack(f["As_train"][()], As_shape)

    xs_test = f["xs_test"][()]
    As_test = unpack(f["As_test"][()], As_shape)

    ts = np.arange(0, xs_train.shape[0]) * 1e-3

xs_train_flt = nengo.Lowpass(100e-3).filtfilt(xs_train)
As_train_flt = nengo.Lowpass(100e-3).filtfilt(As_train)

xs_test_flt = nengo.Lowpass(100e-3).filtfilt(xs_test)
As_test_flt = nengo.Lowpass(100e-3).filtfilt(As_test)

As_subset = np.random.randint(0, As_shape[1], 200)
ts_subset = np.random.randint(0, xs_train.shape[0], 500)

fig = plt.figure(figsize=(7.66, 5.0))

ts_range = ts <= 5.0
tsp = ts[ts_range]


def setup_ax(ax):
    utils.remove_frame(ax)
    ax.set_xlim(np.min(tsp), np.max(tsp))
    ax.set_ylim(-1.25, 1.25)
    return ax


ax1 = setup_ax(fig.add_axes([0.07, 0.455, 0.15, 0.1]))
ax2 = setup_ax(fig.add_axes([0.07, 0.79, 0.15, 0.1]))
ax3 = setup_ax(fig.add_axes([0.73, 0.455, 0.15, 0.1]))
ax6 = setup_ax(fig.add_axes([0.73, 0.625, 0.15, 0.1]))
ax4 = setup_ax(fig.add_axes([0.73, 0.795, 0.15, 0.1]))
ax5 = setup_ax(fig.add_axes([0.375, 0.95, 0.2, 0.075]))

# Select interesting looking spike trains
activity_sums = np.sum(As_test[ts_range], axis=0)
non_silent_idcs = np.where(
    np.logical_and(activity_sums > np.percentile(activity_sums, 40),
                   activity_sums < np.percentile(activity_sums, 60)))[0]
As_subset_plot = np.random.choice(non_silent_idcs, 75)
Opre1 = pp(tsp, As_test[ts_range][:, As_subset_plot])
ax5.imshow(quadruple(reduce_spike_train(Opre1, 10).T),
           cmap='Greys',
           vmin=0.0,
           vmax=1.0,
           extent=[0, 10, 0.5, Opre1.shape[1] + 0.5],
           origin='lower',
           interpolation='none',
           zorder=0)
ax5.set_ylim(0.0, Opre1.shape[1] + 1)
ax5.set_aspect('auto')
ax5.plot([0, 0.2], [-0.1, -0.1],
         'k',
         lw=1.5,
         solid_capstyle='butt',
         transform=ax5.transAxes,
         clip_on=False)
ax5.text(0.1,
         -0.15,
         "$1\,\mathrm{s}$",
         size=8,
         va="top",
         ha="center",
         transform=ax5.transAxes)

ax1.plot(tsp, xs_test[ts_range][:, 0], 'k-', lw=0.7, clip_on=False)
ax1.plot([0, 0.2], [0.0, 0.0],
         'k',
         lw=1.5,
         solid_capstyle='butt',
         transform=ax1.transAxes,
         clip_on=False)
ax1.text(0.1,
         -0.05,
         "$1\,\mathrm{s}$",
         size=8,
         va="top",
         ha="center",
         transform=ax1.transAxes)

ax2.plot(tsp, xs_test[ts_range][:, 1], 'k-', lw=0.7, clip_on=False)
ax2.plot([0, 0.2], [0.0, 0.0],
         'k',
         lw=1.5,
         solid_capstyle='butt',
         transform=ax2.transAxes,
         clip_on=False)
ax2.text(0.1,
         -0.05,
         "$1\,\mathrm{s}$",
         size=8,
         va="top",
         ha="center",
         transform=ax2.transAxes)

ax3.plot(tsp,
         xs_test_flt[ts_range][:, 0],
         'k--',
         zorder=101,
         lw=0.5,
         clip_on=False)
ax4.plot(tsp,
         xs_test_flt[ts_range][:, 1],
         'k--',
         zorder=101,
         lw=0.5,
         clip_on=False)

ax3.plot([0.5, 1.5], [-0.8, -0.8],
         'k',
         lw=1.5,
         solid_capstyle='butt',
         clip_on=False)
ax3.text(1.0,
         -0.89,
         "$\\theta$",
         size=8,
         va="top",
         ha="center")

n_thetas = 20
for j, theta in enumerate(tqdm.tqdm(np.linspace(0, 1.0, n_thetas))):
    color = cm.get_cmap('inferno')(1.0 - j / (n_thetas - 1))

    xs_train_flt_shift = shift(xs_train_flt[:, 0], theta)
    d1 = np.linalg.lstsq(As_train_flt[:, As_subset][ts_subset],
                         xs_train_flt_shift[ts_subset],
                         rcond=1e-2)[0]

    xs_train_flt_shift = shift(xs_train_flt[:, 1], theta)
    d2 = np.linalg.lstsq(As_train_flt[:, As_subset][ts_subset], xs_train_flt_shift[ts_subset], rcond=1e-2)[0]

    ax3.plot(tsp,
             As_test_flt[:, As_subset][ts_range] @ d1,
             color=color,
             zorder=100 - j,
             clip_on=False)
    ax4.plot(tsp, As_test_flt[:, As_subset][ts_range] @ d2, color=color, zorder=100-j, clip_on=False)

xs_train_flt_shift1 = shift(xs_train_flt[:, 0], 0.2)
xs_train_flt_shift2 = shift(xs_train_flt[:, 1], 0.4)
dprod = np.linalg.lstsq(As_train_flt[:, As_subset][ts_subset],
                     (xs_train_flt_shift1 * xs_train_flt_shift2)[ts_subset],
                     rcond=1e-2)[0]

xs_test_flt_shift1 = shift(xs_test_flt[:, 0], 0.2)
xs_test_flt_shift2 = shift(xs_test_flt[:, 1], 0.4)
ax6.plot(tsp, (xs_test_flt_shift1 * xs_test_flt_shift2)[ts_range], 'k', lw=0.5, clip_on=False)
ax6.plot(tsp, As_test_flt[:, As_subset][ts_range] @ dprod, clip_on=False, color=utils.blues[0])
ax6.plot(tsp, (xs_test_flt_shift1 * xs_test_flt_shift2)[ts_range], 'white', linestyle=(0, (1, 1)), lw=0.5, clip_on=False)

utils.save(fig)

