import matplotlib.pyplot as plt
import nengo
import h5py
import tqdm

def unpack(As, As_shape):
    return np.unpackbits(As, count=np.prod(As_shape)).reshape(*As_shape).astype(np.float64)

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

def shift(xs, t, dt=1e-3):
    N = xs.shape[0]
    N_shift = int(t / dt)
    return np.concatenate((np.zeros(N_shift), xs))[:N]

fig, ax = plt.subplots()

for theta in tqdm.tqdm(np.linspace(0, 1.0, 5)):
    xs_train_flt_shift = shift(xs_train_flt[:, 0], theta) * shift(xs_train_flt[:, 1], theta) 
    xs_test_flt_shift = shift(xs_test_flt[:, 0], theta) * shift(xs_test_flt[:, 1], theta) 
    d1 = np.linalg.lstsq(As_train_flt, xs_train_flt_shift, rcond=1e-2)[0]

    ax.plot(ts, As_test_flt @ d1)
    ax.plot(ts, xs_test_flt_shift, 'k--')
    print(np.sqrt(np.mean(np.square(As_test_flt @ d1 - xs_test_flt_shift))))

ax.set_xlim(50, 60)

#    xs_test_flt_shift = shift(xs_test_flt[:, 0], theta)
#    ax.plot(ts, xs_test_flt_shift,)

utils.save(fig)
