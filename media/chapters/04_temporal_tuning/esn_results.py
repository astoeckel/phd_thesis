import h5py

with h5py.File(utils.datafile("esn_decode_delays_1d.h5"), "r") as f:
    delays_1d = f["delays_1d"][()]
    errs_1d = f["errs_1d"][()]


with h5py.File(utils.datafile("esn_decode_delays_2d.h5"), "r") as f:
    delays_2d = f["delays_2d"][()]
    errs_2d = f["errs_2d"][()]


fig, axs = plt.subplots(1, 3)
axs[0].plot(delays_1d, errs_1d.T)

axs[1].contourf(delays_2d, delays_2d, errs_2d, vmin=0.0, vmax=1.0, levels=np.linspace(0, 1, 11))

utils.save(fig)
