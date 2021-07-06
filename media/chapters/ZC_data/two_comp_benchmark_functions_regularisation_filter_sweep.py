import h5py
import scipy.interpolate

with h5py.File(
        utils.datafile(
            "two_comp_benchmark_functions_regularisation_filter_sweep_2.h5")
) as f:
    network = bool(f["network"][()])
    regs_linear = f["regs_linear"][()]
    regs_cond = f["regs_cond"][()]
    tau_pre_filts = f["tau_pre_filts"][()]
    param_keys = str(f["params_keys"][()], "utf-8").split("\n")
    errs = f["errs"][()]

print(param_keys, errs.shape)

param_keys_idx_map = {
    "linear": (0, 0),
    "linear_2d": (0, 1),
    "gc50_no_noise": (1, 0),
    "gc50_noisy": (2, 0),
    "gc100_no_noise": (1, 1),
    "gc100_noisy": (2, 1),
    "gc200_no_noise": (1, 2),
    "gc200_noisy": (2, 2),
}

param_keys_title_map = {
    "linear":
    "Current-based LIF",
    "linear_2d":
    "Current-based LIF\n(Two-Layer)",
    "gc50_no_noise":
    "Two-compartment LIF\n($c_{12} = 50\,\\mathrm{nS}$)",
    "gc100_no_noise":
    "Two-compartment LIF\n($c_{12} = 100\,\\mathrm{nS}$)",
    "gc200_no_noise":
    "Two-compartment LIF\n($c_{12} = 200\,\\mathrm{nS}$)",
    "gc50_noisy":
    "Two-compartment LIF\n($c_{12} = 50\,\\mathrm{nS}$; noise model)",
    "gc100_noisy":
    "Two-compartment LIF\n($c_{12} = 100\,\\mathrm{nS}$; noise model)",
    "gc200_noisy":
    "Two-compartment LIF\n($c_{12} = 200\,\\mathrm{nS}$; noise model)",
}

fig, axs = plt.subplots(3,
                        3,
                        figsize=(7.0, 7.0),
                        gridspec_kw={
                            "hspace": 0.5,
                            "wspace": 0.5,
                        })
utils.remove_frame(axs[0, 2])


def interpolate_in_log_space(xs, ys, zss, ss=10):
    xs, ys, zss = np.log10(xs), np.log10(ys), np.log10(zss)

    xsp = np.linspace(xs[0], xs[-1], len(xs) * ss)
    ysp = np.linspace(ys[0], ys[-1], len(ys) * ss)

    f = scipy.interpolate.interp2d(xs, ys, zss.T, 'cubic')
    zssp = f(xsp, ysp).T

    return np.power(10.0, xsp), np.power(10.0, ysp), np.power(10.0, zssp)


for i, key in enumerate(param_keys):
    ax = axs[param_keys_idx_map[key]]
    regs = regs_linear if "linear" in key else regs_cond
    flts = tau_pre_filts
    E = np.median(errs[i, :, :, 1, :, 0], axis=-1)
    Elog = np.log10(E)
    vmin, vmax = np.log10(0.03), np.log10(0.4)
    C = ax.contourf(flts,
                    regs,
                    Elog,
                    vmin=vmin,
                    vmax=vmax,
                    levels=np.linspace(vmin, vmax, 21))
    ax.contour(flts,
               regs,
               Elog,
               vmin=vmin,
               vmax=vmax,
               levels=C.levels,
               colors=['white'],
               linestyles=['--'],
               linewidths=[0.7])
    ax.set_title(param_keys_title_map[key])
    ax.set_xscale("log")
    ax.set_yscale("log")

    regsp, fltsp, Ep = interpolate_in_log_space(regs, flts, E)
    iregp, ifltp = np.unravel_index(np.argmin(Ep, axis=None), Ep.shape)
    ax.plot(fltsp[ifltp], regsp[iregp], '+')

    iregp = np.unravel_index(np.argmin(Ep[:, 0], axis=None), Ep[:, 0].shape)
    ax.plot(fltsp[0], regsp[iregp], '+')

utils.save(fig)

