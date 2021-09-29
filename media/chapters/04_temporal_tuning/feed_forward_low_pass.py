import scipy.signal


def mk_delta(t, ts):
    res = np.zeros_like(ts_flt)
    res[np.argmin(np.abs(ts_flt - t))] = 1.0 / dt
    return res


#def solve_for_filter(flt, tar):
#    M, N = len(flt), len(tar)
#    F = np.zeros((N, M))
#    for i in range(min(N, M)):
#        for j in range(i + 1):
#            F[i, j] = flt[i - j]

#    return F, np.linalg.lstsq(F, tar, rcond=None)[0]

T, dt = 2.002, 1e-4
ts_flt = np.arange(-T / 2, T / 2, dt)

taus = np.geomspace(10e-3, 100e-3, 101)
flts = np.array([(ts_flt >= 0.0) * (np.exp(-ts_flt / tau) / tau)
                 for tau in taus]).T

tau_tar = 5e-3
flt_tar = (ts_flt >= 0.0) * (np.exp(-ts_flt / tau_tar) / tau_tar)


def ff(pre, flts, tar):
    pre_flts = np.array(
        [scipy.signal.fftconvolve(pre, flt, 'same') for flt in flts.T]).T
    D = np.linalg.lstsq(pre_flts, tar, rcond=None)[0]
    return pre_flts, D


fig, axs = plt.subplots(2,
                        5,
                        figsize=(7.8, 2.0),
                        gridspec_kw={
                            "hspace": 1.2,
                        })

us1, us2 = mk_delta(0.0, ts_flt), mk_delta(0.0, ts_flt)
for i in range(5):
    ax = axs[0, i]
    fax = axs[1, i]

    ax.plot(ts_flt * 1e3, us1, 'k-')

    FUS1 = np.fft.fftshift(np.fft.fft(us1))
    fs = np.fft.fftshift(np.fft.fftfreq(len(us1), dt))
    fax.plot(fs, 10 * np.log10(np.abs(FUS1) / np.max(np.abs(FUS1))), 'k-')

    pre_flts, D = ff(us1, flts, flt_tar)
    us1_new = pre_flts @ D

    _, D = ff(us1 if i == 0 else flt_tar, flts, flt_tar)
    pre_flts, _ = ff(us1, flts, flt_tar)

    if i > 0:
        ax.plot(ts_flt * 1e3, flt_tar, color='k', linewidth=0.5)
        ax.plot(ts_flt * 1e3,
                flt_tar,
                color='white',
                linestyle=(0, (1, 2)),
                linewidth=0.5)

        mask = np.logical_and(ts_flt >= 0.0, ts_flt <= 0.1)
        mask_shift = np.logical_and(
            ts_flt - dt >= 0.0,
            ts_flt - dt <= 0.1)  # Compensate for delay of dt
        err = np.sqrt(np.mean(
            np.square(us1[mask_shift] - flt_tar[mask]))) / np.sqrt(
                np.mean(np.square(flt_tar[mask])))
        ax.text(1.0,
                0.95,
                f"$E = {err*100:0.1f}\\%$",
                va="top",
                ha="right",
                size=8,
                transform=ax.transAxes)

        FTAR = np.fft.fftshift(np.fft.fft(flt_tar))
        fax.plot(fs,
                 10 * np.log10(np.abs(FTAR) / np.max(np.abs(FTAR))),
                 color='k',
                 linewidth=0.5)
        fax.plot(fs,
                 10 * np.log10(np.abs(FTAR) / np.max(np.abs(FTAR))),
                 color='white',
                 linestyle=(0, (1, 2)),
                 linewidth=0.5)

    if i == 1:
        utils.annotate(ax,
                       10,
                       75,
                       25,
                       75,
                       "$\\tau = 5\,\\mathrm{ms}$",
                       va="center",
                       ha="left")

    us2 = pre_flts @ D
    us1 = us1_new

    ax.set_xlim(-25, 100)
    ax.set_ylim(-15, 200)
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 100, 3))
    ax.set_xticks(np.linspace(-25, 100, 6), minor=True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Time $t$ (ms)")

    fax.set_xlim(0, 200)
    fax.set_ylim(-20, 0.5)
    fax.set_yticks([])
    fax.set_xticks(np.linspace(0, 200, 3))
    fax.set_xticks(np.linspace(0, 200, 5), minor=True)
    fax.spines["left"].set_visible(False)
    fax.set_xlabel("Frequency $f$ (Hz)")

    fax.plot([0, 0], [-7.5, -17.5],
             'k-',
             solid_capstyle='butt',
             linewidth=2,
             clip_on=False)
    fax.text(5,
             -12.5,
             '$-10 \\, \\mathrm{dB}$',
             size=8,
             va="center",
             ha="left")

    utils.outside_ticks(ax)
    utils.outside_ticks(fax)

utils.save(fig)

