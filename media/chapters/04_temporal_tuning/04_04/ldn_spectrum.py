import numpy as np
import random
import multiprocessing
import tqdm
import nengo

from dlop_ldn_function_bases import *

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['figure.dpi'] = 200


def bandlimited_white_noise(fmax, N, seed=0):
    xs = np.random.RandomState(seed).randn(N)
    Xs = np.fft.fft(xs)
    fs = np.fft.fftfreq(N, 1 / N)
    Xs[np.abs(fs) > fmax] = 0
    return np.real(np.fft.ifft(Xs))


with h5py.File(utils.datafile("ldn_spectrum.h5"), "r") as f:
    FREQS = f["freqs"][()]
    QS = f["qs"][()]
    ERRS = f["errs"][()]

fig, axs = plt.subplots(1,
                        3,
                        figsize=(7.225, 1.75),
                        gridspec_kw={
                            "wspace": 0.3,
                            "width_ratios": [2, 3, 3]
                        })
qs = QS[np.geomspace(1, len(QS) + 0.5, 6, dtype=np.int) - 1]
N = 1000

THS = np.linspace(0.01, 0.2, 5)

us_test = np.random.randn(100 * N)
us_test = nengo.Lowpass(10e-3).filt(us_test)
us_test_valid = us_test[N-1:]
Us_test_valid = np.fft.fftshift(np.fft.fft(us_test_valid))
fs = np.fft.fftshift(np.fft.fftfreq(len(us_test_valid), 1e-3))

for i, q in enumerate(qs):
    H = mk_ldn_basis(q, N)
    ms = np.array([np.convolve(us_test, H[j], 'valid') for j in range(q)]).T
    D = np.linalg.lstsq(ms, us_test_valid, rcond=1e-6)[0]
    us_rec = ms @ D
    Us_rec = np.fft.fftshift(np.fft.fft(us_rec))
    X = np.abs((Us_test_valid * np.conj(Us_rec)) / (Us_test_valid * np.conj(Us_test_valid)))
    color = cm.get_cmap('viridis')(i / (len(qs) - 1))
    axs[0].plot(fs,
                10 * np.log10(X / np.max(X)),
                color=color,
                label="$q = {}$".format(q),
                zorder=-i)

axs[0].set_xlim(0, 50)
axs[0].set_xticks(np.linspace(0, 50, 6), minor=True)
axs[0].set_ylim(-15, 0)
axs[0].set_ylabel("Gain $|X|$ (dB)")
axs[0].set_xlabel("Frequency $f$ (Hz)")
axs[0].legend(ncol=len(qs),
              handlelength=1.0,
              handletextpad=0.5,
              columnspacing=1.0,
              loc='upper center',
              bbox_to_anchor=(1.08, 1.35))

#fs, Es = utils.run_with_cache(compute_spectral_error, qs)
for i, q in enumerate(qs):
    i_q = list(QS).index(qs[i])
    color = cm.get_cmap('viridis')(i / (len(qs) - 1))
    axs[1].plot(FREQS, np.median(ERRS[i_q], axis=-1), color=color)

axs[1].set_xticks(np.linspace(0, 500, 6), minor=True)
axs[1].set_xlabel("Bandlimit $\\hat f$ (Hz)")
axs[1].set_ylabel("NRMSE $E$")
axs[1].set_ylim(0, 1)
axs[1].set_xlim(min(FREQS), max(FREQS))

for i, th in enumerate(THS):
    color = mpl.cm.get_cmap("inferno")(i / 5)
    axs[1].axhline(th, linestyle=':', color='grey', linewidth=0.5, zorder=-10)
    axs[1].plot(1.0,
                th,
                'o',
                color=color,
                clip_on=False,
                transform=axs[1].transAxes,
                zorder=100,
                markersize=4)
# Compute the frequencies at which the errors fall below a certain threshold
for i, th in enumerate(THS):
    color = mpl.cm.get_cmap("inferno")(i / 5)
    th_idcs = np.argmin(np.abs(ERRS - th), axis=1)
    fs = FREQS[0] + (th_idcs / (len(FREQS) - 1)) * (FREQS[-1] - FREQS[0])

    fs_rel = fs  #/ QS[:, None]

    fs25 = np.percentile(fs_rel, 25, axis=-1)
    fs50 = np.mean(fs_rel, axis=-1)
    fs75 = np.percentile(fs_rel, 75, axis=-1)

    axs[2].plot(QS, fs50, color=color, label=f"${th:0.2f}$")
#    axs[2].fill_between(QS, fs25, fs75, alpha=0.5, lw=0.0, color=color)
#    axs[2].set_ylim(4, 10)

axs[2].legend(ncol=len(THS),
              handlelength=1.0,
              handletextpad=0.5,
              columnspacing=1.0,
              loc='upper center',
              bbox_to_anchor=(0.35, 1.35))

axs[2].set_xlabel("Basis order $q$")
axs[2].set_ylabel("Bandlimit frequency $\\hat f$")
axs[2].set_xlim(0, 100)
axs[2].set_ylim(0, 500)
axs[2].set_xticks(np.linspace(0, 100, 5), minor=True)

axs[0].set_title("\\textbf{LDN power spectrum}")
axs[0].text(-0.43,
            1.065,
            "\\textbf{A}",
            size=12,
            va="baseline",
            ha="left",
            transform=axs[0].transAxes)


axs[1].set_title("\\textbf{Bandlimiting tolerance}")
axs[1].text(-0.21,
            1.065,
            "\\textbf{B}",
            size=12,
            va="baseline",
            ha="left",
            transform=axs[1].transAxes)

axs[2].set_title("\\textbf{Effective bandwidth}")
axs[2].text(-0.24,
            1.065,
            "\\textbf{C}",
            size=12,
            va="baseline",
            ha="left",
            transform=axs[2].transAxes)

fig.align_labels()

utils.save(fig)

