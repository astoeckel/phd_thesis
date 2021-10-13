#!/usr/bin/env python3

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))

import itertools
import tqdm
import numpy as np
import multiprocessing
import env_guard
import h5py
import dlop_ldn_function_bases as bases
import copy
import random

import nengo

N = 1000

FREQS = np.linspace(0, 500, 251)
N_FREQS = len(FREQS)

N_TRIALS = 1001

QS = np.arange(1, 101, 1, dtype=int)
N_QS = len(QS)

# Compute the basis transformation matrices
Hs, Hs_inv = [], []
for q in tqdm.tqdm(QS):
    H = bases.mk_ldn_basis(q, N)
    H_inv = np.linalg.pinv(H)

    Hs.append(H)
    Hs_inv.append(H_inv)


def bandlimited_white_noise(fmax, N, seed=0):
    xs = np.random.RandomState(seed).randn(N)
    Xs = np.fft.fft(xs)
    fs = np.fft.fftfreq(N, 1 / N)
    Xs[np.abs(fs) > fmax] = 0
    return np.real(np.fft.ifft(Xs))


def run(idcs):
    i_q, i_freq = idcs

    Es = np.zeros(N_TRIALS)
    for i_trial in range(N_TRIALS):
        # Generate the reference signal and the band-limited input signal
        sig_orig = bandlimited_white_noise(N, N, seed=i_trial)
        sig_lim = bandlimited_white_noise(FREQS[i_freq], N, seed=i_trial)

        # Feed all signals through the LDN filter
        H, H_inv = Hs[i_q], Hs_inv[i_q]
        sig_orig_rec = H_inv @ (H @ sig_orig)
        sig_lim_rec = H_inv @ (H @ sig_lim)

        # Compute the original signal rms
        rms = np.sqrt(np.mean(np.square(sig_orig_rec)))

        # Compute the error
        Es[i_trial] = np.sqrt(np.mean(np.square(sig_lim_rec - sig_orig_rec))) / rms

    return idcs, Es


def main():
    params = list(
        itertools.product(range(N_QS), range(N_FREQS)))
    random.shuffle(params)

    fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
                      "ldn_spectrum.h5")

    with h5py.File(fn, "w") as f:
        f.create_dataset("freqs", data=FREQS)
        f.create_dataset("qs", data=QS)
        errs = f.create_dataset("errs",
                                shape=(N_QS, N_FREQS, N_TRIALS),
                                compression="gzip")

        with env_guard.SingleThreadEnvGuard():
            with multiprocessing.get_context('spawn').Pool() as pool:
                for idcs, Es in tqdm.tqdm(pool.imap_unordered(run, params),
                                         total=len(params)):
                    errs[idcs] = Es


if __name__ == "__main__":
    main()

