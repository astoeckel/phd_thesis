#!/usr/bin/env python3

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))

import itertools
import tqdm
import numpy as np
import multiprocessing
import random
import env_guard
import h5py
import json

from basis_delay_analysis_common import *
import dlop_ldn_function_bases as bases


def mk_low_pass_filter_basis(q, N, tau0=2e-3, tau1=100.0):
    ts = np.linspace(0, 1, N)
    taus = np.geomspace(tau0, tau1, q)
    res = np.array([np.exp(-ts / tau) for tau in taus])
    res /= np.sqrt(
        np.array([np.inner(res[i], res[i]) for i in range(q)])[:, None])
    return res


# Number of seconds in each trial and timestep
T = 10.0
DT = 0.25e-2
N_SIG = int(T / DT + 1e-9)
THETA = 1.0
N_H = int(THETA / DT + 1e-9)
Q = 63

# Number of samples used when determining the basis LTI system
N = 10000

# Bases to try
BASES = ["fourier", "cosine", "mod_fourier", "legendre"]
N_BASES = len(BASES)

# Windows to try
WINDOWS = ["optimal", "bartlett", "erasure"]
N_WINDOWS = len(WINDOWS)

# Delays to try
N_THETAS = 51

# Frequencies to try
N_FREQS = 31
FREQS = np.geomspace(1, 100, N_FREQS)

# Number of training and test samples
N_TRAIN = 101
N_TEST = 100


def execute_single(idcs):
    fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
                      "evaluate_bases_delays_freq_sweep_dataset.h5")
    with h5py.File(fn, 'r') as f:
        XS_TEST_ALL = f["xs_test_all"]
        YS_TEST_ALL = f["ys_test_all"]
        XS_TRAIN_ALL = f["xs_train_all"]
        YS_TRAIN_ALL = f["ys_train_all"]

        # Fetch the parameters
        i_basis, i_window, i_freq, i_theta = idcs
        basis = BASES[i_basis]
        window = WINDOWS[i_window]
        q = Q

        # Fetch the test and training data
        xs_test = np.array(XS_TEST_ALL[i_freq, i_theta])
        ys_test = np.array(YS_TEST_ALL[i_freq, i_theta])
        xs_train = np.array(XS_TRAIN_ALL[i_freq, i_theta])
        ys_train = np.array(YS_TRAIN_ALL[i_freq, i_theta])

        # Generate the filter
        ts_H, H = mk_impulse_response(basis, window, q=q, dt=DT)

        # Convolve the training data with the basis
        xs_train_conv = convolve(H.T, xs_train)

        # Solve for decoding weights
        D = np.linalg.lstsq(xs_train_conv.reshape(-1, q),
                            ys_train.reshape(-1),
                            rcond=1e-6)[0]

        # Use the decoding weights to compute the delayed test signal
        xs_test_conv = convolve(H.T, xs_test)
        ys_test_hat = xs_test_conv @ D

        # Compute the decoding error
        Es = np.sqrt(np.mean(np.square(ys_test_hat - ys_test), axis=1))

        return idcs, Es


def main():
    # Generate the training and test data for all thetas
    print("Generating dataset....")
    THETAS, XS_TEST_ALL, YS_TEST_ALL, XS_TRAIN_ALL, YS_TRAIN_ALL = [
        [] for _ in range(5)
    ]
    for freq in tqdm.tqdm(FREQS):
        rng = np.random.RandomState(58201)
        thetas, xs_test_all, ys_test_all, xs_train_all, ys_train_all = generate_full_dataset(
            N_THETAS, N_TEST, N_TRAIN, N_SIG, N_H, rng, freq_high=freq)

        THETAS.append(thetas)
        XS_TEST_ALL.append(xs_test_all)
        YS_TEST_ALL.append(ys_test_all)
        XS_TRAIN_ALL.append(xs_train_all)
        YS_TRAIN_ALL.append(ys_train_all)

    fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
                      "evaluate_bases_delays_freq_sweep_dataset.h5")
    with h5py.File(fn, 'w') as f:
        f.create_dataset("thetas", data=np.array(THETAS))
        f.create_dataset("xs_test_all", data=np.array(XS_TEST_ALL))
        f.create_dataset("ys_test_all", data=np.array(YS_TEST_ALL))
        f.create_dataset("xs_train_all", data=np.array(XS_TEST_ALL))
        f.create_dataset("ys_train_all", data=np.array(YS_TEST_ALL))

    print("Executing experiments...")

    fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
                      "evaluate_bases_delays_freq_sweep.h5")
    with h5py.File(fn, 'w') as f:
        f.attrs["bases"] = json.dumps(BASES)
        f.attrs["windows"] = json.dumps(WINDOWS)
        f.attrs["freqs"] = FREQS
        f.attrs["thetas"] = THETAS

        errs = f.create_dataset("errs",
                                shape=(N_BASES, N_WINDOWS, N_FREQS, N_THETAS,
                                       N_TEST))

        idcs = list(
            itertools.product(range(N_BASES), range(N_WINDOWS), range(N_FREQS),
                              range(N_THETAS)))

        random.shuffle(idcs)
        with env_guard.SingleThreadEnvGuard():
            with multiprocessing.get_context('spawn').Pool(16) as pool:
                for (i_basis, i_window, i_freq, i_theta), Es in tqdm.tqdm(
                        pool.imap_unordered(execute_single,
                                            idcs), total=len(idcs)):
                    errs[i_basis, i_window, i_freq, i_theta] = Es


if __name__ == "__main__":
    main()

