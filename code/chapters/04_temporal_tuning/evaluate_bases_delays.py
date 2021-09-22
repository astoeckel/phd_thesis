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
DT = 1e-2
N_SIG = int(T / DT + 1e-9)
THETA = 1.0
N_H = int(THETA / DT + 1e-9)

# Number of samples used when determining the basis LTI system
N = 10000
Q_MIN = 3
Q_MAX = 63

# Bases to try
BASES = ["fourier", "cosine", "mod_fourier", "legendre"]
N_BASES = len(BASES)

# Windows to try
WINDOWS = ["optimal", "bartlett", "erasure"]
N_WINDOWS = len(WINDOWS)

# Delays to try
N_THETAS = 51

# qs to try, only use odd QS
N_QS = 31
QS = np.unique([2 * int(q / 2) + 1 for q in np.linspace(Q_MIN, Q_MAX, N_QS)])
N_QS = len(QS)

# Generate the training and test data for all thetas
print("Generating dataset....")
N_TRAIN = 101
N_TEST = 100
rng = np.random.RandomState(58201)
THETAS, XS_TEST_ALL, YS_TEST_ALL, XS_TRAIN_ALL, YS_TRAIN_ALL = generate_full_dataset(
    N_THETAS, N_TEST, N_TRAIN, N_SIG, N_H, rng)


def execute_single(idcs):
    # Fetch the parameters
    i_basis, i_window, i_q, i_theta = idcs
    q = QS[i_q]
    basis = BASES[i_basis]
    window = WINDOWS[i_window]

    # Fetch the test and training data
    xs_test, ys_test = XS_TEST_ALL[i_theta], YS_TEST_ALL[i_theta]
    xs_train, ys_train = XS_TRAIN_ALL[i_theta], YS_TRAIN_ALL[i_theta]

    # Generate the filter
    ts_H, H = mk_impulse_response(basis, window, q=q, dt=DT)

    # Convolve the training data with the basis
    xs_train_conv = convolve(H.T, xs_train)

    # Solve for decoding weights
    D = np.linalg.lstsq(xs_train_conv.reshape(-1, q),
                        ys_train.reshape(-1),
                        rcond=1e-4)[0]

    # Use the decoding weights to compute the delayed test signal
    xs_test_conv = convolve(H.T, xs_test)
    ys_test_hat = xs_test_conv @ D

    # Compute the decoding error
    Es = np.sqrt(np.mean(np.square(ys_test_hat - ys_test), axis=1))

    return idcs, Es


def main():
    print("Executing experiments...")

    fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
                      "evaluate_bases_delays.h5")
    with h5py.File(fn, 'w') as f:
        f.attrs["bases"] = json.dumps(BASES)
        f.attrs["windows"] = json.dumps(WINDOWS)
        f.attrs["qs"] = QS
        f.attrs["thetas"] = THETAS

        errs = f.create_dataset("errs",
                                shape=(N_BASES, N_WINDOWS, N_QS, N_THETAS,
                                       N_TEST))

        idcs = list(
            itertools.product(range(N_BASES), range(N_WINDOWS), range(N_QS),
                              range(N_THETAS)))

        random.shuffle(idcs)
        with env_guard.SingleThreadEnvGuard():
            with multiprocessing.get_context('spawn').Pool() as pool:
                for (i_basis, i_window, i_q, i_theta), Es in tqdm.tqdm(
                        pool.imap_unordered(execute_single,
                                            idcs), total=len(idcs)):
                    errs[i_basis, i_window, i_q, i_theta] = Es


if __name__ == "__main__":
    main()

