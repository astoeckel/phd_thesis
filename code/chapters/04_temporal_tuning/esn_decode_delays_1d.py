#!/usr/bin/env python3

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))

import itertools
import tqdm
import numpy as np
import multiprocessing
import env_guard
import h5py
import pyESN
import copy

import nengo


def unpack(As, As_shape):
    return np.unpackbits(
        As, count=np.prod(As_shape)).reshape(*As_shape).astype(np.float64)


def shift(xs, t, dt=1e-3):
    N = xs.shape[0]
    N_shift = int(t / dt)
    return np.concatenate((np.zeros(N_shift), xs))[:N]


datafile = os.path.join(
    os.path.dirname(__file__),
    "../../../data/manual/chapters/04_temporal_tuning/212fd1234166fc1b_spatio_temporal_network.h5"
)

with h5py.File(datafile, "r") as f:
    xs_train = f["xs_train"][()]
    xs_test = f["xs_test"][()]

XS_TRAIN_FLT = nengo.Lowpass(100e-3).filtfilt(xs_train)
XS_TEST_FLT = nengo.Lowpass(100e-3).filtfilt(xs_test)

# Initialise the ESN
np.random.seed(5781)
ESN_INST = pyESN.ESN(n_inputs=2,
                     n_outputs=1,
                     n_reservoir=1000,
                     teacher_forcing=True,
                     spectral_radius=0.8)

N_DIMS = 2

DELAYS_1D = np.linspace(0, 1, 101 + 1)[:-1]
N_DELAYS_1D = len(DELAYS_1D)


def run_1d_delay_decoder_experiment(idcs):
    i_dim, i_delay = idcs

    # Copy the echo state network
    esn = copy.deepcopy(ESN_INST)

    # Fit the ESN to the training data
    thetap = DELAYS_1D[i_delay]
    xs_train_flt_shift = shift(XS_TRAIN_FLT[:, i_dim], thetap)
    esn.fit(XS_TRAIN_FLT, xs_train_flt_shift)

    # Test the delay decoder on the test data
    xs_test_flt_shift = shift(XS_TEST_FLT[:, i_dim], thetap)
    xs_test_flt_shift_pred = esn.predict(XS_TEST_FLT, continuation=False)[:, 0]
    rms = np.sqrt(np.mean(np.square(xs_test_flt_shift)))
    err = np.sqrt(
        np.mean(np.square(xs_test_flt_shift - xs_test_flt_shift_pred))) / rms

    return idcs, err


def main():
    params_1d = list(itertools.product(range(N_DIMS), range(N_DELAYS_1D)))
    errs_1d = np.zeros((N_DIMS, N_DELAYS_1D))
    with env_guard.SingleThreadEnvGuard():
        with multiprocessing.get_context('spawn').Pool(16) as pool:
            for idcs, E in tqdm.tqdm(pool.imap_unordered(
                    run_1d_delay_decoder_experiment, params_1d),
                                     total=len(params_1d)):
                errs_1d[idcs] = E

    fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
                      "esn_decode_delays_1d.h5")

    with h5py.File(fn, "w") as f:
        f.create_dataset("delays_1d", data=DELAYS_1D)
        f.create_dataset("errs_1d", data=errs_1d)


if __name__ == "__main__":
    main()

