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

DELAYS_2D = np.linspace(0, 1, 31 + 1)[:-1]
N_DELAYS_2D = len(DELAYS_2D)


def run_2d_delayed_multiplication_experiment(idcs):
    i_delay1, i_delay2 = idcs

    # Copy the echo state network
    esn = copy.deepcopy(ESN_INST)

    # Fit the ESN to the training data
    thetap1 = DELAYS_2D[i_delay1]
    thetap2 = DELAYS_2D[i_delay2]
    xs_train_flt_shift = shift(XS_TRAIN_FLT[:, 0], thetap1) * shift(
        XS_TRAIN_FLT[:, 1], thetap2)
    esn.fit(XS_TRAIN_FLT, xs_train_flt_shift)

    # Test the delay decoder on the test data
    xs_test_flt_shift = shift(XS_TEST_FLT[:, 0], thetap1) * shift(
        XS_TEST_FLT[:, 1], thetap2)
    xs_test_flt_shift_pred = esn.predict(XS_TEST_FLT, continuation=False)[:, 0]
    rms = np.sqrt(np.mean(np.square(xs_test_flt_shift)))
    err = np.sqrt(
        np.mean(np.square(xs_test_flt_shift - xs_test_flt_shift_pred))) / rms

    return idcs, err


def main():
    params_2d = list(itertools.product(range(N_DELAYS_2D), range(N_DELAYS_2D)))
    errs_2d = np.zeros((N_DELAYS_2D, N_DELAYS_2D))
    with env_guard.SingleThreadEnvGuard():
        with multiprocessing.get_context('spawn').Pool(8) as pool:
            for idcs, E in tqdm.tqdm(pool.imap_unordered(
                    run_2d_delayed_multiplication_experiment, params_2d),
                                     total=len(params_2d)):
                errs_2d[idcs] = E

    fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
                      "esn_decode_delays_2d.h5")

    with h5py.File(fn, "w") as f:
        f.create_dataset("delays_2d", data=DELAYS_2D)
        f.create_dataset("errs_2d", data=errs_2d)


if __name__ == "__main__":
    main()

