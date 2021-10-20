#!/usr/bin/env python3

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))

import itertools
import tqdm
import numpy as np
import multiprocessing
import env_guard
import h5py

import nengo


def unpack(As, As_shape):
    return np.unpackbits(
        As, count=np.prod(As_shape)).reshape(*As_shape).astype(np.float64)


def shift(xs, t, dt=1e-3):
    N = xs.shape[0]
    N_shift = int(t / dt)
    return np.concatenate((np.zeros(N_shift), xs))[:N]


if len(sys.argv) == 3:
    fn_in = sys.argv[1]
    fn_suffix = "_" + sys.argv[2]
elif len(sys.argv) == 1:
    fn_in = "212fd1234166fc1b_spatio_temporal_network.h5"
    fn_suffix = ""
else:
    print("Invalid command line")
    sys.exit(1)

datafile = os.path.join(os.path.dirname(__file__),
                        "../../../data/manual/chapters/04_temporal_tuning/",
                        fn_in)

with h5py.File(datafile, "r") as f:
    As_shape = f["As_shape"][()]
    xs_train = f["xs_train"][()]
    As_train = unpack(f["As_train"][()], As_shape)
    xs_test = f["xs_test"][()]
    As_test = unpack(f["As_test"][()], As_shape)

xs_train_flt = nengo.Lowpass(100e-3).filtfilt(xs_train)
As_train_flt = nengo.Lowpass(100e-3).filtfilt(As_train)

xs_test_flt = nengo.Lowpass(100e-3).filtfilt(xs_test)
As_test_flt = nengo.Lowpass(100e-3).filtfilt(As_test)

N_DIMS = 2

DELAYS_1D = np.linspace(0, 1, 31 + 1)[:-1]
N_DELAYS_1D = len(DELAYS_1D)

DELAYS_2D = np.linspace(0, 1, 31 + 1)[:-1]
N_DELAYS_2D = len(DELAYS_2D)


def run_1d_delay_decoder_experiment(idcs):
    i_dim, i_delay = idcs

    # Compute a delay decoder on the training data
    thetap = DELAYS_1D[i_delay]
    xs_train_flt_shift = shift(xs_train_flt[:, i_dim], thetap)
    D = np.linalg.lstsq(As_train_flt, xs_train_flt_shift, rcond=1e-2)[0]

    # Test the delay decoder on the test data
    xs_test_flt_shift = shift(xs_test_flt[:, i_dim], thetap)
    rms = np.sqrt(np.mean(np.square(xs_test_flt_shift)))
    err = np.sqrt(np.mean(
        np.square(xs_test_flt_shift - As_test_flt @ D))) / rms

    return idcs, err


def run_2d_delayed_multiplication_experiment(idcs):
    i_delay1, i_delay2 = idcs

    # Compute a delayed multiplication decoder on the training data
    thetap1 = DELAYS_2D[i_delay1]
    thetap2 = DELAYS_2D[i_delay2]
    xs_train_flt_shift = shift(xs_train_flt[:, 0], thetap1) * shift(
        xs_train_flt[:, 1], thetap2)
    D = np.linalg.lstsq(As_train_flt, xs_train_flt_shift, rcond=1e-2)[0]

    # Test the delay decoder on the test data
    xs_test_flt_shift = shift(xs_test_flt[:, 0], thetap1) * shift(
        xs_test_flt[:, 1], thetap2)
    rms = np.sqrt(np.mean(np.square(xs_test_flt_shift)))
    err = np.sqrt(np.mean(
        np.square(xs_test_flt_shift - As_test_flt @ D))) / rms

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

    params_2d = list(itertools.product(range(N_DELAYS_2D), range(N_DELAYS_2D)))
    errs_2d = np.zeros((N_DELAYS_2D, N_DELAYS_2D))
    with env_guard.SingleThreadEnvGuard():
        with multiprocessing.get_context('spawn').Pool(16) as pool:
            for idcs, E in tqdm.tqdm(pool.imap_unordered(
                    run_2d_delayed_multiplication_experiment, params_2d),
                                     total=len(params_2d)):
                errs_2d[idcs] = E

    fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
                      "spatio_temporal_network_analysis" + fn_suffix + ".h5")

    with h5py.File(fn, "w") as f:
        f.create_dataset("delays_1d", data=DELAYS_1D)
        f.create_dataset("delays_2d", data=DELAYS_2D)
        f.create_dataset("errs_1d", data=errs_1d)
        f.create_dataset("errs_2d", data=errs_2d)


if __name__ == "__main__":
    main()

