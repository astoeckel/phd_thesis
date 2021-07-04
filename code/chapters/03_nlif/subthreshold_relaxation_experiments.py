#!/usr/bin/env python3

import itertools
import multiprocessing
import sys, os

import numpy as np

sys.path.append(os.path.join('..', 'lib'))

import random

import lif_utils
import env_guard
import nonneg_common

import bioneuronqp

import tqdm
import h5py

# Maximum firing rate (hard-coded in nonneg_common.mk_ensemble)
A_MAX = 100

# Number of repetitions for each parameter set
N_REPEAT = 11

# Number of regularisation factors
N_REGS = 21
REGS = np.logspace(-2, -1, N_REGS)

# Number of different thresholds to try
N_THS = 7
THS = [None, False] + list(np.linspace(1, 0, N_THS - 2))[::-1]

# Functions to analyse
FUNCTIONS = [lambda x: x, lambda x: 2.0 * np.square(x) - 1.0, lambda x: 3 * (x - 0.75) * (x - 0.0) * (x + 0.75)]
N_FUNCTIONS = len(FUNCTIONS)

# Number of excitatory to inhibitory ratios to explore
N_RATIOS = 4
RATIOS = np.linspace(0, 1, N_RATIOS + 1)[1:]

# Number of training samples
N_SMPLS_TRAIN = 101

# Number of test samples
N_SMPLS_TEST = 1001

# Number of pre- and post-neurons
N_PRE = 101
N_POST = 102


def forkrng(rng=np.random):
    return np.random.RandomState(rng.randint((1 << 31)))


def make_network(idx, sigma=0.01, compute_decoder=True, compute_test=True):
    rng = np.random.RandomState(67349 * idx + 13109)

    # Compute the ensemble parameters
    α_pre, β_pre, E_pre = nonneg_common.mk_ensemble(N_PRE, rng=forkrng(rng))
    α_post, β_post, E_post = nonneg_common.mk_ensemble(N_POST,
                                                       rng=forkrng(rng))

    # Compute the inputs
    xs_train = np.linspace(-1, 1, N_SMPLS_TRAIN).reshape(-1, 1)
    Js_train_pre = α_pre[None, :] * (xs_train @ E_pre.T) + β_pre[None, :]
    As_train_pre = lif_utils.lif_rate(Js_train_pre)

    # Compute the target decoders
    Js_train_post = α_pre[None, :] * (xs_train @ E_pre.T) + β_pre[None, :]
    As_train_post = lif_utils.lif_rate(Js_train_pre)

    # Determine the post-population decoder
    if compute_decoder:
        Js_train_post = α_post[None, :] * (xs_train @ E_post.T) + β_post[
            None, :]
        As_train_post = lif_utils.lif_rate(Js_train_post)
        A = (As_train_post.T @ As_train_post +
             N_SMPLS_TRAIN * np.square(sigma * A_MAX) * np.eye(N_POST))
        Y = As_train_post.T @ xs_train[:, 0]
        D = np.linalg.solve(A, Y)
    else:
        D = None

    # For each function, determine the target activities
    Js_train_post_tar = [
        α_post[None, :] * (f(xs_train) @ E_post.T) + β_post[None, :]
        for f in FUNCTIONS
    ]

    # Compute the test pre-activities
    if compute_test:
        xs_test = np.linspace(-1, 1, N_SMPLS_TEST).reshape(-1, 1)
        Js_test_pre = α_pre[None, :] * (xs_test @ E_pre.T) + β_pre[None, :]
        As_test_pre = lif_utils.lif_rate(Js_test_pre)

        # Compute the target currents and the target decoded output
        ys_test = [f(xs_test)[:, 0] for f in FUNCTIONS]
        Js_test_post = [
            α_post[None, :] * (ys.reshape(-1, 1) @ E_post.T) + β_post[None, :] for ys in ys_test
        ]
    else:
        As_test_pre = None
        ys_test = None
        Js_test_post = None

    return idx, {
        "As_train_pre": As_train_pre,
        "Js_train_post_tar": Js_train_post_tar,
        "D": D,
        "As_test_pre": As_test_pre,
        "ys_test": ys_test,
        "Js_test_post": Js_test_post
    }


def compute_weights(params):
    # Unpack the parameters
    idx_fun, idx_reg, idx_th, idx_ratio, idx_net = params

    # Fetch the parameters
    reg = np.square(REGS[idx_reg] * A_MAX)
    iTh = THS[idx_th]
    p_exc = RATIOS[idx_ratio]

    # Generate the network
    _, net = make_network(idx_net, compute_decoder=False, compute_test=False)

    # Arbitrarily decide whether the pre neurons are excitatory or inhibitory
    rng = np.random.RandomState(67349 * idx_net + 13109)
    is_exc = rng.choice([True, False], (N_PRE, 1), p=[p_exc, 1.0 - p_exc])
    is_inh = ~is_exc

    # Assemble the connectivity matrix
    C = np.array((
        np.repeat(is_exc, N_POST, axis=1),
        np.repeat(is_inh, N_POST, axis=1),
    ),
                 dtype=bool)

    # Compute the weights
    Js_tar = net["Js_train_post_tar"][idx_fun]
    if iTh is False:
        Js_tar = np.clip(Js_tar, 0.0, None)
    W = np.zeros((N_PRE, N_POST))
    WE, WI = bioneuronqp.solve(net["As_train_pre"],
                               Js_tar,
                               np.array([0, 1, -1, 1, 0, 0]),
                               C,
                               iTh=iTh if not iTh is False else None,
                               renormalise=False,
                               reg=reg,
                               n_threads=1,
                               progress_callback=None,
                               warning_callback=None)
    W = WE - WI

    # Return the weights
    return idx_fun, idx_reg, idx_th, idx_ratio, idx_net, W


def main():
    with h5py.File(
            os.path.join(os.path.dirname(__file__), '..', '..', 'data',
                         'subthreshold_relaxation_experiment.h5'), 'w') as f:
        # Store the parameters
        f.create_dataset('regs', data=REGS)
        f.create_dataset('ths', data=THS[2:])
        f.create_dataset('ratios', data=RATIOS)

        # Store the target functions
        xs = np.linspace(-1, 1, N_SMPLS_TEST)
        f.create_dataset('xs', data=xs)
        DS_YS = f.create_dataset('ys', (N_SMPLS_TEST, N_FUNCTIONS),
                                 compression="lzf")
        for i in range(N_FUNCTIONS):
            DS_YS[:, i] = FUNCTIONS[i](xs)

        # Store the pre-activities and decoders
        print("Storing decoders and test pre-activities...")
        DS_D = f.create_dataset('D', (N_REPEAT, N_POST), compression="lzf")
        DS_AS_TEST_PRE = f.create_dataset('as_test_pre',
                                          (N_REPEAT, N_SMPLS_TEST, N_PRE),
                                          compression="lzf")
        DS_JS_TEST_POST = f.create_dataset('js_test_post',
                                          (N_REPEAT, N_FUNCTIONS, N_SMPLS_TEST, N_POST),
                                          compression="lzf")
        with env_guard.SingleThreadEnvGuard():
            with multiprocessing.get_context('spawn').Pool() as pool:
                for i, net in tqdm.tqdm(pool.imap_unordered(
                        make_network, range(N_REPEAT)),
                                        total=N_REPEAT):
                    DS_D[i] = net['D']
                    DS_AS_TEST_PRE[i] = net['As_test_pre']
                    DS_JS_TEST_POST[i] = net['Js_test_post']

        # Assemble the parameter list
        params = list(
            itertools.product(range(N_FUNCTIONS), range(N_REGS), range(N_THS),
                              range(N_RATIOS), range(N_REPEAT)))
        random.shuffle(params)

        # Compute the connection weights
        print("Computing connection weights")
        DS_WS = f.create_dataset(
            'ws',
            (N_FUNCTIONS, N_REGS, N_THS, N_RATIOS, N_REPEAT, N_PRE, N_POST),
            compression="lzf")
        with env_guard.SingleThreadEnvGuard():
            with multiprocessing.get_context('spawn').Pool() as pool:
                for i, j, k, l, m, W in tqdm.tqdm(pool.imap_unordered(
                        compute_weights, params),
                                                  total=len(params)):
                    DS_WS[i, j, k, l, m] = W


if __name__ == "__main__":
    main()

