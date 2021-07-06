#!/usr/bin/env python3

import itertools
import multiprocessing
import sys, os

import numpy as np

import random

import scipy.optimize

import lif_utils
import env_guard
import nonneg_common

import bioneuronqp

import tqdm
import h5py

# Number of evaluations to determine regularisation factors
N_BUDGET = 100

# Maximum firing rate (hard-coded in nonneg_common.mk_ensemble)
A_MAX = 100

# Number of repetitions for each parameter set (i.e., number of networks)
N_REPEAT = 100

# Number of networks to explore for the regularisation sweep
N_REPEAT_EXPLORE = 20

# Number of noise parameters to use
N_SIGMAS = 11
SIGMAS = np.logspace(-3, 0, N_SIGMAS)

# Number of noise passes
N_NOISE_PASSES = 9

# Number of different thresholds to try
N_THS = 7
THS = [None, False] + list(np.linspace(1, 0, N_THS - 2))[::-1]

# Functions to analyse
FUNCTIONS = [
    lambda x: x,
    lambda x: 2.0 * np.square(x) - 1.0,
    lambda x: 3 * (x - 0.75) *(x - 0.0) * (x + 0.75)
]
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


def rms(x):
    return np.sqrt(np.mean(np.square(x)))


def rmse(x, y):
    return rms(x - y)


def make_network(idx, sigma=0.01):
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
    Js_train_post = α_post[None, :] * (xs_train @ E_post.T) + β_post[None, :]
    As_train_post = lif_utils.lif_rate(Js_train_post)
    A = (As_train_post.T @ As_train_post +
         N_SMPLS_TRAIN * np.square(sigma * A_MAX) * np.eye(N_POST))
    Y = As_train_post.T @ xs_train[:, 0]
    D = np.linalg.solve(A, Y)

    # For each function, determine the target activities
    ys_train_tar = [f(xs_train)[:, 0] for f in FUNCTIONS]
    Js_train_tar = [
        α_post[None, :] * (f(xs_train) @ E_post.T) + β_post[None, :]
        for f in FUNCTIONS
    ]

    # Compute the test pre-activities
    xs_test = np.linspace(-1, 1, N_SMPLS_TEST).reshape(-1, 1)
    Js_test_pre = α_pre[None, :] * (xs_test @ E_pre.T) + β_pre[None, :]
    As_test_pre = lif_utils.lif_rate(Js_test_pre)

    # Compute the target currents and the target decoded output
    ys_test_tar = [f(xs_test)[:, 0] for f in FUNCTIONS]
    Js_test_tar = [
        α_post[None, :] * (ys.reshape(-1, 1) @ E_post.T) + β_post[None, :]
        for ys in ys_test_tar
    ]

    return idx, {
        "As_train_pre": As_train_pre,
        "ys_train_tar": ys_train_tar,
        "Js_train_tar": Js_train_tar,
        "D": D,
        "As_test_pre": As_test_pre,
        "ys_test_tar": ys_test_tar,
        "Js_test_tar": Js_test_tar
    }


def evaluate_network(W,
                     D,
                     As_test_pre,
                     Js_test_tar,
                     ys_test_tar,
                     sigma=0.0,
                     rng=np.random,
                     eval_current=True):
    # Add some noise to the pre-activities
    if sigma == 0.0:
        As_test_pre_noise = As_test_pre
    else:
        noise = rng.normal(0, sigma * A_MAX, As_test_pre.shape)
        As_test_pre_noise = (As_test_pre + noise)

    # Compute the post-activities
    Js_post = As_test_pre_noise @ W
    As_post = lif_utils.lif_rate(Js_post)

    # Decode the function
    ys_dec = As_post @ D

    # Compute the decoding error
    E_dec = rmse(ys_dec, ys_test_tar) / rms(ys_test_tar)

    # Compute the current-space error
    if eval_current:
        iTh = 1.0
        is_sup = Js_test_tar > iTh
        is_inv = np.logical_and(Js_post > iTh, ~is_sup)
        E_cur = rms(is_sup * (Js_post - Js_test_tar) + is_inv *
                    (Js_post - iTh)) / rms(Js_test_tar[is_sup])

        return E_dec, E_cur
    else:
        return E_dec


def compute_weights(reg, iTh, conn, As_train_pre, Js_train_tar):
    def mkreg(reg):
        return np.square(np.power(10.0, reg) * A_MAX)

    # Clip the weights if iTh is set to "False"
    if iTh is False:
        Js_train_tar = np.clip(Js_train_tar, 0.0, None)
        iTh = None

    # Compute the weights
    WE, WI = bioneuronqp.solve(As_train_pre,
                               Js_train_tar,
                               np.array([0, 1, -1, 1, 0, 0]),
                               conn,
                               iTh=iTh,
                               renormalise=False,
                               reg=mkreg(reg),
                               n_threads=1,
                               progress_callback=None,
                               warning_callback=None)

    return WE - WI


def compute_reg_wrap(params):
    # Unpack the parameters
    idx_fun, idx_th, idx_ratio, idx_sigma = params

    # Fetch the parameters
    iTh = THS[idx_th]
    p_exc = RATIOS[idx_ratio]
    sigma = SIGMAS[idx_sigma]

    # Generate all networks
    N = min(N_REPEAT_EXPLORE, N_REPEAT)
    nets, conns = [None] * N, [None] * N
    for idx_net in range(N):
        # Assemble the connectivity matrix
        rng = np.random.RandomState(67349 * idx_net + 13109)
        is_exc = rng.choice([True, False], (N_PRE, 1), p=[p_exc, 1.0 - p_exc])
        is_inh = ~is_exc
        conn = np.array((np.repeat(is_exc, N_POST,
                                   axis=1), np.repeat(is_inh, N_POST, axis=1)),
                        dtype=bool)

        _, nets[idx_net] = make_network(idx_net)
        conns[idx_net] = conn

    lst_regs = []
    lst_errs = []

    def objective_fun(reg):
        # Evaluate all networks
        Es = []
        for idx_net in range(N):
            # Compute the weights
            W = compute_weights(reg, iTh, conns[idx_net],
                                nets[idx_net]["As_train_pre"],
                                nets[idx_net]["Js_train_tar"][idx_fun])

            # Evaluate the network
            for i in range(N_NOISE_PASSES):
                rng = np.random.RandomState(47699 * i + 2143)
                E = evaluate_network(
                    W=W,
                    D=nets[idx_net]["D"],
                    As_test_pre=nets[idx_net]["As_train_pre"],
                    Js_test_tar=nets[idx_net]["Js_train_tar"][idx_fun],
                    ys_test_tar=nets[idx_net]["ys_train_tar"][idx_fun],
                    sigma=sigma,
                    rng=rng,
                    eval_current=False)
                Es.append(E)

        err = np.mean(Es)

        lst_regs.append(reg)
        lst_errs.append(err)

        return err + 1e-6 * reg

    # Compute the regularisation factor
    reg = scipy.optimize.fminbound(objective_fun, -3, 0, maxfun=N_BUDGET, xtol=1e-3)

    # Return the weights
    return idx_fun, idx_th, idx_ratio, idx_sigma, reg


def compute_weights_wrap(params):
    # Unpack the parameters
    idx_fun, idx_th, idx_ratio, idx_sigma, idx_net, reg = params

    # Fetch the parameters
    iTh = THS[idx_th]
    p_exc = RATIOS[idx_ratio]
    sigma = SIGMAS[idx_sigma]

    # Generate the network
    _, net = make_network(idx_net)

    # Arbitrarily decide whether the pre neurons are excitatory or inhibitory
    rng = np.random.RandomState(67349 * idx_net + 13109)
    is_exc = rng.choice([True, False], (N_PRE, 1), p=[p_exc, 1.0 - p_exc])
    is_inh = ~is_exc

    # Assemble the connectivity matrix
    conn = np.array(
        (np.repeat(is_exc, N_POST, axis=1), np.repeat(is_inh, N_POST, axis=1)),
        dtype=bool)

    # Solve for the weights
    W = compute_weights(reg, iTh, conn, net["As_train_pre"],
                        net["Js_train_tar"][idx_fun])

    # Compute the errors
    Es_dec, Es_cur = np.zeros((2, N_NOISE_PASSES))
    for i in range(N_NOISE_PASSES):
        rng = np.random.RandomState(47699 * i + 2143)
        Es_dec[i], Es_cur[i] = evaluate_network(
            W=W,
            D=net["D"],
            As_test_pre=net["As_test_pre"],
            Js_test_tar=net["Js_test_tar"][idx_fun],
            ys_test_tar=net["ys_test_tar"][idx_fun],
            sigma=sigma,
            rng=rng)

    # Return the weights
    return idx_fun, idx_th, idx_ratio, idx_sigma, idx_net, (W, Es_dec, Es_cur)


def main():
    with h5py.File(
            os.path.join('data', 'subthreshold_relaxation_experiment_auto_reg.h5'), 'w') as f:
        # Store the parameters
        f.create_dataset('ths', data=THS[2:])
        f.create_dataset('ratios', data=RATIOS)
        f.create_dataset('sigmas', data=SIGMAS)

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
        DS_JS_TEST_TAR = f.create_dataset(
            'js_test_tar', (N_REPEAT, N_FUNCTIONS, N_SMPLS_TEST, N_POST),
            compression="lzf")
        with env_guard.SingleThreadEnvGuard():
            with multiprocessing.get_context('spawn').Pool() as pool:
                for i, net in tqdm.tqdm(pool.imap_unordered(
                        make_network, range(N_REPEAT)),
                                        total=N_REPEAT):
                    DS_D[i] = net['D']
                    DS_AS_TEST_PRE[i] = net['As_test_pre']
                    DS_JS_TEST_TAR[i] = net['Js_test_tar']

        # Compute the regularisation factors first
        DS_REGS = f.create_dataset('regs',
                                   (N_FUNCTIONS, N_THS, N_RATIOS, N_SIGMAS),
                                   compression="lzf")

        print('Determining regularisation factors...')
        params = list(
            itertools.product(range(N_FUNCTIONS), range(N_THS),
                              range(N_RATIOS), range(N_SIGMAS)))
        random.shuffle(params)
        with env_guard.SingleThreadEnvGuard():
            with multiprocessing.get_context('spawn').Pool() as pool:
                for i, j, k, l, reg in tqdm.tqdm(pool.imap_unordered(
                        compute_reg_wrap, params),
                                                 total=len(params)):
                    DS_REGS[i, j, k, l] = reg

        # Compute the connection weights and network evaluations
        params = list(
            itertools.product(range(N_FUNCTIONS),
                              range(N_THS), range(N_RATIOS), range(N_SIGMAS),
                              range(N_REPEAT)))
        params = [(i, j, k, l, m, DS_REGS[i, j, k, l])
                  for i, j, k, l, m in params]
        random.shuffle(params)

        print("Computing connection weights and evaluating...")
        DS_ES_DEC = f.create_dataset(
            'es_dec',
            (N_FUNCTIONS, N_THS, N_RATIOS, N_SIGMAS, N_REPEAT, N_NOISE_PASSES),
            compression="lzf")
        DS_ES_CUR = f.create_dataset(
            'es_cur',
            (N_FUNCTIONS, N_THS, N_RATIOS, N_SIGMAS, N_REPEAT, N_NOISE_PASSES),
            compression="lzf")
        DS_WS = f.create_dataset(
            'ws',
            (N_FUNCTIONS, N_THS, N_RATIOS, N_SIGMAS, N_REPEAT, N_PRE, N_POST),
            compression="lzf")
        with env_guard.SingleThreadEnvGuard():
            with multiprocessing.get_context('spawn').Pool() as pool:
                for i, j, k, l, m, res in tqdm.tqdm(pool.imap_unordered(
                        compute_weights_wrap, params),
                                                    total=len(params)):
                    W, Es_dec, Es_cur = res

                    DS_ES_DEC[i, j, k, l, m] = Es_dec
                    DS_ES_CUR[i, j, k, l, m] = Es_cur
                    DS_WS[i, j, k, l, m] = W


if __name__ == "__main__":
    main()

