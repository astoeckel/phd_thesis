#!/usr/bin/env python3

import random
import multiprocessing

import h5py
import tqdm

import numpy as np
import sys, os

sys.path.append(os.path.join('..', 'lib'))
import lif_utils
import env_guard
from nonneg_common import *

import sklearn.linear_model

def forkrng(rng=np.random):
    return np.random.RandomState(rng.randint((1 << 31)))


def sparsity(*matrices, th=1e-6):
    N_total = 0
    N_zero = 0
    for M in matrices:
        N_total += M.size
        N_zero += np.sum(np.abs(M) < th)
    return N_zero / N_total


def single_experiment_sparsity(mode="nnls",
                               N_pre=101,
                               N_post=102,
                               N_smpls=101,
                               N_test=1001,
                               f=lambda x: x,
                               sigma=0.1,
                               p_exc=None,
                               p_reg_l1=None,
                               p_dropout=None,
                               do_decode_bias=False,
                               rng=np.random):
    # Make sure the given parameters make sense
    assert mode in {"nnls", "lasso", "dropout"}
    B_exc = not p_exc is None
    B_reg_l1 = not p_reg_l1 is None
    B_dropout = not p_dropout is None
    if mode == "nnls":
        assert B_exc and not (B_reg_l1 or B_dropout)
    elif mode == "lasso":
        assert B_reg_l1 and not (B_exc or B_dropout)
    elif mode == "dropout":
        assert B_dropout and not (B_exc or B_reg_l1)

    # Generate the pre- and post-populations
    gains_pre, biases_pre, encoders_pre = mk_ensemble(N_pre, rng=forkrng(rng))
    gains_post, biases_post, encoders_post = mk_ensemble(N_post,
                                                         rng=forkrng(rng))

    # Sample the source space
    xs = np.linspace(-1, 1, N_smpls).reshape(-1, 1)

    # Compute the target values
    ys = f(xs)
    ys_rms = np.sqrt(np.mean(np.square(ys)))

    # Determine the pre-activities
    Js_pre = gains_pre[None, :] * (xs @ encoders_pre.T) + biases_pre[None, :]
    As_pre = lif_utils.lif_rate(Js_pre)

    # Determine the post-population decoder
    Js_post = gains_post[None, :] * (xs @ encoders_post.T) + biases_post[
        None, :]
    As_post = lif_utils.lif_rate(Js_post)
    A = (As_post.T @ As_post +
         N_smpls * np.square(sigma * np.max(As_post)) * np.eye(N_post))
    Y = As_post.T @ xs[:, 0]
    D_post = np.linalg.solve(A, Y)

    # Determine the target currents
    Js_post = gains_post[None, :] * (ys @ encoders_post.T) + biases_post[
        None, :]
    if do_decode_bias:
        Js_tar = Js_post
    else:
        Js_tar = Js_post - biases_post[None, :]

    # Compute the weight matrix
    W = np.zeros((N_post, N_pre))
    if mode == 'nnls':
        # Use our NNLS method
        W_exc, W_inh, idcs_exc, idcs_inh = decode_currents(As_pre,
                                                           Js_tar,
                                                           p_exc=p_exc,
                                                           sigma=sigma,
                                                           rng=forkrng(rng),
                                                           split=True)

        # Compute the sparsity of the excitatory and inhibitory matrix
        s = sparsity(W_exc, W_inh)

        # Compute the actual weight matrix for the error computation
        W[:, idcs_exc] += W_exc
        W[:, idcs_inh] -= W_inh
    elif mode == 'lasso':
        # Compute the L2-regularised problem
        ATA = As_pre.T @ As_pre + np.square(
            sigma * np.max(As_pre)) * N_smpls * np.eye(N_pre)
        Y = As_pre.T @ Js_tar

        # Setup a Lasso classifier
        clf = sklearn.linear_model.Lasso(alpha=np.square(p_reg_l1),
                                         max_iter=10000,
                                         tol=1e-4,
                                         fit_intercept=False,
                                         warm_start=True)
        for i in range(N_post):
            # Initialize the solver with the L2 solution. For some reason
            # the solver will have trouble to converge to the true optimum
            # otherwise
            setattr(clf, 'coef_', np.linalg.solve(ATA, Y[:, i]))

            # Perform the regression
            clf.fit(ATA, Y[:, i])

            # Store the computed coefficients!
            W[i] = clf.coef_

        s = sparsity(W)
    elif mode == 'dropout':
        # Compute the L2-regularised problem
        ATA = As_pre.T @ As_pre + np.square(
            sigma * np.max(As_pre)) * N_smpls * np.eye(N_pre)
        Y = As_pre.T @ Js_tar

        # Compute the solution
        for i in range(N_post):
            W[i] = np.linalg.solve(ATA, Y[:, i])

        # Determine which neurons are below the threshold
        mask = np.abs(W) >= np.percentile(np.abs(W), p_dropout)

        # Re-compute the solution with the mask applied
        W = np.zeros((N_post, N_pre))
        for i in range(N_post):
            As_pre2 = As_pre[:, mask[i]]
            N_pre2 = int(np.sum(mask[i]))
            ATA = As_pre2.T @ As_pre2 + np.square(
                sigma * np.max(As_pre)) * N_smpls * np.eye(N_pre2)
            Y = As_pre2.T @ Js_tar[:, i]
            W[i, mask[i]] = np.linalg.solve(ATA, Y)

        s = sparsity(W)

    # Generate the test data
    xs = np.linspace(-1, 1, N_test).reshape(-1, 1)
    ys = f(xs)
    ys_rms = np.sqrt(np.mean(np.square(ys)))

    Js_pre = gains_pre[None, :] * (xs @ encoders_pre.T) + biases_pre[None, :]
    As_pre = lif_utils.lif_rate(Js_pre)

    Js_dec = As_pre @ W.T
    As_dec = lif_utils.lif_rate(
        Js_dec + (0 if do_decode_bias else biases_post[None, :]))

    e = np.sqrt(np.mean(np.square(As_dec @ D_post.T - ys[:, 0]))) / ys_rms

    return s, e


#FUNCTIONS = [lambda x: x, lambda x: 2.0 * np.square(x) - 1.0]
FUNCTIONS = [lambda x: x]
MODES = ["nnls", "dropout", "lasso"]


def run_single(args):
    # i: Mode index
    # j: Parameter index
    # k: Function index
    # l: Do bias index
    # m: Repetition index
    i, j, k, l, m, P = args

    # Assemble the parameters
    mode = MODES[i]
    if mode == "nnls":
        params = {"p_exc": P}
    elif mode == "lasso":
        params = {"p_reg_l1": P}
    elif mode == "dropout":
        params = {"p_dropout": P}

    rng = np.random.RandomState(381921 * m + 187)
    res = single_experiment_sparsity(mode=mode,
                                     f=FUNCTIONS[k],
                                     **params,
                                     do_decode_bias=bool(l),
                                     rng=rng)

    return i, j, k, l, m, res


def main():
    N_smpls = 101
    N_repeat = 101
    N_fs = len(FUNCTIONS)
    N_modes = len(MODES)

    p_excs = np.linspace(0, 1, N_smpls)
    p_dropouts = np.linspace(0, 100, N_smpls)
    p_reg_l1s = np.logspace(1, 3, N_smpls)

    params = ([(0, j, k, l, m, p_excs[j]) for j in range(N_smpls)
               for k in range(N_fs) for l in range(2)
               for m in range(N_repeat)] +  #
              [(1, j, k, l, m, p_dropouts[j]) for j in range(N_smpls)
               for k in range(N_fs) for l in range(2)
               for m in range(N_repeat)] +  #
              [(2, j, k, l, m, p_reg_l1s[j]) for j in range(N_smpls)
               for k in range(N_fs) for l in range(2)
               for m in range(N_repeat)])

    random.shuffle(params)

    with h5py.File(
            os.path.join(os.path.dirname(__file__), '..', '..', 'data',
                         'nonnegative_sparsity3.h5'), 'w') as f:
        f.create_dataset('p_excs', data=p_excs)
        f.create_dataset('p_dropouts', data=p_dropouts)
        f.create_dataset('p_reg_l1s', data=p_reg_l1s)

        errs = f.create_dataset('errs',
                                (2, N_modes, N_smpls, N_fs, 2, N_repeat))

        with env_guard.SingleThreadEnvGuard():
            with multiprocessing.get_context('spawn').Pool() as pool:
                for i, j, k, l, m, res in tqdm.tqdm(pool.imap_unordered(
                        run_single, params),
                                                    total=len(params)):
                    errs[:, i, j, k, l, m] = res


if __name__ == "__main__":
    main()

