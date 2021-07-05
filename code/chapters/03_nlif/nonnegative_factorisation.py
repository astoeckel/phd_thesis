#!/usr/bin/env python3

import random
import multiprocessing

import h5py
import tqdm

import numpy as np
import sys, os

import lif_utils
import env_guard
from nonneg_common import *

def forkrng(rng=np.random):
    return np.random.RandomState(rng.randint((1 << 31)))


def single_experiment(N_pre=101,
                      N_post=101,
                      N_smpls=101,
                      N_test=1001,
                      f=lambda x: x,
                      p_exc=0.5,
                      do_dale=True,
                      do_decode_bias=True,
                      sigma=0.1,
                      rng=np.random):
    gains_pre, biases_pre, encoders_pre = mk_ensemble(N_pre, rng=forkrng(rng))
    gains_post, biases_post, encoders_post = mk_ensemble(N_post,
                                                         rng=forkrng(rng))

    # Sample the source space
    xs = np.linspace(-1, 1, N_smpls).reshape(-1, 1)
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
    W_exc, W_inh, idcs_exc, idcs_inh = decode_currents(As_pre,
                                                       Js_tar,
                                                       p_exc=p_exc,
                                                       do_dale=do_dale,
                                                       sigma=sigma,
                                                       split=True,
                                                       rng=forkrng(rng))

    # Compute a SVD of the matrix
    max_d = min(N_pre, N_post)

    try:
        U_exc, S_exc, V_exc = np.linalg.svd(W_exc)
        U_inh, S_inh, V_inh = np.linalg.svd(W_inh)
    except np.linalg.LinAlgError:
        return np.ones(max_d) * np.nan
    N_exc = len(idcs_exc)
    N_inh = len(idcs_inh)

    # Generate the test samples
    xs = np.linspace(-1, 1, N_test).reshape(-1, 1)
    ys = f(xs)
    ys_rms = np.sqrt(np.mean(np.square(ys)))
    Js_pre = gains_pre[None, :] * (xs @ encoders_pre.T) + biases_pre[None, :]
    As_pre = lif_utils.lif_rate(Js_pre)

    errs = np.zeros(max_d)
    for d in range(max_d):
        d_exc = min(N_exc, d)
        d_inh = min(N_inh, d)
        W_exc_rec = U_exc[:, :d_exc] @ np.diag(S_exc[:d_exc]) @ V_exc[:d_exc]
        W_inh_rec = U_inh[:, :d_inh] @ np.diag(S_inh[:d_inh]) @ V_inh[:d_inh]

        Js_exc_dec = np.clip(As_pre[:, idcs_exc] @ W_exc_rec.T, 0, None)
        Js_inh_dec = np.clip(As_pre[:, idcs_inh] @ W_inh_rec.T, 0, None)
        Js_dec = Js_exc_dec - Js_inh_dec

        As_dec = lif_utils.lif_rate(
            Js_dec + (0 if do_decode_bias else biases_post[None, :]))
        errs[d] = np.sqrt(np.mean(
            np.square(As_dec @ D_post.T - ys[:, 0]))) / ys_rms

    return errs


FUNCTIONS = [lambda x: x, lambda x: 2.0 * np.square(x) - 1.0]


def run_single(args):
    i, do_bias, j, k, p_exc = args

    rng = np.random.RandomState(381921 * k + 187)
    res = single_experiment(f=FUNCTIONS[j],
                            p_exc=p_exc,
                            do_decode_bias=do_bias,
                            rng=rng)

    return i, do_bias, j, k, res


def main():

    N_smpls = 31
    N_repeat = 1001
    N_fs = len(FUNCTIONS)
    p_excs = np.linspace(0, 1, N_smpls)

    params = [(i, do_bias, j, k, p_exc) for i, p_exc in enumerate(p_excs)
              for do_bias in [True, False] for j in range(N_fs)
              for k in range(N_repeat)]

    random.shuffle(params)

    with h5py.File(
            os.path.join('data', 'nonnegative_factorisation.h5'), 'w') as f:
        f.create_dataset('p_excs', data=p_excs)
        errs = f.create_dataset('errs', (101, N_smpls, 2, N_fs, N_repeat))
        errs[...] = np.nan

        with env_guard.SingleThreadEnvGuard():
            with multiprocessing.get_context('spawn').Pool(
                    multiprocessing.cpu_count()) as pool:
                for i, do_bias, j, k, res in tqdm.tqdm(pool.imap_unordered(
                        run_single, params),
                                                       total=len(params)):
                    errs[:, i, int(do_bias), j, k] = res


if __name__ == "__main__":
    main()

