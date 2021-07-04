#!/usr/bin/env python3

import random
import multiprocessing

import h5py
import tqdm

import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
import lif_utils
import env_guard
from nonneg_common import *

import matplotlib.pyplot as plt
import scipy.optimize


def forkrng(rng=np.random):
    return np.random.RandomState(rng.randint((1 << 31)))


def single_experiment(N_pre=101,
                      N_post=101,
                      N_smpls=1001,
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

    # Compute the target values
    ys = f(xs)

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

    # Determine the target activities
    As_post = lif_utils.lif_rate(Js_post)

    # Compute the weight matrix
    W = decode_currents(As_pre,
                        Js_tar,
                        p_exc=p_exc,
                        do_dale=do_dale,
                        sigma=sigma,
                        rng=forkrng(rng))

    # Compute the decoded currents
    Js_dec = As_pre @ W.T

    # Compute and return the RMSE and NRMSE
    Js_tar_rms = np.sqrt(np.mean(np.square(Js_tar)))
    E_J = np.sqrt(np.mean(np.square(Js_dec - Js_tar)))

    # Compute the error between the target activities and the actual activities
    As_dec = lif_utils.lif_rate(
        Js_dec + (0 if do_decode_bias else biases_post[None, :]))
    E_A = np.sqrt(np.mean(np.square(As_dec - As_post)))

    # Generate the test data
    xs = np.linspace(-1, 1, N_test).reshape(-1, 1)
    ys = f(xs)
    ys_rms = np.sqrt(np.mean(np.square(ys)))

    Js_pre = gains_pre[None, :] * (xs @ encoders_pre.T) + biases_pre[None, :]
    As_pre = lif_utils.lif_rate(Js_pre)

    Js_dec = As_pre @ W.T
    As_dec = lif_utils.lif_rate(
        Js_dec + (0 if do_decode_bias else biases_post[None, :]))

    # Compute the error of the decoded signal
    ys_rms = np.sqrt(np.mean(np.square(ys)))
    E_D = np.sqrt(np.mean(np.square(As_dec @ D_post.T - ys[:, 0])))

    return E_J, E_J / Js_tar_rms, E_A, E_D, E_D / ys_rms


FUNCTIONS = [lambda x: x, lambda x: 2.0 * np.square(x) - 1.0]


def run_single(args):
    i, do_dale, do_bias, j, k, p_exc = args

    rng = np.random.RandomState(381921 * k + 187)
    res = single_experiment(f=FUNCTIONS[j],
                            p_exc=p_exc,
                            do_dale=do_dale,
                            do_decode_bias=do_bias,
                            rng=rng)

    return i, do_dale, do_bias, j, k, res


def main():

    N_smpls = 31
    N_repeat = 1001
    N_fs = len(FUNCTIONS)
    p_excs = np.linspace(0, 1, N_smpls)

    params = [(i, do_dale, do_bias, j, k, p_exc)
              for i, p_exc in enumerate(p_excs) for do_dale in [True, False]
              for do_bias in [True, False] for j in range(N_fs)
              for k in range(N_repeat)]

    # Do not repeat the same experiment over and over again if do_dale is False
    params = list(filter(lambda x: x[0] == 0 or x[1],
                         params))  # i == 0 or do_dale

    random.shuffle(params)

    with h5py.File(
            os.path.join(os.path.dirname(__file__), '..', '..', 'data',
                         'nonnegative_experiment_large5.h5'), 'w') as f:
        f.create_dataset('p_excs', data=p_excs)
        errs = f.create_dataset('errs', (5, N_smpls, 2, 2, N_fs, N_repeat))
        errs[...] = np.nan

        with env_guard.SingleThreadEnvGuard():
            with multiprocessing.get_context('spawn').Pool(multiprocessing.cpu_count() // 2) as pool:
                for i, do_dale, do_bias, j, k, res in tqdm.tqdm(
                        pool.imap_unordered(run_single,
                                            params), total=len(params)):
                    errs[:, i, int(do_dale), int(do_bias), j, k] = res


if __name__ == "__main__":
    main()

