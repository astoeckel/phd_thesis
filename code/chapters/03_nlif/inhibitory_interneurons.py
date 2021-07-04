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


def forkrng(rng=np.random):
    return np.random.RandomState(rng.randint((1 << 31)))


def single_inhibitory_interneuron_experiment(N_pre=101,
                                             N_post=102,
                                             N_smpls=103,
                                             N_test=1001,
                                             f=lambda x: x,
                                             sigma=0.1,
                                             rng=np.random):
    gains_pre1, biases_pre1, encoders_pre1 = mk_ensemble(N_pre,
                                                         rng=forkrng(rng))
    gains_pre2, biases_pre2, encoders_pre2 = mk_ensemble(N_pre,
                                                         rng=forkrng(rng))

    gains_post, biases_post, encoders_post = mk_ensemble(N_post,
                                                         rng=forkrng(rng))

    # Sample the source space
    xs = np.linspace(-1, 1, N_smpls).reshape(-1, 1)

    # Determine the post-population decoder
    Js_post = gains_post[None, :] * (xs @ encoders_post.T) + biases_post[
        None, :]
    As_post = lif_utils.lif_rate(Js_post)
    A = (As_post.T @ As_post +
         N_smpls * np.square(sigma * np.max(As_post)) * np.eye(N_post))
    Y = As_post.T @ xs[:, 0]
    D_post = np.linalg.solve(A, Y)

    # Determine the pre-activities
    Js_pre1 = gains_pre1[None, :] * (xs @ encoders_pre1.T) + biases_pre1[
        None, :]
    As_pre1 = lif_utils.lif_rate(Js_pre1)

    # Compute the connection weights from pre1 to pre2
    Js_pre2 = gains_pre2[None, :] * (xs @ encoders_pre2.T) + biases_pre2[
        None, :]
    W1 = decode_currents(As_pre1, Js_pre2, p_exc=1.0, sigma=sigma, rng=None)

    # Compute the combined activities
    Js_pre2_dec = As_pre1 @ W1.T
    As_pre2 = lif_utils.lif_rate(Js_pre2_dec)
    As_pre = np.concatenate((As_pre1, As_pre2), axis=1)

    # Compute the combined weight matrix
    ys = f(xs)
    Js_post = gains_post[None, :] * (ys @ encoders_post.T) + biases_post[
        None, :]
    W2 = decode_currents(As_pre,
                         Js_post,
                         p_exc=None,
                         is_exc=np.arange(2 * N_pre) < N_pre,
                         is_inh=np.arange(2 * N_pre) >= N_pre,
                         sigma=sigma,
                         rng=None)

    # Generate the test data
    xs = np.linspace(-1, 1, N_test).reshape(-1, 1)
    ys = f(xs)
    ys_rms = np.sqrt(np.mean(np.square(ys)))

    # Compute the activities of the first pre-population
    Js_pre1 = gains_pre1[None, :] * (xs @ encoders_pre1.T) + biases_pre1[
        None, :]
    As_pre1 = lif_utils.lif_rate(Js_pre1)

    # Compute the activities of the second pre-population
    As_pre2 = lif_utils.lif_rate(As_pre1 @ W1.T)

    # Assemble the combined activities
    As_pre = np.concatenate((As_pre1, As_pre2), axis=1)

    # Compute the final activities
    As_post_dec = lif_utils.lif_rate(As_pre @ W2.T)

    # Compute the error of the decoded signal
    ys_rms = np.sqrt(np.mean(np.square(ys)))
    ys_dec = As_post_dec @ D_post.T
    E_D = np.sqrt(np.mean(np.square(ys_dec - ys[:, 0])))

    return xs[:, 0], ys[:, 0], ys_dec, E_D / ys_rms


def single_inhibitory_communication_channel_experiment(N_pre=101,
                                                       N_post=102,
                                                       N_smpls=103,
                                                       N_test=1001,
                                                       f=lambda x: x,
                                                       sigma=0.1,
                                                       rng=np.random):
    gains_pre1, biases_pre1, encoders_pre1 = mk_ensemble(N_pre,
                                                         rng=forkrng(rng))
    gains_pre2, biases_pre2, encoders_pre2 = mk_ensemble(N_pre,
                                                         rng=forkrng(rng))

    gains_post, biases_post, encoders_post = mk_ensemble(N_post,
                                                         rng=forkrng(rng))

    # Determine the post-population decoder
    xs = np.linspace(-1, 1, N_smpls).reshape(-1, 1)
    Js_post = gains_post[None, :] * (xs @ encoders_post.T) + biases_post[
        None, :]
    As_post = lif_utils.lif_rate(Js_post)
    A = (As_post.T @ As_post +
         N_smpls * np.square(sigma * np.max(As_post)) * np.eye(N_post))
    Y = As_post.T @ xs[:, 0]
    D_post = np.linalg.solve(A, Y)

    # Sample the source space
    xs1 = np.linspace(-1, 1, N_smpls)
    xs2 = np.linspace(-1, 1, N_smpls)
    xss1, xss2 = np.meshgrid(xs1, xs2)
    idcs = forkrng(rng).randint(0, N_smpls * N_smpls, N_smpls)
    xs1 = xss1.flatten()[idcs].reshape(-1, 1)
    xs2 = xss2.flatten()[idcs].reshape(-1, 1)

    # Determine the pre-activities
    Js_pre1 = gains_pre1[None, :] * (xs1 @ encoders_pre1.T) + biases_pre1[
        None, :]
    Js_pre2 = gains_pre2[None, :] * (xs2 @ encoders_pre2.T) + biases_pre2[
        None, :]
    As_pre1 = lif_utils.lif_rate(Js_pre1)
    As_pre2 = lif_utils.lif_rate(Js_pre2)

    # Compute the combined activities
    As_pre = np.concatenate((As_pre1, As_pre2), axis=1)

    # Compute the combined weight matrix
    ys = f(xs1)
    Js_post = gains_post[None, :] * (ys @ encoders_post.T) + biases_post[
        None, :]
    W = decode_currents(As_pre,
                        Js_post,
                        p_exc=None,
                        is_exc=np.arange(2 * N_pre) >= N_pre,
                        is_inh=np.arange(2 * N_pre) < N_pre,
                        sigma=sigma,
                        rng=None)

    # Generate the test data
    idcs = np.random.RandomState(582131).randint(0, N_smpls * N_smpls, N_test)
    xs1 = xss1.flatten()[idcs].reshape(-1, 1)
    xs2 = xss2.flatten()[idcs].reshape(-1, 1)

    isort = np.argsort(xs1[:, 0])
    xs1 = xs1[isort]
    xs2 = xs2[isort]

    ys = f(xs1)
    ys_rms = np.sqrt(np.mean(np.square(ys)))

    # Compute the activities of the first pre-population
    Js_pre1 = gains_pre1[None, :] * (xs1 @ encoders_pre1.T) + biases_pre1[
        None, :]
    Js_pre2 = gains_pre2[None, :] * (xs2 @ encoders_pre2.T) + biases_pre2[
        None, :]
    As_pre1 = lif_utils.lif_rate(Js_pre1)
    As_pre2 = lif_utils.lif_rate(Js_pre2)

    # Assemble the combined activities
    As_pre = np.concatenate((As_pre1, As_pre2), axis=1)

    # Compute the final activities
    As_post_dec = lif_utils.lif_rate(As_pre @ W.T)

    # Compute the error of the decoded signal
    ys_rms = np.sqrt(np.mean(np.square(ys)))
    ys_dec = As_post_dec @ D_post.T
    E_D = np.sqrt(np.mean(np.square(ys_dec - ys[:, 0])))

    return xs1[:, 0], ys[:, 0], ys_dec, E_D / ys_rms


FUNCTIONS = [lambda x: x, lambda x: 2.0 * np.square(x) - 1.0]
MODES = ["inter", "inhcom"]


def run_single(args):
    i, j, k = args

    f = FUNCTIONS[i]
    mode = MODES[j]

    if mode == "inter":
        res = single_inhibitory_interneuron_experiment(
            rng=np.random.RandomState(4091 * k + 127), f=f)
    elif mode == "inhcom":
        res = single_inhibitory_communication_channel_experiment(
            rng=np.random.RandomState(4091 * k + 127), f=f)

    return i, j, k, res


def main():
    N_repeat = 1001
    N_fs = len(FUNCTIONS)
    N_modes = len(MODES)

    params = [(i, j, k) for i in range(N_fs) for j in range(N_modes)
              for k in range(N_repeat)]
    random.shuffle(params)

    with h5py.File(
            os.path.join('data', 'inhibitory_interneurons.h5'), 'w') as f:

        xss = f.create_dataset('xss', (N_fs, N_modes, N_repeat, 1001))
        yss = f.create_dataset('yss', (N_fs, N_modes, N_repeat, 1001))
        ys_decs = f.create_dataset('ys_decs', (N_fs, N_modes, N_repeat, 1001))
        nrmses = f.create_dataset('nrmses', (N_fs, N_modes, N_repeat))

        with env_guard.SingleThreadEnvGuard():
            with multiprocessing.get_context('spawn').Pool() as pool:
                for i, j, k, res in tqdm.tqdm(pool.imap_unordered(
                        run_single, params),
                                              total=len(params)):

                    xs, ys, ys_dec, nrmse = res

                    xss[i, j, k] = xs
                    yss[i, j, k] = ys
                    ys_decs[i, j, k] = ys_dec
                    nrmses[i, j, k] = nrmse


if __name__ == "__main__":
    main()

