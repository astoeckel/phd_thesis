#!/usr/bin/env python3

import random
import multiprocessing
import numpy as np

import h5py
import tqdm

import sys, os

import lif_utils
import env_guard


def mk_ensemble(N,
                x_intercepts_cback=None,
                encoders_cback=None,
                d=1,
                rng=np.random):
    def default_x_intercepts(rng, N):
        return rng.uniform(-0.99, 0.99, N)

    def default_encoders_cback(rng, N, d):
        return rng.normal(0, 1, (N, d))

    x_intercepts_cback = (default_x_intercepts if x_intercepts_cback is None
                          else x_intercepts_cback)
    encoders_cback = (default_encoders_cback
                      if encoders_cback is None else encoders_cback)

    max_rates = rng.uniform(50, 100, N)
    x_intercepts = x_intercepts_cback(rng, N)  #rng.uniform(-0.99, 0.99, N)

    J0s = lif_utils.lif_rate_inv(1e-3)
    J1s = lif_utils.lif_rate_inv(max_rates)

    gains = (J1s - J0s) / (1.0 - x_intercepts)
    biases = (J0s - x_intercepts * J1s) / (1.0 - x_intercepts)

    encoders = encoders_cback(rng, N, d)  #rng.normal(0, 1, (N, d))
    encoders /= np.linalg.norm(encoders, axis=1)[:, None]

    if d == 1:
        idcs = np.argsort(x_intercepts * encoders[:, 0])
    else:
        idcs = np.arange(N)

    return gains[idcs], biases[idcs], encoders[idcs]


def do_measure_error(N_pre,
                     N_post=102,
                     N_smpls=1001,
                     x_intercepts_cback=None,
                     encoders_cback=None,
                     rng=np.random):
    # Generate the pre-activities
    xs = np.linspace(-1, 1, N_smpls).reshape(-1, 1)
    gains_pre, biases_pre, encoders_pre = mk_ensemble(N_pre,
                                                      x_intercepts_cback,
                                                      encoders_cback,
                                                      rng=rng)
    Js_pre = gains_pre[None, :] * (xs @ encoders_pre.T) + biases_pre[None, :]
    As_pre = lif_utils.lif_rate(Js_pre)

    gains_post, biases_post, encoders_post = mk_ensemble(N_post, rng=rng)
    Js_post = gains_post[None, :] * (xs @ encoders_post.T) + biases_post[
        None, :]
    As_post = lif_utils.lif_rate(Js_post)

    # Compute an identity decoder, as well as a bias decoder
    sigma = 0.1 * np.max(As_pre)
    ATA = (As_pre.T @ As_pre + np.square(sigma) * N_smpls * np.eye(N_pre))
    Y_id = As_pre.T @ xs
    Y_one = As_pre.T @ np.ones_like(xs)
    D_id = np.linalg.solve(ATA, Y_id[:, 0])
    D_one = np.linalg.solve(ATA, Y_one[:, 0])

    xs_dec = (As_pre @ D_id).reshape(-1, 1)
    ones_dec = (As_pre @ D_one).reshape(-1, 1)

    err_dec_xs = np.sqrt(np.mean(np.square(xs - xs_dec)))
    err_dec_ones = np.sqrt(np.mean(np.square(1.0 - ones_dec)))

    Js_post_1 = gains_post[None, :] * (xs_dec @ encoders_post.T) + biases_post[
        None, :]
    Js_post_2 = gains_post[None, :] * (
        xs_dec @ encoders_post.T) + biases_post[None, :] * ones_dec

    As_post_1 = lif_utils.lif_rate(Js_post_1)
    As_post_2 = lif_utils.lif_rate(Js_post_2)

    err_As_bias = np.sqrt(np.mean(np.square(As_post_1 - As_post)))
    err_As_dec_bias = np.sqrt(np.mean(np.square(As_post_2 - As_post)))

    return err_dec_xs, err_dec_ones, err_As_bias, err_As_dec_bias


def run_single(args):
    i, j, N_pre = args

    rng = np.random.RandomState(578291 * j + 381)
    errs = do_measure_error(N_pre, rng=rng)

    return i, j, errs


def main():

    N_smpls = 9
    N_repeat = 1000
    N_pres = np.logspace(1, 3, N_smpls, dtype=int)

    params = [(i, j, N_pre) for i, N_pre in enumerate(N_pres)
              for j in range(N_repeat)]
    random.shuffle(params)

    with h5py.File(
            os.path.join('data', 'bias_decoding_impact.h5'), 'w') as f:
        f.create_dataset('N_pres', data=N_pres)
        errs = f.create_dataset('errs', (4, N_smpls, N_repeat))

        with env_guard.SingleThreadEnvGuard():
            with multiprocessing.Pool() as pool:
                for i, j, res in tqdm.tqdm(pool.imap_unordered(
                        run_single, params),
                                           total=len(params)):
                    errs[:, i, j] = res


if __name__ == "__main__":
    main()

