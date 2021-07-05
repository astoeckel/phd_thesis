#!/usr/bin/env python3

import h5py
import numpy as np
import random
import tqdm
import multiprocessing

import os
import sys
import env_guard


def mk_relu_ensemble(N,
                     d,
                     reg=0.2,
                     N_smpls_train=10000,
                     N_smpls_test=10000,
                     rng=np.random):

    rng2 = np.random.RandomState(rng.randint(0, 1 << 16))

    max_rates = rng.uniform(0.5, 1.0, N)
    x_intercepts = rng.uniform(-1.0, 1.0, N)

    gains = max_rates / (1.0 - x_intercepts)
    biases = -x_intercepts * max_rates / (1.0 - x_intercepts)

    encoders = rng.normal(0, 1, (N, d))
    encoders /= np.linalg.norm(encoders, axis=1)[:, None]

    def compute_and_test_decoder(f):
        # Sample the training data from the unit ball
        Xs = rng2.normal(0, 1, (N_smpls_train, d))
        Xs /= np.linalg.norm(Xs, axis=1)[:, None]
        Xs *= np.power(rng2.uniform(0, 1, N_smpls_train)[:, None], 1.0 / d)

        # Determine the target dimensionality
        d_out = f(Xs[0:1]).size
        Ys = np.zeros((N_smpls_train, d_out))
        Ys[...] = f(Xs)

        # Compute the currents and activities
        Js = gains * (Xs @ encoders.T) + biases
        As = np.clip(Js, 0, None)

        # Compute the regularised decoder
        X = As.T @ As + N_smpls_train * np.square(reg) * np.eye(N)
        Y = As.T @ Ys
        D = np.zeros((N, Ys.shape[1]))
        for i in range(Ys.shape[1]):
            D[:, i] = np.linalg.solve(X, Y[:, i])

        # Sample the test data from the unit ball
        Xs = rng2.normal(0, 1, (N_smpls_test, d))
        Xs /= np.linalg.norm(Xs, axis=1)[:, None]
        Xs *= np.power(rng2.uniform(0, 1, N_smpls_test)[:, None], 1.0 / d)

        Ys = np.zeros((N_smpls_train, d_out))
        Ys[...] = f(Xs)

        # Compute the decoding error
        Js = gains * (Xs @ encoders.T) + biases
        As = np.clip(Js, 0, None)
        return np.sqrt(np.mean(np.square(Ys - As @ D))) / np.std(Ys)

    err_id = compute_and_test_decoder(lambda x: x)
    err_prod = compute_and_test_decoder(
        lambda x: np.prod(x, axis=1).reshape(-1, 1))

    return err_id, err_prod


NS = np.logspace(1, 3, 20, dtype=int)
DIMS = [1, 2, 3, 4]
REPEAT = 1


def run_single(args):
    i, j, k = args
    return i, j, k, mk_relu_ensemble(NS[i],
                                     DIMS[j],
                                     rng=np.random.RandomState(4890147 * k +
                                                               48183))


def main():
    args = [(i, j, k) for i in range(len(NS)) for j in range(len(DIMS))
            for k in range(REPEAT)]
    random.shuffle(args)

    with h5py.File(
            os.path.join('data', 'nef_noise_visualisation.h5'), 'w') as f:
        f.create_dataset('NS', data=NS)
        f.create_dataset('DIMS', data=DIMS)
        f.create_dataset('REPEAT', data=REPEAT)
        errs_id = f.create_dataset("errs_id", (len(NS), len(DIMS), REPEAT))
        errs_prod = f.create_dataset("errs_prod", (len(NS), len(DIMS), REPEAT))

        with env_guard.SingleThreadEnvGuard() as guard:
            with multiprocessing.get_context("spawn").Pool(
                    multiprocessing.cpu_count() // 2) as pool:
                for (i, j, k,
                     (E1,
                      E2)) in tqdm.tqdm(pool.imap_unordered(run_single, args),
                                        total=len(args)):
                    errs_id[i, j, k] = E1
                    errs_prod[i, j, k] = E2


if __name__ == "__main__":
    main()

