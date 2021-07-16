#!/usr/bin/env python3

import os, sys
import numpy as np
import multiprocessing
import h5py
import random
import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "lib"))
import env_guard


def eval_basis(xs, i, O):
    N = xs.shape[1]  # Number of dimensions

    idcs = []
    for _ in range(N):
        idcs.append(i % O)
        i //= O

    res = np.ones(xs.shape[0])
    for i in range(N):
        res *= np.polynomial.Legendre([0] * idcs[i] + [1])(xs[:, i])
    return res


def lstsq(A, Y, sigma=0.1):
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n, m = A.shape
    d = Y.shape[1]
    ATA = A.T @ A + n * np.square(sigma) * np.eye(m)
    D = np.zeros((m, d))
    for i in range(d):
        D[:, i] = np.linalg.solve(ATA, A.T @ Y[:, i])
    return D


def mk_ens(N_neurons, N_dims, rng=np.random):
    max_rates = rng.uniform(0.5, 1.0, N_neurons)
    x_intercepts = rng.uniform(-1.0, 1.0, N_neurons)

    gains = max_rates / (1.0 - x_intercepts)
    biases = -x_intercepts * max_rates / (1.0 - x_intercepts)

    encoders = rng.normal(0, 1, (N_neurons, N_dims))
    encoders /= np.linalg.norm(encoders, axis=1)[:, None]

    return gains, biases, encoders


def mk_activities(Xs, gains, biases, encoders):
    Js = gains * (Xs @ encoders.T) + biases
    As = np.clip(Js, 0, None)
    return As


def run_single(args):
    i_dim, i_basis, i_neurons, i_repeat = args

    N_dims = i_dim + 1
    N_neurons = N_NEURONS[i_neurons]

    rng1 = np.random.RandomState(i_repeat * 438971 + 201)
    gains, biases, encoders = mk_ens(N_neurons, N_dims, rng1)

    rng2 = np.random.RandomState(i_repeat * 19819 + 123)
    Xs_train = rng2.uniform(-1, 1, (N_SMPLS_TRAIN, N_dims))
    As_train = mk_activities(Xs_train, gains, biases, encoders)
    Ys_train = eval_basis(Xs_train, i_basis, N_ORDER)
    D = lstsq(As_train, Ys_train)

    Xs_test = rng2.uniform(-1, 1, (N_SMPLS_TEST, N_dims))
    As_test = mk_activities(Xs_test, gains, biases, encoders)
    Ys_test = eval_basis(Xs_test, i_basis, N_ORDER)

    Ys_dec = As_test @ D

    rms = np.sqrt(np.mean(np.square(Ys_test)))
    rmse = np.sqrt(np.mean(np.square(Ys_test.flatten() - Ys_dec.flatten())))

    return i_dim, i_basis, i_neurons, i_repeat, rmse / rms


N_ORDER = 20
N_SMPLS_TRAIN = 1000
N_SMPLS_TEST = 1000
N_REPEAT = 10
N_DIMS = 3

N_NEURONS = np.unique(np.geomspace(10, 1000, 60, dtype=np.int))
N_NEURONS_LEN = len(N_NEURONS)


def main():
    fn = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'data',
        f'decode_basis_functions.h5')

    with h5py.File(fn, 'w') as f:
        f.create_dataset('N_NEURONS', data=N_NEURONS)
        f.create_dataset('N_DIMS', data=N_DIMS)

        Es, args = [], []
        for i in range(N_DIMS):
            N_BASIS_FUNS = N_ORDER**(i + 1)

            Es.append(
                f.create_dataset(f'E{i}',
                                 shape=(N_BASIS_FUNS, N_NEURONS_LEN,
                                        N_REPEAT)))

            args += [(i, i_basis, i_neurons, i_repeat)
                     for i_basis in range(N_BASIS_FUNS)
                     for i_neurons in range(N_NEURONS_LEN)
                     for i_repeat in range(N_REPEAT)]

        random.shuffle(args)

        with env_guard.SingleThreadEnvGuard() as guard:
            with multiprocessing.get_context("spawn").Pool() as pool:
                for (i_dim, i_basis, i_neurons, i_repeat,
                     E) in tqdm.tqdm(pool.imap_unordered(run_single, args),
                                     total=len(args)):
                    Es[i_dim][i_basis, i_neurons, i_repeat] = E


if __name__ == "__main__":
    main()

