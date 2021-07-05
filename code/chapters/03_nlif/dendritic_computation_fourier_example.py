#!/usr/bin/env python3

import dlop_ldn_function_bases as bases
import numpy as np
import random
import h5py
import multiprocessing
import tqdm
import scipy.optimize

import os
import sys

import env_guard

# Number of basis functions
d = 16

# Number of samples
N = 32

# Generate the basis
A = bases.mk_fourier_basis(d, N) * np.sqrt(0.5 * N)
A2D = np.einsum('ij,kl->ikjl', A, A)
A1D = np.concatenate((
    A2D[0, :].reshape(d, -1),
    A2D[:, 0].reshape(d, -1),
)).T


def lstsq(A, Y, sigma=1e-2):
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n, m = A.shape
    d = Y.shape[1]
    ATA = A.T @ A + n * np.square(sigma) * np.eye(m)
    D = np.zeros((m, d))
    for i in range(d):
        D[:, i] = np.linalg.solve(ATA, A.T @ Y[:, i])
    return D

def solve_additive(basis, tar):
    return basis @ lstsq(basis, tar)


def solve_multiplicative(basis, tar, rng):
    def f(w):
        w1, w2 = w.reshape(2, -1)
        return (basis @ w1) * (basis @ w2)

    def E(w):
        return np.mean(np.square(f(w) - tar)) + 1e-1 * np.mean(np.square(w))


    Ds, Es = [[None] * 10 for _ in range(2)]
    for i in range(len(Ds)):
        w = rng.randn(4 * d)
        Ds[i] = scipy.optimize.minimize(E, w, tol=1e-3, method='BFGS').x
        Es[i] = E(Ds[i])
    return f(Ds[np.argmin(Es)])


def solve_mlp(Ys, N_neurons=1000, rng=np.random):

    max_rates = rng.uniform(0.5, 1.0, N_neurons)
    x_intercepts = rng.uniform(-1.0, 1.0, N_neurons)

    gains = max_rates / (1.0 - x_intercepts)
    biases = -x_intercepts * max_rates / (1.0 - x_intercepts)

    encoders = rng.normal(0, 1, (N_neurons, 2))
    encoders /= np.linalg.norm(encoders, axis=1)[:, None]

    # Sample the mesh
    xs = np.linspace(-1, 1, N)
    xss, yss = np.meshgrid(xs, xs)
    Xs = np.array((xss.flatten(), yss.flatten())).T

    # Compute the currents and activities
    Js = gains * (Xs @ encoders.T) + biases
    As = np.clip(Js, 0, None)

    # Compute the regularised decoder
    D = lstsq(As, Ys)
    return As @ D


def coeffs(d, sigma, mu=0.0, rng=np.random):
    ds = np.arange(0, d)
    mask = np.exp(-np.square(ds - mu) / np.square(sigma + 1e-3))
    mask2d = np.outer(mask, mask)
    mask2d /= np.sum(mask2d)
    res = rng.normal(0, 1, (d, d)) * mask2d
    res /= d * np.sqrt(np.mean(np.square(res)))
    return res


def run_single(args):
    i, j, sigma = args
    #i, j, mu = args

    rng = np.random.RandomState(34891 * j + 480)
    #X = np.einsum('ij,ijkl->kl', coeffs(d, 1.0, mu, rng), A2D)
    X = np.einsum('ij,ijkl->kl', coeffs(d, sigma, rng=rng), A2D)
    rms = np.sqrt(np.mean(np.square(X)))

    Y1 = solve_additive(A1D, X.reshape(-1)).reshape(N, N)

    rng = np.random.RandomState(34891 * j + 481)
    Y2 = solve_multiplicative(A1D, X.reshape(-1), rng=rng).reshape(N, N)

    rng = np.random.RandomState(34891 * j + 482)
    Y3 = solve_mlp(X.reshape(-1), rng=rng).reshape(N, N)

    E1 = np.sqrt(np.mean(np.square(X - Y1))) / rms
    E2 = np.sqrt(np.mean(np.square(X - Y2))) / rms
    E3 = np.sqrt(np.mean(np.square(X - Y3))) / rms

    return i, j, (E1, E2, E3)


def main():
    N_REPEAT = 10
    N_TRIALS = 20
    SIGMAS = np.logspace(-0.4, 1, N_TRIALS)
    #    MUS = np.linspace(0, 3, N_TRIALS)

    #args = [(i, j, mu) for j in range(N_REPEAT) for i, mu in enumerate(MUS)]
    args = [(i, j, sigma) for j in range(N_REPEAT) for i, sigma in enumerate(SIGMAS)]
    random.shuffle(args)

    with h5py.File(
            os.path.join(os.path.dirname(__file__), '..', '..', 'data',
                         'dendritic_computation_fourier_example.h5'),
            'w') as f:
        f.create_dataset('SIGMAS', data=SIGMAS)
        #f.create_dataset('MUS', data=MUS)
        f.create_dataset('REPEAT', data=N_REPEAT)
        Es_add = f.create_dataset("Es_add", (N_TRIALS, N_REPEAT))
        Es_mul = f.create_dataset("Es_mul", (N_TRIALS, N_REPEAT))
        Es_mlp = f.create_dataset("Es_mlp", (N_TRIALS, N_REPEAT))

        with env_guard.SingleThreadEnvGuard() as guard:
            with multiprocessing.get_context("spawn").Pool(
                    multiprocessing.cpu_count() // 2) as pool:
                for (i, j,
                     (E1, E2,
                      E3)) in tqdm.tqdm(pool.imap_unordered(run_single, args),
                                        total=len(args)):
                    Es_add[i, j] = E1
                    Es_mul[i, j] = E2
                    Es_mlp[i, j] = E3


if __name__ == "__main__":
    main()

