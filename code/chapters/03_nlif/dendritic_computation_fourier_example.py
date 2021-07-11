#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "lib"))

import numpy as np
import random
import h5py
import multiprocessing
import tqdm
import scipy.optimize

import function_bases as bases
import env_guard
import gen_2d_fun

# Number of basis functions
d = 7

# Number of samples
N = 63

REG = 1e-3

# Generate the basis
A = bases.mk_dlop_basis(d, N) * np.sqrt(0.5 * N)
A2D = np.einsum('ij,kl->ikjl', A, A)
A1D = np.concatenate((
    A2D[0, :].reshape(d, -1),
    A2D[:, 0].reshape(d, -1),
)).T
A2D_flat = A2D.reshape(d, d, N * N).T.reshape(N * N, d * d)

def lstsq(A, Y):
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n, m = A.shape
    d = Y.shape[1]
    ATA = A.T @ A + n * REG * np.eye(m)
    D = np.zeros((m, d))
    for i in range(d):
        D[:, i] = np.linalg.solve(ATA, A.T @ Y[:, i])
    return D


def solve_additive(basis, tar, rng):
    return basis @ lstsq(basis, tar)


def solve_multiplicative(basis, tar, rng):
    def f(w):
        w1, w2 = w.reshape(2, -1)
        return (basis @ w1) * (basis @ w2)

    def E(w):
        return np.mean(np.square(f(w) - tar)) + np.sqrt(REG) * np.mean(np.square(w))


    Ds, Es = [[None] * 10 for _ in range(2)]
    for i in range(len(Ds)):
        w = rng.randn(4 * d)
        Ds[i] = scipy.optimize.minimize(E, w, tol=1e-3, method='BFGS').x
        Es[i] = E(Ds[i])
    return f(Ds[np.argmin(Es)])


def solve_mlp(Ys, N_neurons=100, rng=np.random):

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


def run_single(args):
    i, j, sigma = args

    rng = np.random.RandomState(34891 * j + 480)
    X = gen_2d_fun.gen_2d_fun(gen_2d_fun.mk_2d_flt(sigma, N), N, rng)

    Y1 = solve_additive(A1D, X.reshape(-1), rng=rng).reshape(N, N)

    rng = np.random.RandomState(34891 * j + 481)
    Y2 = solve_multiplicative(A1D, X.reshape(-1), rng=rng).reshape(N, N)

    rng = np.random.RandomState(34891 * j + 482)
    Y3 = solve_additive(A2D_flat, X.reshape(-1), rng=rng).reshape(N, N)

    E1 = np.sqrt(np.mean(np.square(X - Y1)))
    E2 = np.sqrt(np.mean(np.square(X - Y2)))
    E3 = np.sqrt(np.mean(np.square(X - Y3)))

    return i, j, (E1, E2, E3)


def main():
    N_REPEAT = 1000
    N_TRIALS = 60
    SIGMAS = np.logspace(-1, 1, N_TRIALS)
    #    MUS = np.linspace(0, 3, N_TRIALS)

    #args = [(i, j, mu) for j in range(N_REPEAT) for i, mu in enumerate(MUS)]
    args = [(i, j, sigma) for j in range(N_REPEAT) for i, sigma in enumerate(SIGMAS)]
    random.shuffle(args)

    with h5py.File(
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
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

