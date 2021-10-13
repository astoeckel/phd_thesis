#!/usr/bin/env python3

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))

import numpy as np
import random
import multiprocessing
import tqdm
import h5py
import env_guard
import scipy.linalg

import dlop_ldn_function_bases as bases

modes = ["zoh", "euler", "midpoint", "runge_kutta"]
Ns = np.arange(1, 1000, dtype=np.int)
qs = np.arange(1, 51)

N_JITTER = 1000


def mexpfin(A, o):
    if o == 0:
        return np.eye(A.shape[0])
    res = np.zeros_like(A)
    f = 1.0
    for i in range(1, o + 1):
        res += np.linalg.matrix_power(A, i) * f
        f /= (i + 1)
    return res


def run_single(params):
    idcs, mode, q, N = params

    # (Approximately) determine asymptotic stability
    A, B = bases.mk_ldn_lti(q)
    A, B = A / N, B / N  # Apply the timestep
    if mode == "zoh":
        Ad = scipy.linalg.expm(A)
        Bd = np.linalg.solve(A, (Ad - np.eye(A.shape[0])) @ B)
        A, B = Ad - np.eye(q), Bd
    elif mode == "euler":
        pass
    elif mode == "midpoint":
        A = mexpfin(A, 2)
    elif mode == "runge_kutta":
        A = mexpfin(A, 4)

    # There are some weird numerical artifacts when computing the eigenvalues
    # that are not visible when numerically estimating stability (see below).
    # Add some noise and repeat N_JITTER times
    np.random.seed(47381)
    max_Ls = np.zeros(N_JITTER)
    for i in range(N_JITTER):
        L = np.linalg.eigvals(A + (0.01 / N) * np.random.randn(q, q))
        max_Ls[i] = np.max(np.abs((1.0 + L))) # Update equation growth factor
    lambda_ = np.mean(max_Ls)

    #    x = np.zeros(q)
    #    x[...] = B  # Impulse
    #    dt = 1.0 / N
    #    A = A * dt  # Account for dt
    #    integral = 0.0
    #    for i in range(1000000):  # Iterate for 1 mio. samples
    #        x += A @ x
    #        if np.any(np.isnan(x)):
    #            break
    #        norm = np.linalg.norm(x)
    #        if norm < 1e-15:
    #            break
    #        integral += norm * dt

    #    if np.any(np.isnan(x)):
    #        mx, mdx = np.inf, np.inf
    #        integral = np.inf
    #    else:
    #        mx, mdx = np.linalg.norm(x), np.linalg.norm(A @ x)

    # Compute the basis transformation matrix
    At, Bt = A + np.eye(q), B / N
    H = np.zeros((q, N))
    Aexp = np.eye(q)
    for i in range(N):
        H[:, N - i - 1] = Aexp @ Bt
        Aexp = At @ Aexp
    H = H / np.linalg.norm(H, axis=1)[:, None]

    # Compute the error between the LDN basis generated via ZOH and the
    # selected method
    H_ref = bases.mk_ldn_basis(q, N)
    rmse = np.sqrt(np.mean(np.square(H - H_ref)))
    rms = np.sqrt(np.mean(np.square(H_ref)))
    nrmse = rmse / rms

    # Compute the singular value sum
    _, S, _ = np.linalg.svd(H)
    sigma = np.sum(S / np.max(S))

    return idcs, lambda_, nrmse, sigma


def main():
    fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
                      "ldn_integrator_analysis.h5")

    params = [((i, j, k), modes[i], qs[j], Ns[k]) for i in range(len(modes))
              for j in range(len(qs)) for k in range(len(Ns))]
    random.shuffle(params)

    with h5py.File(fn, "w") as f:
        f.create_dataset("qs", data=qs)
        f.create_dataset("Ns", data=Ns)

        lambdas = f.create_dataset("lambdas",
                                   shape=(len(modes), len(qs), len(Ns)))
        errs = f.create_dataset("errs", shape=(len(modes), len(qs), len(Ns)))
        sigmas = f.create_dataset("sigmas",
                                  shape=(len(modes), len(qs), len(Ns)))

        with env_guard.SingleThreadEnvGuard():
            with multiprocessing.get_context('spawn').Pool() as pool:
                for (i, j, k), lambda_, nrmse, sigma in tqdm.tqdm(
                        pool.imap_unordered(run_single,
                                            params), total=len(params)):
                    lambdas[i, j, k] = lambda_
                    errs[i, j, k] = nrmse
                    sigmas[i, j, k] = sigma


if __name__ == "__main__":
    main()

