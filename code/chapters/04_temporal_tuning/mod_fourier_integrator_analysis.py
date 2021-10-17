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
qs = np.arange(1, 51, 2)

N_JITTER = 1000

def mk_mod_fourier_lti(q, fac=0.9, Ninternal=1000):
    def mk_fourier_oscillator(q, mul=1.0):
        B = (np.arange(0, q) + 1) % 2
        A = np.zeros((q, q))
        for k in range(1, q):
            ki = (k + 1) // 2
            fk = 2.0 * np.pi * mul * ki
            A[2 * ki - 1, 2 * ki - 1] = 0
            A[2 * ki - 1, 2 * ki + 0] = fk
            A[2 * ki + 0, 2 * ki - 1] = -fk
            A[2 * ki + 0, 2 * ki + 0] = 0
        return A, B

    assert q % 2 == 1

    A, B = mk_fourier_oscillator(q, mul=0.9)
    Ad, Bd = np.zeros((q, q)), np.zeros((q, ))

    Ad[1:, 1:], Bd[1:] = bases.discretize_lti(1.0 / Ninternal, A[1:, 1:],
                                              B[1:])
    Bd[0] = 1e-3
    Ad[0, 0] = 1.0

    H = bases.mk_lti_basis(Ad,
                           Bd,
                           Ninternal,
                           from_discrete_lti=True,
                           normalize=False)
    enc = H[:, 0]
    dec = np.linalg.pinv(H, rcond=1e-2)[0]

    Ad = Ad - np.outer(enc, dec) @ Ad
    Bd = Bd - np.outer(enc, dec) @ Bd

    A = np.real(scipy.linalg.logm(Ad)) * Ninternal
    return A, B


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
    Ao, Bo = mk_mod_fourier_lti(q)
    A, B = Ao / N, Bo / N  # Apply the timestep
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
        max_Ls[i] = np.max(np.abs((1.0 + L)))  # Update equation growth factor
    lambda_ = np.mean(max_Ls)

    # Compute the basis transformation matrix
    At, Bt = A + np.eye(q), B / N
    H = np.zeros((q, N))
    Aexp = np.eye(q)
    for i in range(N):
        H[:, N - i - 1] = Aexp @ Bt
        Aexp = At @ Aexp
    H = H / np.linalg.norm(H, axis=1)[:, None]

    # Compute the error between the basis generated via ZOH and the
    # selected method
    H_ref = bases.mk_lti_basis(A, B, N)
    rmse = np.sqrt(np.mean(np.square(H - H_ref)))
    rms = np.sqrt(np.mean(np.square(H_ref)))
    nrmse = rmse / rms

    # Compute the singular value sum
    try:
        _, S, _ = np.linalg.svd(H)
        sigma = np.sum(S / np.max(S))
    except np.linalg.LinAlgError:
        sigma = np.nan

    return idcs, lambda_, nrmse, sigma


def main():
    fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
                      "mod_fourier_integrator_analysis.h5")

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

