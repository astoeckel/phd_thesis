#!/usr/bin/env python3

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))

import numpy as np
import random
import multiprocessing
import tqdm
import h5py
import env_guard

import dlop_ldn_function_bases as bases

Ns = np.arange(1, 1000, dtype=np.int)
qs = np.arange(1, 51)


def run_single(params):
    idcs, q, N = params

    # Compute the error between the LDN basis generated via ZOH and Euler
    H_euler = bases.mk_ldn_basis_euler(q, N)
    H_zoh = bases.mk_ldn_basis(q, N)
    rmse = np.sqrt(np.mean(np.square(H_euler - H_zoh)))
    rms = np.sqrt(np.mean(np.square(H_zoh)))
    nrmse = rmse / rms

    # (Approximately) determine BIBO stability
    A, B = bases.mk_ldn_lti(q, N)
    x = np.zeros(q)
    x[...] = B  # Impulse
    dt = 1.0 / N
    A = A * dt  # Account for dt
    integral = 0.0
    for i in range(1000000):  # Iterate for 1 mio. samples
        x += A @ x
        if np.any(np.isnan(x)):
            break
        norm = np.linalg.norm(x)
        if norm < 1e-15:
            break
        integral += norm * dt

    if np.any(np.isnan(x)):
        mx, mdx = np.inf, np.inf
        integral = np.inf
    else:
        mx, mdx = np.linalg.norm(x), np.linalg.norm(A @ x)

    return idcs, nrmse, mx, mdx, integral


def main():
    fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
                      "ldn_bibo_stability.h5")

    params = [((i, j), qs[i], Ns[j]) for i in range(len(qs))
              for j in range(len(Ns))]
    random.shuffle(params)

    with h5py.File(fn, "w") as f:
        f.create_dataset("qs", data=qs)
        f.create_dataset("Ns", data=Ns)

        Es = f.create_dataset("Es", shape=(len(qs), len(Ns)))
        mxs = f.create_dataset("mxs", shape=(len(qs), len(Ns)))
        mdxs = f.create_dataset("mdxs", shape=(len(qs), len(Ns)))
        Is = f.create_dataset("Is", shape=(len(qs), len(Ns)))

        with env_guard.SingleThreadEnvGuard():
            with multiprocessing.get_context('spawn').Pool() as pool:
                for (i, j), E, mx, mdx, I in tqdm.tqdm(pool.imap_unordered(
                        run_single, params),
                                                    total=len(params)):
                    Es[i, j] = E
                    mxs[i, j] = mx
                    mdxs[i, j] = mdx
                    Is[i, j] = I


if __name__ == "__main__":
    main()

