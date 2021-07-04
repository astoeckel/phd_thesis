#!/usr/bin/env python3

import itertools
import multiprocessing
import sys, os

import numpy as np

sys.path.append(os.path.join('..', 'lib'))

import random
import tqdm
import h5py

import env_guard
import lif_utils
from nonneg_common import mk_ensemble
import bioneuronqp


def forkrng(rng=np.random):
    return np.random.RandomState(rng.randint((1 << 31)))

def nrmse(x, x_ref):
    return np.sqrt(np.mean(np.square(x - x_ref))) / np.sqrt(
        np.mean(np.square(x_ref)))

def solve_for_weights(N_pre,
                      f,
                      w,
                      N_smpls=40,
                      N_res=201,
                      rng=np.random):

    # Create two pre-populations
    gains_pre1, biases_pre1, encoders_pre1 = mk_ensemble(N_pre,
                                                         rng=forkrng(rng))
    gains_pre2, biases_pre2, encoders_pre2 = mk_ensemble(N_pre,
                                                         rng=forkrng(rng))

    # Create the post-neuron. Set the x-intercept to zero.
    gains_post, biases_post, encoders_post = mk_ensemble(
        1,
        rng=forkrng(rng),
        x_intercepts_cback=lambda _, N: np.zeros(N),
        encoders_cback=lambda _, N, d: np.ones((N, d)),
        max_rates=(100, 100))

    # Generate pre-population samples. We implicitly map the range [-1, 1] onto
    # [0, 1]
    xs, ys = np.linspace(-1, 1, N_smpls), np.linspace(-1, 1, N_smpls)
    xss, yss = np.meshgrid(xs, ys)
    smpls = np.array((xss.flatten(), yss.flatten())).T

    # Evaluate the pre-populations at those points
    Js_pre1 = gains_pre1[None, :] * (
        smpls[:, 0].reshape(-1, 1) @ encoders_pre1.T) + biases_pre1[None, :]
    Js_pre2 = gains_pre2[None, :] * (
        smpls[:, 1].reshape(-1, 1) @ encoders_pre2.T) + biases_pre2[None, :]

    # Evaluate the post-neuron over the range [0, 1]
    xs, ys = np.linspace(0, 1, N_smpls), np.linspace(0, 1, N_smpls)
    xss, yss = np.meshgrid(xs, ys)
    zss = f(xss, yss)
    Js_post = gains_post[None, :] * (
        zss.reshape(-1, 1) @ encoders_post.T) + biases_post[None, :]

    As_pre1 = lif_utils.lif_rate(Js_pre1)
    As_pre2 = lif_utils.lif_rate(Js_pre2)

    As_pre = np.concatenate((As_pre1, As_pre2), axis=1)

    # Solve for weights
    WE, WI = bioneuronqp.solve(As_pre,
                               Js_post,
                               np.array(w),
                               None,
                               iTh=None,
                               renormalise=True,
                               reg=10.0,
                               n_threads=1,
                               progress_callback=None,
                               warning_callback=None)

    def H(gE, gI):
        b0, b1, b2, a0, a1, a2 = w
        return (b0 + b1 * gE + b2 * gI) / (a0 + a1 * gE + a2 * gI)

    def dec(J):
        return (J - biases_post[0]) / gains_post[0]

    # Now evaluate the pre-populations at higher resolutions in 1D
    xs, ys = np.linspace(-1, 1, N_res), np.linspace(-1, 1, N_res)

    Js_pre1 = gains_pre1[None, :] * (
        xs.reshape(-1, 1) @ encoders_pre1.T) + biases_pre1[None, :]
    Js_pre2 = gains_pre2[None, :] * (
        ys.reshape(-1, 1) @ encoders_pre2.T) + biases_pre2[None, :]

    As_pre1 = lif_utils.lif_rate(Js_pre1)
    As_pre2 = lif_utils.lif_rate(Js_pre2)

    gE1 = As_pre1 @ WE[:N_pre]
    gE2 = As_pre2 @ WE[N_pre:]
    gE = 0.5 * (gE1 + gE2)  # Functions should be symmetric

    gI1 = As_pre1 @ WI[:N_pre]
    gI2 = As_pre2 @ WI[N_pre:]
    gI = 0.5 * (gI1 + gI2)  # Functions should be symmetric

    # Map onto [0, 1]
    xs, ys = np.linspace(0, 1, N_res), np.linspace(0, 1, N_res)
    zss = f(xs[:, None], ys[None, :])
    zss_dec = dec(H(gE + gE.T, gI + gI.T))


    return WE, WI, nrmse(zss_dec, zss)


FS = [
    lambda x, y: 0.5 * (x + y),
    lambda x, y: x * y,
]
N_FS = len(FS)

WS = [
    [0, 1, -1, 1, 0, 0],
    [-19.5e-6, 1000.0, -425.5, 9.0e6, 296.4, 132.2],
]
N_WS = len(WS)

N_REPEAT  = 1000
N_SMPLS = 20
NS = []
i = 0
while len(NS) < N_SMPLS:
    NS = np.unique(np.geomspace(1, 300, N_SMPLS + i, dtype=int))
    i += 1

F_DATA = os.path.join('data', 'two_comp_weights_examples_statistics.h5')


def run_single_experiment(args):
    i, j, k, l, N = args

    rng = np.random.RandomState(49291 * l + 77781)

    WE, WI, E = solve_for_weights(N_pre=N, f=FS[j], w=WS[k], rng=rng)

    return i, j, k, l, WE, WI, E


def main():
    params = [
        (i, j, k, l, int(N)) for (i, N) in enumerate(NS)
        for j in range(N_FS) for k in range(N_WS) for l in range(N_REPEAT)
    ]
    random.shuffle(params)

    # Open the experiment data and write the evaluation data
    with h5py.File(F_DATA, 'w') as f:
        D_WE = f.create_dataset("WE", (N_SMPLS, N_FS, N_WS, N_REPEAT, 2 * max(NS)))
        D_WE[...] = np.nan

        D_WI = f.create_dataset("WI", (N_SMPLS, N_FS, N_WS, N_REPEAT, 2 * max(NS)))
        D_WI[...] = np.nan

        D_E = f.create_dataset("E", (N_SMPLS, N_FS, N_WS, N_REPEAT))
        D_E[...] = np.nan

        D_N = f.create_dataset("N", data=NS)

        with env_guard.SingleThreadEnvGuard():
            with multiprocessing.get_context('spawn').Pool() as pool:
                for i, j, k, l, WE, WI, E in tqdm.tqdm(
                        pool.imap_unordered(run_single_experiment,
                                            params), total=len(params)):

                    D_E[i, j, k, l] = E
                    D_WE[i, j, k, l, :WE.size] = WE.flatten()
                    D_WI[i, j, k, l, :WI.size] = WI.flatten()

if __name__ == "__main__":
    main()
