#!/usr/bin/env python3

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))

import itertools
import tqdm
import numpy as np
import multiprocessing
import random
import env_guard
import h5py

from temporal_encoder_common import *

TAUS_INT = (100e-3, 120e-3, 10e-3, 30e-3)
TAUS_PASS = (200e-3, 220e-3, 100e-3, 120e-3)
TAUS_REF = (100e-3, 100e-3, 100e-3, 100e-3)

TARGETS = ["integrator", "pass-through"]
N_TARGETS = len(TARGETS)

NETWORKS = ["ref", "ff", "rec"]
N_NETWORKS = len(NETWORKS)

FILTERS = [1, 2, 3, 5, 10]
N_FILTERS = len(FILTERS)

N_REPEAT = 100

N_FREQS = 31
FREQS = np.logspace(-1, 2, N_FREQS)


def solve(flts_A, flts_B, flt_tar):
    # Solve for weights
    W = solve_for_linear_dynamics(
        [Filters.lowpass(tau, order)
         for tau, order in flts_B] +  # input filters
        [Filters.lowpass(tau, order)
         for tau, order in flts_A],  # feedback filters
        [Filters.dirac() for _ in range(len(flts_B))] +  # input dynamics
        [flt_tar for _ in range(len(flts_A))],  # feedback dynamics
        [flt_tar],  # desired feedback dynamics
        T=10.0,
        silent=True,
        rcond=1e-6)

    # Return the A and B matrices
    return W[len(flts_B):].reshape(-1, 1, 1), W[:len(flts_B)].reshape(-1, 1)


def simulate(high, flts_A, flts_B, flt_tar, A, B):
    dt = 1e-3
    ts, xs, ys = simulate_dynamics(flts_A,
                                   flts_B,
                                   A,
                                   B,
                                   "noise",
                                   high=high,
                                   T=10.0,
                                   dt=dt,
                                   silent=True)
    ys_tar = scipy.signal.fftconvolve(xs[:, 0], flt_tar(ts, dt),
                                      'full')[:len(ts)] * dt

    rmse = np.sqrt(np.mean(np.square(ys[:, 0] - ys_tar)))
    rms = np.sqrt(np.mean(np.square(ys_tar)))

    #    import matplotlib.pyplot as plt

    #    fig, ax = plt.subplots()
    #    ax.plot(ts, xs)
    #    ax.plot(ts, ys)
    #    ax.plot(ts, ys_tar)

    #    plt.show()

    #    print(rmse, rms, rmse / rms)

    return rmse / rms


def execute_single(idcs):
    i_target, i_net, i_filter, i_seed = idcs

    target = TARGETS[i_target]
    net = NETWORKS[i_net]
    n_flts = FILTERS[i_filter]

    np.random.seed(5178 * i_seed + 310)

    # Select the dynamics to solve for
    if target == "integrator":
        flt_tar = Filters.step()
        tau0A, tau1A, tau0B, tau1B = TAUS_INT
    elif target == "pass-through":
        flt_tar = Filters.lowpass(1e-3)
        tau0A, tau1A, tau0B, tau1B = TAUS_PASS
    if net == "ref":
        tau0A, tau1A, tau0B, tau1B = TAUS_REF

    # Make sure we only have one filter if we are in reference mode; use
    # first-order filters
    assert (n_flts == 1) or (net != "ref")
    order = 0 if net == "ref" else 1

    # Assemble the recurrent and input filters
    if net == "ff":
        flts_A = []
    else:
        flts_A = [(tau, order) for tau in np.linspace(tau0A, tau1A, n_flts)]
    flts_B = [(tau, order) for tau in np.linspace(tau0B, tau1B, n_flts)]

    # Solve for the connection matrices A, B
    if net == "ref":
        tau = flts_B[0][0]
        if target == "integrator":
            A, B = 1.0, tau
        else:
            A, B = -tau / 1e-3 + 1.0, tau / 1e-3
    else:
        A, B = solve(flts_A, flts_B, flt_tar)

    # Simulate the network with different bandwidths

    Es = np.zeros(N_FREQS)
    for i, f in enumerate(FREQS):
        np.random.seed(4917 * i_seed + i)
        Es[i] = simulate(f, flts_A, flts_B, flt_tar, A, B)

    return idcs, Es


def main():
    print("Executing experiments...")

    def valid_parameter_set(idcs):
        i_target, i_net, i_filter, i_seed = idcs
        net = NETWORKS[i_net]
        n_flts = FILTERS[i_filter]
        return (n_flts == 1) or (net != "ref")

    idcs = list(
        filter(
            valid_parameter_set,
            itertools.product(range(N_TARGETS), range(N_NETWORKS),
                              range(N_FILTERS), range(N_REPEAT))))

    fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
                      "heterogeneous_recurrence_exploration.h5")
    with h5py.File(fn, 'w') as f:
        f.attrs["filters"] = FILTERS
        f.attrs["networks"] = NETWORKS
        f.attrs["targets"] = TARGETS
        f.attrs["freqs"] = FREQS

        errs = f.create_dataset("errs",
                                shape=(N_TARGETS, N_NETWORKS, N_FILTERS,
                                       N_REPEAT, N_FREQS))

        random.shuffle(idcs)
        with env_guard.SingleThreadEnvGuard():
            with multiprocessing.get_context('spawn').Pool() as pool:
                for (i_target, i_net, i_filter, i_seed), Es in tqdm.tqdm(
                        pool.imap_unordered(execute_single,
                                            idcs), total=len(idcs)):
                    errs[i_target, i_net, i_filter, i_seed] = Es


if __name__ == "__main__":
    main()

