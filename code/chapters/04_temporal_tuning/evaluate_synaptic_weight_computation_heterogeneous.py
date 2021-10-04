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
import json

import lif_utils
import nonneg_common
import basis_delay_analysis_common
import temporal_encoder_common
from temporal_encoder_common import Filters
import dlop_ldn_function_bases as bases

import nengo

nengo.rc.set('decoder_cache', 'enabled', 'False')

#
# Parameters
#

DT = 1e-3
T_SIM = 10.0
T_TRAIN = 10.0  # When computing the weights; needs to be longer because of potentially long filters
N_TRAIN_SMPLS = 250  # Training samples

N_NEURONS = 100

THETA = 1.0
TAU_MU = 250e-3
TAU_MIN = 5e-3
TAU_DECODE = 25e-3
XS_SIGMA = 3.0

MODES = [
    "non_lindep_cosine",
    "legendre_erasure",
]
N_MODES = len(MODES)

QS = np.array([3, 5, 7])
N_QS = len(QS)

N_TAU_SIGMAS = 21
TAU_SIGMAS = np.linspace(0, TAU_MU, N_TAU_SIGMAS)
N_TAU_SIGMAS = len(TAU_SIGMAS)

N_DELAYS_TEST = 20

N_REPEAT = 100

N_REPEAT_TEST = 10

#
# Network simulation code
#


def LP(*args):
    return nengo.LinearFilter(*Filters.lowpass_laplace(*args), analog=True)


def simulate_network(n_neurons, dimensions, gain, bias, encoders, flts,
                     flts_rec_map, W_in, W_rec, xs, dt):

    # There seems to be some small random process within negno that (very
    # slightly) influences the results...
    np.random.seed(587929)

    # Instantiate the network
    with nengo.Network() as model:
        nd_in = nengo.Node(lambda t: xs[int(t / dt) % len(xs)])
        ens_x = nengo.Ensemble(
            n_neurons=n_neurons,
            dimensions=dimensions,
            bias=bias,
            gain=gain,
            encoders=encoders,
        )

        for i, tau in enumerate(flts):
            nengo.Connection(nd_in,
                             ens_x.neurons,
                             transform=(W_in[:, i:(i + 1)]),
                             synapse=LP(tau))
            mask_rec = flts_rec_map == i

            if np.any(mask_rec):
                nengo.Connection(ens_x.neurons,
                                 ens_x.neurons,
                                 transform=(W_rec * mask_rec),
                                 synapse=LP(tau))

        p_x = nengo.Probe(ens_x.neurons, synapse=None)

    # Run the simulation
    with nengo.Simulator(model, progress_bar=False) as sim:
        sim.run(len(xs) * dt)

    return sim.trange(), np.copy(sim.data[p_x])


#
# Experiment runner
#


def mk_cosine_bartlett_basis_with_spread(q,
                                         N,
                                         T=1.0,
                                         dt=1e-3,
                                         phi_max=2.0,
                                         decay_min=1.5,
                                         decay_max=1.5,
                                         rng=np.random):
    phis = np.linspace(0, phi_max, q) * 2.0 * np.pi
    phases = rng.uniform(0, np.pi, q)
    t1 = rng.uniform(decay_min, decay_max, q) * T
    ts = np.arange(N) * dt
    return (
        np.cos(ts[None, :] * phis[:, None] / t1[:, None] + phases[:, None]) *
        (1.0 - ts[None, :] / t1[:, None]) * (ts[None, :] <= t1[:, None]))


def execute_single(idcs):
    i_modes, i_qs, i_tau_sigmas, i_repeat = idcs

    # Set the random seed, just in case something uses np.random
    np.random.seed(49101 * i_repeat + 431)

    # Fetch all parameters
    mode = MODES[i_modes]
    q = QS[i_qs]
    n_neurons = N_NEURONS
    tau_sigma = TAU_SIGMAS[i_tau_sigmas]

    # Determine the number of temporal dimensions to use
    n_temporal_dimensions = n_neurons if (mode == "non_lindep_cosine") else q

    # Construct the temporal encoders Ms
    ts_train = np.arange(0, T_TRAIN, DT)
    N_train = len(ts_train)
    N_theta = int(THETA / DT)
    if mode == "non_lindep_cosine":
        Ms = mk_cosine_bartlett_basis_with_spread(n_temporal_dimensions,
                                                  N_train,
                                                  phi_max=0.5 * (q - 1)).T
    else:
        basis, window = mode.rsplit("_", 1)
        _, Ms = basis_delay_analysis_common.mk_impulse_response(
            basis=basis,
            window=window,
            q=n_temporal_dimensions,
            T=T_TRAIN,
            dt=DT,
            use_euler=True,
            rescale_ldn=False)

    # Generate the ensemble
    np.random.seed(47881 * i_repeat + 133)
    G = lif_utils.lif_rate
    gains, biases, Es = nonneg_common.mk_ensemble(n_neurons,
                                                  d=n_temporal_dimensions,
                                                  max_rates=(100, 200))

    # Filters to use
    rng = np.random.RandomState(7892 * i_repeat + 535)

    def mk_flts(shape, digits=1):
        taus = np.clip(rng.normal(TAU_MU, tau_sigma, shape), TAU_MIN, None)
        taus = np.exp(np.round(np.log(taus), digits))
        return taus

    flts_rec = mk_flts((n_neurons, n_neurons))
    flts = np.unique(flts_rec)
    flts_map = {value: idx for idx, value in enumerate(flts)}
    flts_rec_map = np.array([flts_map[flt] for flt in flts_rec.flat
                             ]).reshape(n_neurons, n_neurons)
    i_mu = np.argmin(np.abs(flts - TAU_MU))

    # Solve for weights

    def solve(flts_rec_map):
        rng = np.random.RandomState(7193 * i_repeat + 481)
        return temporal_encoder_common.solve_for_recurrent_population_weights_heterogeneous_filters(
            G,
            gains,
            biases,
            None,
            None,
            Es, [Filters.lowpass(tau) for tau in flts],
            flts_rec_map,
            Ms=Ms,
            N_smpls=N_TRAIN_SMPLS,
            T=T_TRAIN,
            biased=False,
            rng=rng,
            silent=True)

    W_in, W_rec, Es_solver = solve(flts_rec_map)
    W_in_ref, W_rec_ref, Es_solver_ref = solve(
        np.ones((n_neurons, n_neurons), dtype=int) * i_mu)

    # Simulate the network and compute the tuning error
    def compute_errors(W_in, W_rec):
        Es_tuning = np.zeros(N_REPEAT_TEST)

        for i_repeat_test in range(N_REPEAT_TEST):
            # Use the same test signals for each network
            rng = np.random.RandomState(340043 * i_repeat + 2814 * i_repeat_test + 213)

            # Generate a a test signal for the tuning error computation
            N_sim = int(T_SIM / DT)
            xs_test = temporal_encoder_common.mk_sig(N_sim,
                                                     DT,
                                                     sigma=XS_SIGMA,
                                                     rng=rng)
            xs_test_flt = nengo.Lowpass(TAU_DECODE).filtfilt(xs_test, dt=DT)
            xs_test_rms = np.sqrt(np.mean(np.square(xs_test_flt)))

            ts, As_test = simulate_network(n_neurons=n_neurons,
                                           dimensions=n_temporal_dimensions,
                                           gain=gains,
                                           bias=biases,
                                           encoders=Es,
                                           flts=flts,
                                           flts_rec_map=flts_rec_map,
                                           W_in=W_in,
                                           W_rec=W_rec,
                                           xs=xs_test,
                                           dt=DT)

            # Compute the expected activities for xs_test
            As_test_ref = np.zeros((N_sim, n_neurons))
            xs_test_conv = np.array([
                np.convolve(xs_test, Ms[:, j], 'full')[:N_sim] * DT
                for j in range(n_temporal_dimensions)
            ]).T

            for i_neuron in range(n_neurons):
                As_test_ref[:, i_neuron] = G(gains[i_neuron] *
                                             (xs_test_conv @ Es[i_neuron]) +
                                             biases[i_neuron])

                # Scale the errors by the maximum rate
                max_rate = G(gains[i_neuron] + biases[i_neuron])
                As_test[:, i_neuron] /= max_rate
                As_test_ref[:, i_neuron] /= max_rate

            As_test_flt = nengo.Lowpass(TAU_DECODE).filtfilt(As_test, dt=DT)
            As_test_ref_flt = nengo.Lowpass(TAU_DECODE).filtfilt(As_test_ref,
                                                                 dt=DT)

            # Compute the activity error
            rmse = np.sqrt(np.mean(np.square(As_test_flt - As_test_ref_flt)))
            rms = np.sqrt(np.mean(np.square(As_test_ref_flt)))
            Es_tuning[i_repeat_test] = rmse / rms

        return Es_tuning

    Es_tuning = compute_errors(W_in, W_rec)
    Es_tuning_ref = compute_errors(W_in_ref, W_rec_ref)

    return idcs, Es_solver, Es_solver_ref, Es_tuning, Es_tuning_ref


#
# Main program
#


def main():
    if len(sys.argv) > 1:
        n_partitions = int(sys.argv[1])
        partition_idx = int(sys.argv[2])
    else:
        n_partitions = 1
        partition_idx = 0

    assert n_partitions > 0
    assert partition_idx < n_partitions
    assert partition_idx >= 0

    if n_partitions == 1:
        fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
                          "evaluate_synaptic_weight_computation_heterogeneous.h5")
    else:
        fn = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'data',
            "evaluate_synaptic_weight_computation_heterogeneous_{}.h5".format(partition_idx))

    with h5py.File(fn, 'w') as f:
        f.attrs["modes"] = json.dumps(MODES)
        f.attrs["qs"] = json.dumps([int(q) for q in QS])
        f.attrs["tau_sigmas"] = json.dumps([int(x) for x in TAU_SIGMAS])

        errs_solver = f.create_dataset("errs_solver",
                                       shape=(N_MODES, N_QS, N_TAU_SIGMAS,
                                              N_REPEAT, N_NEURONS),
                                       compression="gzip")
        errs_solver[...] = np.nan

        errs_solver_ref = f.create_dataset("errs_solver_ref",
                                           shape=(N_MODES, N_QS, N_TAU_SIGMAS,
                                                  N_REPEAT, N_NEURONS),
                                           compression="gzip")
        errs_solver_ref[...] = np.nan

        errs_tuning = f.create_dataset("errs_tuning",
                                       shape=(N_MODES, N_QS, N_TAU_SIGMAS,
                                              N_REPEAT, N_REPEAT_TEST),
                                       compression="gzip")
        errs_tuning[...] = np.nan

        errs_tuning_ref = f.create_dataset("errs_tuning_ref",
                                           shape=(N_MODES, N_QS, N_TAU_SIGMAS,
                                                  N_REPEAT, N_REPEAT_TEST),
                                           compression="gzip")
        errs_tuning_ref[...] = np.nan

        idcs = list(
            itertools.product(range(N_MODES), range(N_QS), range(N_TAU_SIGMAS),
                              range(N_REPEAT)))

        # Always shuffle the indices in the right way to not ruin the indexing
        random.seed(57482)
        random.shuffle(idcs)

        partitions = np.linspace(0, len(idcs), n_partitions + 1, dtype=int)
        i0 = partitions[partition_idx]
        i1 = partitions[partition_idx + 1]
        print(
            f"Partition {partition_idx} out of {n_partitions} (i0={i0}, i1={i1}); total={len(idcs)}"
        )

        with env_guard.SingleThreadEnvGuard():
            with multiprocessing.get_context('spawn').Pool(
                    maxtasksperchild=1) as pool:
                for idcs, Es_solver, Es_solver_ref, Es_tuning, Es_tuning_ref in tqdm.tqdm(
                        pool.imap_unordered(execute_single, idcs[i0:i1]),
                        total=i1 - i0):
                    errs_solver[idcs] = Es_solver
                    errs_solver_ref[idcs] = Es_solver_ref

                    errs_tuning[idcs] = Es_tuning
                    errs_tuning_ref[idcs] = Es_tuning_ref

                    f.flush()


if __name__ == "__main__":
    main()

