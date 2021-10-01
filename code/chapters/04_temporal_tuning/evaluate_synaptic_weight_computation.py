#!/usr/bin/env python3

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))

import itertools
import numpy as np
import random
import env_guard
import h5py
import json
import working_multiprocessing_pool

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
T_TRAIN = 3.0  # When computing the weights
N_TRAIN_SMPLS = 100  # Training samples
T_SIM = 10.0

THETA = 1.0
TAU = 100e-3
TAU_DECODE = 100e-3
XS_SIGMA = 2.0

N_DIMS_CSTR = 3

SOLVER_MODES = ["nef", "biased_xs", "unbiased_xs"]
N_SOLVER_MODES = len(SOLVER_MODES)

MODES = [
    #    "non_lindep_cosine",
    "mod_fourier_bartlett",
    "mod_fourier_erasure",
    "legendre_erasure",
]
N_MODES = len(MODES)

QS = np.array([3, 5, 7])
N_QS = len(QS)

N_NEURONS = 7
NEURONS = np.unique(np.geomspace(10, 1000, N_NEURONS, dtype=int))
N_NEURONS = len(NEURONS)

N_DELAYS_TEST = 20

XS_SIGMA_TEST = [2.0]
N_XS_SIGMA_TEST = len(XS_SIGMA_TEST)

N_REPEAT = 100

N_REPEAT_TEST = 10

#
# Network simulation code
#


def LP(*args):
    return nengo.LinearFilter(*Filters.lowpass_laplace(*args), analog=True)


def simulate_network(n_neurons, dimensions, gain, bias, encoders, flts_in,
                     flts_rec, W_in, W_rec, xs, dt):
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

        for i, flt_in in enumerate(flts_in):
            nengo.Connection(nd_in,
                             ens_x.neurons,
                             transform=W_in[:, i].reshape(-1, 1),
                             synapse=LP(*flt_in))

        for i, flt_rec in enumerate(flts_rec):
            nengo.Connection(ens_x.neurons,
                             ens_x.neurons,
                             transform=W_rec[:, :, i],
                             synapse=LP(*flt_rec))

        p_x = nengo.Probe(ens_x.neurons, synapse=None)

    # Run the simulation
    with nengo.Simulator(model, progress_bar=False) as sim:
        sim.run(len(xs) * dt)

    return sim.trange(), np.copy(sim.data[p_x])


def simulate_network_ref(n_neurons,
                         dimensions,
                         gain,
                         bias,
                         encoders,
                         A,
                         B,
                         xs,
                         dt,
                         tau=TAU):
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

        nengo.Connection(nd_in,
                         ens_x,
                         transform=tau * B.reshape(-1, 1),
                         synapse=tau)
        nengo.Connection(ens_x,
                         ens_x,
                         transform=tau * A + np.eye(A.shape[0]),
                         synapse=tau)

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
    i_solver_modes, i_modes, i_qs, i_neurons, i_repeat = idcs

    # Set the random seed, just in case something uses np.random
    np.random.seed(49101 * i_repeat + 431)

    # Fetch all parameters
    solver_mode = SOLVER_MODES[i_solver_modes]
    mode = MODES[i_modes]
    q = QS[i_qs]
    n_neurons = NEURONS[i_neurons]

    # Determine the number of temporal dimensions to use
    n_temporal_dimensions = n_neurons if (mode == "non_lindep_cosine") else q

    # Determine the bias mode
    if solver_mode == "unbiased_xs":
        biased, bias_cstr_count = False, None
    else:
        biased, bias_cstr_count = True, (1 if mode == "non_lindep_cosine" else
                                         N_DIMS_CSTR)

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
        args = dict(basis=basis,
                    window=window,
                    q=n_temporal_dimensions,
                    T=T_TRAIN,
                    dt=DT,
                    use_euler=True)
        _, Ms = basis_delay_analysis_common.mk_impulse_response(**args)
        if solver_mode == "nef":
            args["return_sys"] = True
            A, B = basis_delay_analysis_common.mk_impulse_response(**args)

    # Generate the ensemble
    np.random.seed(47881 * i_repeat + 133)
    G = lif_utils.lif_rate
    gains, biases, Es = nonneg_common.mk_ensemble(n_neurons,
                                                  d=n_temporal_dimensions,
                                                  max_rates=(100, 200))

    # Filters to use
    flts_in = [(TAU, )]
    flts_rec = [(TAU, )]

    # Solve for weights
    rng = np.random.RandomState(7193 * i_repeat + 481)
    if solver_mode != "nef":
        W_in, W_rec = temporal_encoder_common.solve_for_recurrent_population_weights(
            G,
            gains,
            biases,
            None,
            None,
            Es, [Filters.lowpass(*flt_in) for flt_in in flts_in],
            [Filters.lowpass(*flt_rec) for flt_rec in flts_rec],
            xs_sigma=XS_SIGMA,
            Ms=Ms,
            N_smpls=N_TRAIN_SMPLS,
            T=T_TRAIN,
            biased=biased,
            bias_cstr_count=bias_cstr_count,
            rng=rng,
            silent=True)

    # Simulate the network
    def run_simulation(xs):
        if solver_mode == "nef":
            return simulate_network_ref(n_neurons=n_neurons,
                                        dimensions=n_temporal_dimensions,
                                        gain=gains,
                                        bias=biases,
                                        encoders=Es,
                                        A=A,
                                        B=B,
                                        xs=xs,
                                        dt=DT,
                                        tau=TAU)
        else:
            return simulate_network(n_neurons=n_neurons,
                                    dimensions=n_temporal_dimensions,
                                    gain=gains,
                                    bias=biases,
                                    encoders=Es,
                                    flts_in=flts_in,
                                    flts_rec=flts_rec,
                                    W_in=W_in,
                                    W_rec=W_rec,
                                    xs=xs,
                                    dt=DT)

    Es_tuning = np.zeros((N_XS_SIGMA_TEST, N_REPEAT_TEST))
    Es_delay = np.zeros((N_XS_SIGMA_TEST, N_REPEAT_TEST, N_DELAYS_TEST))
    N_shifts = np.linspace(0, THETA / DT, N_DELAYS_TEST + 1, dtype=int)[:-1]

    for i_xs_sigma in range(N_XS_SIGMA_TEST):
        for i_repeat_test in range(N_REPEAT_TEST):
            # Use the same test signals for each network
            rng = np.random.RandomState(2814 * i_repeat_test + 213)

            # Generate a training and a test signal for the delay and tuning error
            # computation
            N_sim = int(T_SIM / DT)
            xs_sigma_test = XS_SIGMA_TEST[i_xs_sigma]
            xs_train = temporal_encoder_common.mk_sig(N_sim,
                                                      DT,
                                                      sigma=xs_sigma_test,
                                                      rng=rng)
            xs_test = temporal_encoder_common.mk_sig(N_sim,
                                                     DT,
                                                     sigma=xs_sigma_test,
                                                     rng=rng)

            xs_train_flt = nengo.Lowpass(TAU_DECODE).filtfilt(xs_train, dt=DT)
            xs_test_flt = nengo.Lowpass(TAU_DECODE).filtfilt(xs_test, dt=DT)
            xs_test_rms = np.sqrt(np.mean(np.square(xs_test_flt)))

            ts, As_train = run_simulation(xs_train)
            _, As_test = run_simulation(xs_test)

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
                As_train[:, i_neuron] /= max_rate

            As_train_flt = nengo.Lowpass(TAU_DECODE).filtfilt(As_train, dt=DT)
            As_test_flt = nengo.Lowpass(TAU_DECODE).filtfilt(As_test, dt=DT)
            As_test_ref_flt = nengo.Lowpass(TAU_DECODE).filtfilt(As_test_ref,
                                                                 dt=DT)

            # Compute the activity error
            rmse = np.sqrt(np.mean(np.square(As_test_flt - As_test_ref_flt)))
            rms = np.sqrt(np.mean(np.square(As_test_ref_flt)))
            Es_tuning[i_xs_sigma, i_repeat_test] = rmse / rms

            # Compute the delay decoding errors
            smpls = rng.uniform(0, N_sim, 250).astype(
                int)  # Subsample temporally when computing the decoders
            for i_delay in range(N_DELAYS_TEST):
                N_shift = N_shifts[i_delay]
                xs_train_tar = np.concatenate(
                    (np.zeros(N_shift), xs_train[:N_sim - N_shift]))
                xs_train_tar_flt = nengo.Lowpass(TAU_DECODE).filtfilt(
                    xs_train_tar, dt=DT)

                D = np.linalg.lstsq(As_train_flt[smpls],
                                    xs_train_tar_flt[smpls],
                                    rcond=1e-2)[0]

                xs_test_tar = np.concatenate(
                    (np.zeros(N_shift), xs_test[:N_sim - N_shift]))
                xs_test_tar_flt = nengo.Lowpass(TAU_DECODE).filtfilt(
                    xs_test_tar, dt=DT)
                xs_test_dec = As_test_flt @ D

                rmse = np.sqrt(
                    np.mean(np.square(xs_test_dec - xs_test_tar_flt)))
                Es_delay[i_xs_sigma, i_repeat_test,
                         i_delay] = rmse / xs_test_rms

    return idcs, Es_tuning, Es_delay


#
# Main program
#


def main():
    def task_init():
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
                              "evaluate_synaptic_weight_computation.h5")
        else:
            fn = os.path.join(
                os.path.dirname(__file__), '..', '..', '..', 'data',
                "evaluate_synaptic_weight_computation_{}.h5".format(partition_idx))

        def idcs_valid(idcs):
            return (SOLVER_MODES[idcs[0]] != "nef") or (MODES[idcs[1]] !=
                                                        "non_lindep_cosine")

        idcs = list(
            filter(
                idcs_valid,
                itertools.product(range(N_SOLVER_MODES), range(N_MODES),
                                  range(N_QS), range(N_NEURONS),
                                  range(N_REPEAT))))

        # Always shuffle the indices in the right way to not ruin the indexing
        np.random.seed(57482)
        random.shuffle(idcs)

        partitions = np.linspace(0, len(idcs), n_partitions + 1, dtype=int)
        i0 = partitions[partition_idx]
        i1 = partitions[partition_idx + 1]
        print(
            f"Partition {partition_idx} out of {n_partitions} (i0={i0}, i1={i1}); total={len(idcs)}"
        )

        f = h5py.File(fn, 'w')

        f.attrs["solver_modes"] = json.dumps(SOLVER_MODES)
        f.attrs["modes"] = json.dumps(MODES)
        f.attrs["qs"] = json.dumps([int(q) for q in QS])
        f.attrs["neurons"] = json.dumps([int(x) for x in NEURONS])
        f.attrs["xs_sigma_test"] = json.dumps(list(XS_SIGMA_TEST))

        errs_tuning = f.create_dataset(
            "errs_tuning",
            shape=(N_SOLVER_MODES, N_MODES, N_QS, N_NEURONS, N_REPEAT,
                   N_XS_SIGMA_TEST, N_REPEAT_TEST),
            compression="gzip")
        errs_tuning[...] = np.nan
        errs_delay = f.create_dataset(
            "errs_delay",
            shape=(N_SOLVER_MODES, N_MODES, N_QS, N_NEURONS, N_REPEAT,
                   N_XS_SIGMA_TEST, N_REPEAT_TEST, N_DELAYS_TEST),
            compression="gzip")
        errs_delay[...] = np.nan

        return idcs[i0:i1], f

    def task_serialise(res, f):
        idcs, Es_tuning, Es_delay = res
        f["errs_tuning"][idcs] = Es_tuning
        f["errs_delay"][idcs] = Es_delay
        f.flush()

    f = working_multiprocessing_pool.run(sys.argv, task_init, execute_single,
                                     task_serialise)
    if not f is None:
        f.close()


if __name__ == "__main__":
    main()

