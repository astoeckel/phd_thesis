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

N_NEURONS = 1000
N_TEMP_DIMS = 5
N_DIMS = 2

USE_SPATIOTEMPORAL_MATRICES = False

BASIS = "legendre"
WINDOW = "erasure"

TAU = 100e-3

T_TRAIN = 3.0
N_SMPLS = 1000
XS_SIGMA_TRAIN = 3.0
XS_SIGMA_SIM = 1.0

T_SIM = 100.0
DT = 1e-3

#
# Main program
#


def mk_encs(n, d, rng=np.random):
    encs = rng.normal(0, 1, (n, d))
    encs /= np.linalg.norm(encs, axis=1)[:, None]
    return encs


def mk_gaussian_basis(q, N, T=1.0, dt=1e-3):
    ts = np.arange(N) * dt
    mus = np.random.uniform(-0.1, T, q)
    sigmas = np.power(10.0, np.random.uniform(-1.0, -0.5, q))
    res = np.exp(-np.square(ts[None, :] - mus[:, None]) /
                  np.square(sigmas[:, None]))
    return res / np.sum(res, axis=1)[:, None] / dt


def execute_network(W_in, W_rec, gains, biases, T=T_SIM, dt=DT, tau=TAU):
    N = int(T / dt + 1e-9)
    n_dims = N_DIMS
    n_neurons = len(gains)

    with nengo.Network() as model:
        nd_in = nengo.Node(size_in=n_dims)
        for i in range(n_dims):
            nd_noise = nengo.Node(
                nengo.processes.WhiteSignal(period=100.0,
                                            high=XS_SIGMA_SIM,
                                            y0=0.0,
                                            rms=0.5))
            nengo.Connection(nd_noise, nd_in[i], synapse=None)

        ens_x = nengo.Ensemble(n_neurons=n_neurons,
                               dimensions=1,
                               bias=biases,
                               gain=gains,
                               encoders=np.ones((n_neurons, 1)))

        nengo.Connection(nd_in,
                         ens_x.neurons,
                         transform=W_in[:, :, 0],
                         synapse=tau)

        nengo.Connection(ens_x.neurons,
                         ens_x.neurons,
                         transform=W_rec[:, :, 0],
                         synapse=tau)

        p_in = nengo.Probe(nd_in, synapse=None)
        p_out = nengo.Probe(ens_x.neurons, synapse=None)

    with nengo.Simulator(model, dt=dt) as sim:
        sim.run(T)

    return sim.data[p_in], sim.data[p_out]


def pack(As):
    return np.packbits(As.astype(bool))


def main():
    global BASIS
    global USE_SPATIOTEMPORAL_MATRICES

    np.random.seed(58381)

    if len(sys.argv) > 1:
        if sys.argv[1] == "gaussian":
            BASIS = "gaussian"
            fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
                              "spatio_temporal_network_gaussian.h5")
        elif sys.argv[1] == "matrices":
            USE_SPATIOTEMPORAL_MATRICES = True
            fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
                              "spatio_temporal_network_matrices.h5")
    else:
        fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
                          "spatio_temporal_network.h5")

    gains, biases, _ = nonneg_common.mk_ensemble(N_NEURONS, d=1)
    G = lif_utils.lif_rate
    if USE_SPATIOTEMPORAL_MATRICES:
        print("Generating spatiotemporal matrices...")
        MEs = mk_encs(N_NEURONS, N_DIMS * N_TEMP_DIMS).reshape(N_NEURONS, N_DIMS, N_TEMP_DIMS)
    else:
        print("Generating spatial and temporal encoders...")
        TEs = mk_encs(N_NEURONS, N_TEMP_DIMS)
        Es = mk_encs(N_NEURONS, N_DIMS)

    if BASIS == "gaussian":
        print("Generating Gaussian tuning curves")
        N = int(T_TRAIN / DT + 1e-9)
        Ms = mk_gaussian_basis(N_NEURONS, N).T
        TEs = np.diag(np.random.choice([1.0, -1.0], N_NEURONS))
    else:
        print("Generating LDN impulse response")
        _, Ms = basis_delay_analysis_common.mk_impulse_response(
            basis=BASIS,
            window=WINDOW,
            q=N_TEMP_DIMS,
            T=T_TRAIN,
            dt=DT,
            use_euler=True,
            rescale_ldn=False)

    flts_in = [(TAU, )]
    flts_rec = [(TAU, )]

    if USE_SPATIOTEMPORAL_MATRICES:
        print("Solving for weights with spatiotemporal matrices...")
        W_in, W_rec, errs = temporal_encoder_common.solve_for_recurrent_population_weights_with_spatiotemporal_matrices(
            G,
            gains,
            biases,
            None,
            None,
            MEs,
            [Filters.lowpass(*flt_in) for flt_in in flts_in],
            [Filters.lowpass(*flt_rec) for flt_rec in flts_rec],
            Ms=Ms,
            N_smpls=N_SMPLS,
            T=T_TRAIN,
            dt=DT,
            xs_sigma=XS_SIGMA_TRAIN,
            biased=False,
        )
    else:
        print("Solving for weights with temporal and spatial encoding vectors...")
        W_in, W_rec, errs = temporal_encoder_common.solve_for_recurrent_population_weights_with_spatial_encoder(
            G,
            gains,
            biases,
            None,
            None,
            TEs,
            Es,
            [Filters.lowpass(*flt_in) for flt_in in flts_in],
            [Filters.lowpass(*flt_rec) for flt_rec in flts_rec],
            Ms=Ms,
            N_smpls=N_SMPLS,
            T=T_TRAIN,
            dt=DT,
            xs_sigma=XS_SIGMA_TRAIN,
            biased=False,
        )

    xs_test, As_test = execute_network(W_in, W_rec, gains, biases)
    xs_train, As_train = execute_network(W_in, W_rec, gains, biases)

    with h5py.File(fn, "w") as f:
        f.create_dataset("W_in", data=W_in)
        f.create_dataset("W_rec", data=W_rec)
        f.create_dataset("gains", data=gains)
        f.create_dataset("biases", data=biases)
        if USE_SPATIOTEMPORAL_MATRICES:
            f.create_dataset("MEs", data=MEs)
        else:
            f.create_dataset("Es", data=Es)
            f.create_dataset("TEs", data=TEs)
        f.create_dataset("xs_train", data=xs_train)
        f.create_dataset("xs_test", data=xs_test)
        f.create_dataset("As_train", data=pack(As_train), compression="gzip")
        f.create_dataset("As_test", data=pack(As_test), compression="gzip")
        f.create_dataset("As_shape", data=As_train.shape)


if __name__ == "__main__":
    main()

