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


def pack(As):
    return np.packbits(As.astype(bool))


def main():
    np.random.seed(58381)

    fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
                      "spatio_temporal_network_nef.h5")

    n_neurons = N_NEURONS
    n_temporal_dimensions = q = N_TEMP_DIMS
    n_dimensions = d = N_DIMS

    gains, biases, _ = nonneg_common.mk_ensemble(n_neurons, d=1)
    encoders = mk_encs(n_neurons, q * d)

    A, B = bases.mk_ldn_lti(n_temporal_dimensions)

    Ap = np.zeros((q * d, q * d))
    Bp = np.zeros((q * d, d))
    for i in range(d):
        Ap[(i * q):((i + 1) * q), (i * q):((i + 1) * q)] = TAU * A + np.eye(q)
        Bp[(i * q):((i + 1) * q), i] = TAU * B

    with nengo.Network() as model:
        nd_in = nengo.Node(size_in=n_dimensions)
        for i in range(n_dimensions):
            nd_noise = nengo.Node(
                nengo.processes.WhiteSignal(period=T_SIM * 2.0,
                                            high=1.0,
                                            y0=0.0,
                                            rms=0.5))
            nengo.Connection(nd_noise, nd_in[i], synapse=None)

        ens_x = nengo.Ensemble(
            n_neurons=n_neurons,
            dimensions=n_dimensions * n_temporal_dimensions,
            bias=biases,
            gain=gains,
            encoders=encoders,
        )

        nengo.Connection(nd_in, ens_x, transform=Bp, synapse=TAU)
        nengo.Connection(ens_x, ens_x, transform=Ap, synapse=TAU)

        p_in = nengo.Probe(nd_in, synapse=None)
        p_out = nengo.Probe(ens_x.neurons, synapse=None)

    with nengo.Simulator(model) as sim:
        sim.run(T_SIM * 2.0)

    xs, As = sim.data[p_in], sim.data[p_out]

    Nmid = int(T_SIM / DT)
    xs_test, As_test = xs[:Nmid], As[:Nmid]
    xs_train, As_train = xs[Nmid:], As[Nmid:]

    with h5py.File(fn, "w") as f:
        f.create_dataset("Bp", data=Bp)
        f.create_dataset("Ap", data=Ap)
        f.create_dataset("gains", data=gains)
        f.create_dataset("biases", data=biases)
        f.create_dataset("encoders", data=encoders)
        f.create_dataset("xs_train", data=xs_train)
        f.create_dataset("xs_test", data=xs_test)
        f.create_dataset("As_train", data=pack(As_train), compression="gzip")
        f.create_dataset("As_test", data=pack(As_test), compression="gzip")
        f.create_dataset("As_shape", data=As_train.shape)


if __name__ == "__main__":
    main()

