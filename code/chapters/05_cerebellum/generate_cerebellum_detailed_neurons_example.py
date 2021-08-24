#!/usr/bin/env python3

#   Code for the PhD Thesis
#   "Harnessing Neural Dynamics as a Computational Resource: Building Blocks
#   for Computational Neuroscience and Artificial Agents"
#   Copyright (C) 2021  Andreas Stöckel
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import h5py
import numpy as np
from nengo_extras.plot_spikes import preprocess_spikes

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

import benchmark

T = 4.0


def unfilter_spike_train(ts, As):
    dt = (ts[-1] - ts[0]) / (len(ts) - 1)
    dAs = (As[1:] - As[:-1])
    spikes = np.zeros_like(As)
    spikes[1:][dAs > 5] = 1 / dt
    return spikes


def make_xs_tars(ts, xs, theta, delays):
    dt = (ts[-1] - ts[0]) / (len(ts) - 1)

    # Compute how well the delayed signal can be approximated
    xs_tars = []
    for i, delay in enumerate(delays):
        # Shift the input signal by n samples
        n = int(theta * delay / dt)
        n0, n1 = 0, max(0, len(xs) - n)
        xs_tars.append(np.concatenate((np.zeros(n), xs[n0:n1])))
    return np.array(xs_tars)


def decode_delay(xs_tar, As, sigma=0.1, seed=58791, dt=1e-3):
    # Check some dimensions
    assert xs_tar.ndim == 1
    assert As.ndim == 2
    assert xs_tar.shape[0] == As.shape[0]

    # Compute this for a random subset of neurons
    n0 = int(0.0 / dt)
    if As.shape[1] > 1000:
        all_idcs = np.arange(As.shape[1], dtype=int)
        idcs = np.random.RandomState(seed).choice(all_idcs,
                                                  1000,
                                                  replace=False)
        As = As[:, idcs]

    Asp, xs_tarp = As[n0:], xs_tar[n0:]
    reg = Asp.shape[0] * np.square(sigma * np.max(Asp))
    D = np.linalg.lstsq(Asp.T @ Asp + reg * np.eye(Asp.shape[1]),
                        Asp.T @ xs_tarp,
                        rcond=None)[0]

    return As @ D


def 𐌈(**kwargs):
    return kwargs


detailed_kwargs = 𐌈(
    mode="two_populations_dales_principle",
    use_spatial_constraints=True,
    n_pcn_golgi_convergence=100,
    n_pcn_granule_convergence=5,
    n_granule_golgi_convergence=100,
    n_golgi_granule_convergence=5,
    n_golgi_golgi_convergence=100,
    n_granule=10000,
    n_golgi=100,
)

print("Running pulse input experiment...")
np.random.seed(4192)
res1 = benchmark.build_and_run_test_network(benchmark.pulse_input(0.5, 0.5),
                                            T=T,
                                            probe_granule_decoded=True,
                                            **detailed_kwargs)

print("\nRunning white noise input experiment...")
np.random.seed(4192)
res2 = benchmark.build_and_run_test_network(benchmark.white_noise_input(5.0),
                                            T=T,
                                            probe_granule_decoded=True,
                                            **detailed_kwargs)

print("\nGathering spatial data")
np.random.seed(4192)
res3 = benchmark.build_and_run_test_network(benchmark.pulse_input(0.5, 0.5),
                                            T=1.0,
                                            probe_spatial_data=True,
                                            **detailed_kwargs)


def store_plot_data(grp, ts, xs, ys, As, theta):
    # Select a subset of the spikes to plot
    print("\nExtracting spikes...")
    n_neurons = 40
    spikes = unfilter_spike_train(ts, As)
    spikes = spikes[:,
                    np.random.
                    choice(np.arange(As.shape[1]), n_neurons, replace=False)]
    _, spikes = preprocess_spikes(ts, spikes)

    # Compute the decoded signals
    delays = np.linspace(0.0, 1.0, 3)
    xs_tars = make_xs_tars(ts, xs, theta, delays)
    ys_hats = np.zeros((len(delays), len(xs)))
    for i in range(len(delays)):
        print("Computing delay decoders ({}/{})...".format(i + 1, len(delays)))
        ys_hats[i] = decode_delay(xs_tars[i], As)

    # Store the signals in the target file
    grp.create_dataset("ts", data=ts, compression="gzip")
    grp.create_dataset("xs", data=xs, compression="gzip")
    grp.create_dataset("ys", data=ys, compression="gzip")
    grp.create_dataset("spikes", data=spikes, compression="gzip")
    grp.create_dataset("delays", data=delays, compression="gzip")
    grp.create_dataset("xs_tars", data=xs_tars, compression="gzip")
    grp.create_dataset("ys_hats", data=ys_hats, compression="gzip")
    grp.create_dataset("theta", data=xs_tars, compression="gzip")


fn = os.path.join(os.path.dirname(__file__),
                  "../../../data/cerebellum_detailed_neurons_example.h5")
with h5py.File(fn, "w") as f:
    print("Processing and storing pulse input data...")
    store_plot_data(f.create_group("pulse_input"), *res1)
    print("Processing and storing white noise input data...")
    store_plot_data(f.create_group("white_input"), *res2)

    print("Storing spatial data...")
    grp_spatial = f.create_group("spatial")
    for key, value in res3[-1].items():
        if key != "W":
            grp_spatial.create_dataset(key, data=value, compression="gzip")

