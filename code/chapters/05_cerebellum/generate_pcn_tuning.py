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
import numpy as np
import nengo_bio as bio

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

import benchmark


def measure_granule_epscs(seed, bias_mode, T, n_granule=10000):
    np.random.seed(seed)
    ts, xs, ys, _, _, spatial_data, pcn_spikes = benchmark.build_and_run_test_network(
        T=T,
        input_descr=benchmark.pulse_input(t_on=0.25, t_off=1.75),
        probe_granule_decoded=True,
        probe_spatial_data=True,
        probe_pcn_spikes=True,
        mode='two_populations_dales_principle',
        bias_mode=bias_mode,
        pcn_max_rates=(25, 50),
        bias_mode_golgi=bio.JBias,
        tau=60e-3,
        use_spatial_constraints=True,
        n_pcn_golgi_convergence=100,
        n_pcn_granule_convergence=5,
        n_granule_golgi_convergence=100,
        n_golgi_granule_convergence=15,
        n_golgi_golgi_convergence=100,
        n_granule=n_granule,
        n_golgi=100,
        q=6,
    )

    W_exc = spatial_data["W"][bio.Excitatory][:100]
    conn_exc = 1.0 * (W_exc > 1e-12)
    epscs = ((1.0 * (pcn_spikes > 0.0)) @ conn_exc).astype(bool)

    idcs_with_input = np.where(xs > 0.0)[0]
    idcs_without_input = np.where(xs == 0.0)[0]

    T_with_input = len(idcs_with_input) * 1e-3
    T_without_input = len(idcs_without_input) * 1e-3

    n_epscs_with_input = np.sum(
        epscs[idcs_with_input]) / (n_granule * T_with_input)
    n_epscs_without_input = np.sum(
        epscs[idcs_without_input]) / (n_granule * T_without_input)

    print(n_epscs_with_input, n_epscs_without_input)

    return {
        "ts": ts,
        "xs": xs,
        "epscs": epscs,
        "n_granule": n_granule,
        "n_epscs_with_input": n_epscs_with_input,
        "n_epscs_without_input": n_epscs_without_input,
    }


data = measure_granule_epscs(12397, "realistic_pcn_intercepts", 10.0)
fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'granule_pcn_tuning.npz')
np.savez(fn, **data)

