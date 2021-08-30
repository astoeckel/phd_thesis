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

DIRNAME = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")

N_REPEAT = 100

def 𐌈(**kwargs):
    return kwargs


def run_experiment(fn, **kwargs):
    tss, xss, yss, basiss, sigmas = [[] for _ in range(5)]

    for i in range(N_REPEAT):
        it = 0
        while True:
            np.random.seed(12369 + i + N_REPEAT * it)
            ts, xs, ys, As, theta = benchmark.build_and_run_test_network(
                T=2.29,
                input_descr = benchmark.pulse_input(t_on=0.25, t_off=1.75),
                probe_granule_decoded=True,
                probe_spatial_data=False,
                **kwargs
            )

            # Compute this for a random subset of neurons
            if As.shape[1] > 1000:
                all_idcs = np.arange(As.shape[1], dtype=int)
                idcs = np.random.RandomState(58791).choice(all_idcs,
                                                           1000,
                                                           replace=False)
                As = As[:, idcs]

            try:
                U, S, V = np.linalg.svd(As - np.mean(As, axis=0))
                basis = U[:, :ys.shape[1]]
                break
            except np.linalg.LinAlgError:
                print("Error computing SVD, trying again")
            it += 1

        tss.append(ts)
        xss.append(xs)
        yss.append(ys)
        basiss.append(basis)
        sigmas.append(S)

    np.savez(os.path.join(DIRNAME, fn), **{
        "ts": np.array(tss),
        "xs": np.array(xss),
        "ys": np.array(yss),
        "theta": theta,
        "basis": np.array(basiss),
        "sigma": np.array(sigmas),
    })


if __name__ == "__main__":
    default_kwargs = 𐌈(
        bias_mode="jbias_uniform_pcn_intercepts",
        pcn_max_rates=(25, 75),
    )

    detailed_kwargs = 𐌈(
        bias_mode="jbias_realistic_pcn_intercepts",
        use_spatial_constraints=True,
        n_pcn_golgi_convergence=100,
        n_pcn_granule_convergence=5,
        n_granule_golgi_convergence=100,
        n_golgi_granule_convergence=5,
        n_golgi_golgi_convergence=100,
        n_granule=10000,
        n_golgi=100,
        pcn_max_rates=(25, 75),
    )

    detailed_kwargs_no_jbias = dict(detailed_kwargs)
    detailed_kwargs_no_jbias["bias_mode"] = "realistic_pcn_intercepts"

    detailed_kwargs_control = dict(detailed_kwargs)
    detailed_kwargs_control["use_control_lti"] = True

    run_experiment(
        fn="temporal_basis_direct.npz",
        mode="direct",
        **default_kwargs,
    )

    run_experiment(
        fn="temporal_basis_single_population.npz",
        mode="single_population",
        **default_kwargs,
    )

    run_experiment(
        fn="temporal_basis_two_populations.npz",
        mode="two_populations",
        **default_kwargs,
    )

    run_experiment(
        fn="temporal_basis_two_populations_dales_principle.npz",
        mode="two_populations_dales_principle",
        **default_kwargs,
    )

    run_experiment(
        fn="temporal_basis_two_populations_dales_principle_detailed.npz",
        mode="two_populations_dales_principle",
        **detailed_kwargs,
    )

    run_experiment(
        fn=
        "temporal_basis_two_populations_dales_principle_detailed_no_jbias.npz",
        mode="two_populations_dales_principle",
        **detailed_kwargs_no_jbias)

    run_experiment(
        fn=
        "temporal_basis_two_populations_dales_principle_detailed_control.npz",
        mode="two_populations_dales_principle",
        **detailed_kwargs_control)

