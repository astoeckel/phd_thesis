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

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

import benchmark

DIRNAME = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")

def 𐌈(**kwargs):
    return kwargs

def get_benchmark_sweep(input_type="white_noise", **kwargs):
    # Generate a different input sweep depending on the "input_type" parameter
    if input_type == "white_noise":
        input_cback = lambda i, n: \
            benchmark.white_noise_input(benchmark.slin(0.1, 10.0)(i, n))
    elif input_type == "pulse":
        input_cback = lambda i, n: \
            benchmark.pulse_input(1.0 - benchmark.slin(0.1, 0.9)(i, n),
                                        benchmark.slin(0.1, 0.9)(i, n))
    else:
        raise RuntimeError("Invalid input_type")

    # Combine the input sweep with the default parameters
    return benchmark.sweep(
        "input_descr", 21, input_cback,
        [{**kwargs, **{
            "qp_solver_extra_args": {
                "max_iter": 200,
                "n_threads": 1,
            },
        }}]
    )

def get_benchmark_params_single(**kwargs):
    """
    Produces a single 2D sweep over the delays and the input parameter.
    """
    return {
        "dirname": DIRNAME,
        "sweep": get_benchmark_sweep(**kwargs),
        "n_repeat": 1,
        "n_delays": 21,
        "concurrency": None,
        "randomize_all": False,
    }

def get_benchmark_params_multi(name, n, cback, **kwargs):
    """
    Produces multiple 2D sweeps while varying another parameter.
    """
    return {
        "dirname": DIRNAME,
        "sweep": benchmark.sweep(name, n, cback, get_benchmark_sweep(**kwargs)),
        "n_delays": 21,
        "concurrency": None,
        "n_repeat": 1, # Only one repetition, but randomize all experiments
        "randomize_all": True,
    }

def run_benchmark_suite(mode, filename_prefix=None, **kwargs):
    INPUT_TYPE_SHORT = {
        "white_noise": "wn",
        "pulse": "pl"
    }

    for input_type in ["pulse", "white_noise"]:
        prefix = "{}_{}{}".format(
            INPUT_TYPE_SHORT[input_type],
            mode, "" if filename_prefix is None else ("_" + filename_prefix))

        benchmark.run_benchmark(
            filename_prefix=prefix,
            **get_benchmark_params_single(
                input_type=input_type,
                mode=mode,
                **kwargs,
            ))

        if input_type == "pulse":
            continue

        if filename_prefix == "detailed":
            benchmark.run_benchmark(
                filename_prefix="sweep_tau_" + prefix,
                **get_benchmark_params_multi(
                    "tau", 10, benchmark.slin(10e-3, 100e-3),
                    input_type=input_type,
                    mode=mode,
                    **kwargs,
                ))

            benchmark.run_benchmark(
                filename_prefix="sweep_n_pcn_granule_convergence_" + prefix,
                **get_benchmark_params_multi(
                    "n_pcn_granule_convergence", 10, lambda i, n: 2 * i + 1,
                    input_type=input_type,
                    mode=mode,
                    record_weights=False,
                    **kwargs,
                ))

            benchmark.run_benchmark(
                filename_prefix="sweep_n_golgi_granule_convergence_" + prefix,
                **get_benchmark_params_multi(
                    "n_golgi_granule_convergence", 10, lambda i, n: 2 * i + 1,
                    input_type=input_type,
                    mode=mode,
                    record_weights=True,
                    **kwargs,
                ))

if __name__ == "__main__":
    default_kwargs = 𐌈(
            bias_mode="jbias_uniform_pcn_intercepts",
    )

    detailed_kwargs = 𐌈(
            bias_mode="jbias_very_realistic_pcn_intercepts",
            use_spatial_constraints=True,
            n_pcn_golgi_convergence=100,
            n_pcn_granule_convergence=5,
            n_granule_golgi_convergence=100,
            n_golgi_granule_convergence=5,
            n_golgi_golgi_convergence=100,
            n_granule=10000,
            n_golgi=100,
            pcn_max_rates=(50, 120),
    )

    run_benchmark_suite(
            mode="direct",
            **default_kwargs,
    )

    run_benchmark_suite(
            mode="single_population",
            **default_kwargs,
    )

    run_benchmark_suite(
            mode="two_populations",
            **default_kwargs,
    )

    run_benchmark_suite(
            mode="two_populations_dales_principle",
            **default_kwargs,
    )

    run_benchmark_suite(
            mode="two_populations_dales_principle",
            filename_prefix="detailed",
            **detailed_kwargs
    )

    # Pack all recorded weights into a single tar file
    import subprocess
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data'))
    subprocess.run(["tar", "-cf", 'weights.tar', 'weights'])

