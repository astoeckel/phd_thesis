#!/usr/bin/env python3

#    Code for the "Nonlinear Synaptic Interaction" Paper
#    Copyright (C) 2017-2020   Andreas Stöckel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))

from nlif_parameters import *
from nlif_sim_2d_fun_network import run_single_spiking_trial

import numpy as np
import multiprocessing
import time
import random

import env_guard
import h5py
import tqdm

BENCHMARK_FUNCTION_KEYS = [
	"addition",
	"multiplication_limited",
	"multiplication",
	"sqrt-multiplication",
	"sqr-multiplication",
	"shunting",
	"norm",
	"arctan",
	"max",
]

N_BENCHMARK_FUNCTION_KEYS = len(BENCHMARK_FUNCTION_KEYS)

NEURON_KEYS = [
	"lif",
	"lif_2d",
	"two_comp",
	"three_comp",
#	"four_comp"
]

N_NEURON_KEYS = len(NEURON_KEYS)

PARAM_SETS = [
	"normal",
	"optimised",
]

N_PARAM_SETS = len(PARAM_SETS)

N_REPEAT = 10

def run_single_experiment(idcs):
    i_neuron, i_param_set, i_fun, i_repeat = idcs

    rng = np.random.RandomState(4917 * i_repeat + 373)

    # Fetch the neuron model
    kwargs = {
        "model_name": NEURON_KEYS[i_neuron],
        "f": BENCHMARK_FUNCTIONS[BENCHMARK_FUNCTION_KEYS[i_fun]],
        "intermediate": False,
        "rng": rng,
    }
    if kwargs["model_name"].endswith("_2d"):
        kwargs["intermediate"] = True
        kwargs["model_name"] = "_".join(kwargs["model_name"].split("_")[:-1])

    # Switch to the optimised parameter set
    if (PARAM_SETS[i_param_set] == "optimised"):
        if (kwargs["model_name"] == "three_comp") or (kwargs["model_name"] == "four_comp"):
            kwargs["reg"] = 1e-6
        kwargs["intercepts_tar"] = (-0.95, 0.0)
        kwargs["pinh"] = None

    # For simpler neuron types, fewer iterations suffice
    if kwargs["model_name"] == "lif":
        kwargs["N_epochs"] = 1
    elif kwargs["model_name"] == "two_comp":
        kwargs["N_epochs"] = 10

    # Run the actual experiment
    res = run_single_spiking_trial(**kwargs)

    return idcs, (res["errors"]["Emodel"], res["errors"]["Enet"])


def main():
    print("Running network spatial frequency experiments...")
    if len(sys.argv) == 1:
        fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'nlif_2d_benchmark_functions_network.h5')
    else:
        fn = sys.argv[1]

    os.makedirs(os.path.dirname(fn), exist_ok=True)
    print("Writing to {}".format(fn))
    with h5py.File(fn, 'w') as h5f:
        h5f.create_dataset("function_keys", data="\n".join(BENCHMARK_FUNCTION_KEYS))
        h5f.create_dataset("neurons", data="\n".join(NEURON_KEYS))

        # Create the errors dataset
        errs = np.zeros(
            (N_NEURON_KEYS, N_PARAM_SETS, N_BENCHMARK_FUNCTION_KEYS, N_REPEAT, 2)) * np.nan
        if not "errs" in h5f:
            h5f.create_dataset("errs", data=errs)

        # Assemble the parameters we sweep over
        params = [
            (i_neuron, i_param_set, i_fun, i_repeat)
            for i_neuron in range(N_NEURON_KEYS)
            for i_param_set in range(N_PARAM_SETS)
            for i_fun in range(N_BENCHMARK_FUNCTION_KEYS)
            for i_repeat in range(N_REPEAT)
            if (i_param_set == 0) or (i_fun < 3)
        ]
        n_total = len(params)
        random.shuffle(params)

        with env_guard.SingleThreadEnvGuard():
            with multiprocessing.get_context('spawn').Pool(N_CPUS) as pool:
                for ((i, j, k, l), Es) in tqdm.tqdm(pool.imap_unordered(run_single_experiment, params),
                                      total=n_total):
                    h5f["errs"][i, j, k, l] = Es
                    h5f.flush()

    print("Done!")

if __name__ == "__main__":
    main()
