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

from two_comp_sim_2d_fun_network import *

import numpy as np
import multiprocessing
import time
import random

import env_guard
import h5py
import tqdm


def run_single_experiment(idcs):
    i_param, i_sweep, i_repeat = idcs

    # Fetch the parameter key
    p_key = NETWORK_PARAMS_SWEEP_KEYS[i_param]

    # Use an intermediate population if the parameter key ends with "_2d"
    intermediate = p_key.endswith("_2d")

    # Fetch the function filter coefficient
    sigma = SIGMAS[i_sweep]

    # Run each experiment twice: once with mask_negative set to true, once with
    # mask_negative set to false
    kwargs = {
        "model_name": p_key,
        "f": sigma,
        "intermediate": intermediate,
        "silent": True,
    }

    # With "mask negative"
    rng = np.random.RandomState(4917 * i_repeat + 373)
    kwargs["reg"] = NETWORK_REG_MAP[(p_key, True)]
    res1 = run_single_spiking_trial(**kwargs, rng=rng, mask_negative=True)

    # Same experiment without "mask negative"
    rng = np.random.RandomState(4917 * i_repeat + 373)
    kwargs["reg"] = NETWORK_REG_MAP[(p_key, False)]
    res2 = run_single_spiking_trial(**kwargs, rng=rng, mask_negative=False)

    # Fetch and return the errors
    E1 = res1["errors"]
    E2 = res2["errors"]
    return idcs, E1["Enet"], E2["Enet"], E1["Emodel"], E2["Emodel"]


def main():
    print("Running network spatial frequency experiments...")
    if len(sys.argv) == 1:
        fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
                          'two_comp_2d_frequency_sweep_network.h5')
    else:
        fn = sys.argv[1]

    os.makedirs(os.path.dirname(fn), exist_ok=True)
    print("Using {} experiment threads with {} weight solver threads each".
          format(N_CPUS, N_SOLVER_THREADS))
    print("Writing to {}".format(fn))
    with h5py.File(fn, 'a') as f:
        # Write all target information
        if not "network" in f:
            f.create_dataset("network", data=True)
        if not "sigmas" in f:
            f.create_dataset("sigmas", data=SIGMAS)
        if not "params_keys" in f:
            f.create_dataset("params_keys",
                             data="\n".join(NETWORK_PARAMS_SWEEP_KEYS))

        # Create the errors dataset
        errs = np.zeros(
            (N_NETWORK_PARAMS_SWEEP, N_SIGMAS, N_REPEAT, 4)) * np.nan
        if not "errs" in f:
            f.create_dataset("errs", data=errs)

        # Assemble the parameters we sweep over
        args = [(param, sigma, repeat)
                for param in range(N_NETWORK_PARAMS_SWEEP)
                for sigma in range(N_SIGMAS) for repeat in range(N_REPEAT)]
        n_total = len(args)

        # Filter out all already completed sweeps
        args = list(filter(lambda x: np.any(np.isnan(f["errs"][x])), args))
        n_completed = n_total - len(args)
        if n_completed > 0:
            print("Continuing from {} already completed tasks...".format(
                n_completed))

        with env_guard.SingleThreadEnvGuard():
            with multiprocessing.get_context('spawn').Pool(N_CPUS) as pool:
                for ((i, j, k), e1, e2, e3, e4) in tqdm.tqdm(
                        pool.imap_unordered(run_single_experiment, args),
                        total=n_total,
                        initial=n_completed):
                    f["errs"][i, j, k] = (e1, e2, e3, e4)
                    f.flush()

    print("Done!")


if __name__ == "__main__":
    main()

