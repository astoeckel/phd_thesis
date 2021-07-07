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

from two_comp_sim_2d_fun_network import *
import env_guard

import numpy as np
import multiprocessing
import time

import h5py
import tqdm


def update_symlink(src, tar):
    if os.path.lexists(tar):
        os.unlink(tar)
    os.symlink(src, tar)


def run_single_experiment(idcs):
    i_param, i_reg, i_subth, i_repeat = idcs

    # Fetch the parameter key
    p_key = NETWORK_PARAMS_SWEEP_KEYS[i_param]

    # Use an intermediate population if the parameter key ends with "_2d"
    intermediate = p_key.endswith("_2d")

    # Fetch the benchmark function
    f = lambda x, y: 0.5 * (x + 1.0) * (y + 1.0) - 1.0

    # Fetch the regularisation factor
    reg = REGS_FLT_SWEEP_MAP[(p_key, bool(i_subth))][i_reg]

    # Run each experiment twice: once with mask_negative set to true, once with
    # mask_negative set to false
    kwargs = {
        "model_name": p_key,
        "f": f,
        "intermediate": intermediate,
        "reg": reg,
        "silent": True,
    }

    # Compute all network weights in the first pass
    weights = None
    tar_filt = None

    # The result is a matrix
    E = np.zeros((N_TAU_PRE_FILTS, 2))
    for i_pre_filt in range(N_TAU_PRE_FILTS):
        # Chose the pre-filter
        tau_pre_filt = TAU_PRE_FILTS[i_pre_filt]

        # Run the experiment with exactly the same seed
        rng = np.random.RandomState(4917 * i_repeat + 373)
        res = run_single_spiking_trial(**kwargs,
                                       tau_pre_filt=tau_pre_filt,
                                       weights=weights,
                                       tar_filt=tar_filt,
                                       rng=rng,
                                       mask_negative=bool(i_subth))

        # Re-use the weights and signals we computed
        weights = res["weights"]
        tar_filt = res["tar_filt"]

        # Store the computed errors
        E[i_pre_filt] = res["errors"]["Enet"], res["errors"]["Emodel"]

    # Fetch and return the errors
    return idcs, E

def main():
    print("Running network pre filter and regularisation sweep experiments...")
    fn = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "two_comp_benchmark_functions_regularisation_filter_sweep.h5")

    os.makedirs(os.path.dirname(fn), exist_ok=True)
    print("Using {} experiment threads with {} weight solver threads each".format(
        N_CPUS, N_SOLVER_THREADS))
    print("Writing to {}".format(fn))
    with h5py.File(fn, 'w') as h5f:
        # Write all target information
        h5f.create_dataset("network", data=True)
        h5f.create_dataset("params_keys",
                           data="\n".join(NETWORK_PARAMS_SWEEP_KEYS))

        # Create the errors dataset
        errs = np.zeros((N_NETWORK_PARAMS_SWEEP, N_REGS, N_TAU_PRE_FILTS, 2, N_REPEAT,
                         2)) * np.nan
        h5f.create_dataset("errs", data=errs)

        # Assemble the parameters we sweep over
        args = [(i_param, i_reg, i_subth, i_repeat)
                for i_param in range(N_NETWORK_PARAMS_SWEEP)
                for i_subth in [0, 1]
                for i_reg in range(N_REGS) for i_repeat in range(N_REPEAT)]
        n_total = len(args)

        with env_guard.SingleThreadEnvGuard():
            with multiprocessing.get_context("spawn").Pool(N_CPUS) as pool:
                for ((i, j, k, l),
                     E) in tqdm.tqdm(pool.imap_unordered(run_single_experiment, args),
                                     total=n_total):
            h5f["errs"][i, j, :, k, l] = E
            h5f.flush()

    print("Done!")


if __name__ == "__main__":
    main()
