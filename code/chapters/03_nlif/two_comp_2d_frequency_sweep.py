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

# Try to disable detrimental numpy multithreading
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from two_comp_approx_2d_fun import *

import numpy as np
import multiprocessing
import time
import random

import h5py
import tqdm


def update_symlink(src, tar):
    if os.path.lexists(tar):
        os.unlink(tar)
    os.symlink(src, tar)


def run_single_experiment(idcs):
    i_param, i_sweep, i_repeat = idcs
    p_key = SOLVER_PARAMS_SWEEP_KEYS[i_param]
    sigma = SIGMAS[i_sweep]
    reg = (SOLVER_REG_MAP[p_key, True], SOLVER_REG_MAP[p_key, False])
    return (idcs, *run_single_experiment_common(sigma, reg, p_key, i_repeat))


print("Running spatial frequency experiments...")
args = [(param, sigma, repeat) for param in range(N_SOLVER_PARAMS_SWEEP)
        for sigma in range(N_SIGMAS) for repeat in range(N_REPEAT)]
random.shuffle(args)

with multiprocessing.Pool(N_CPUS) as pool:
    errs = np.zeros((N_SOLVER_PARAMS, N_SIGMAS, N_REPEAT, 2))
    for ((i, j, k), e1,
         e2) in tqdm.tqdm(pool.imap_unordered(run_single_experiment, args),
                          total=len(args)):
        errs[i, j, k, 0] = e1
        errs[i, j, k, 1] = e2

print("Writing results to disk...")
fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'two_comp_2d_frequency_sweep.h5')
os.makedirs(os.path.dirname(fn), exist_ok=True)
with h5py.File(fn, 'w') as f:
    f.create_dataset("network", data=False)
    f.create_dataset("sigmas", data=SIGMAS)
    f.create_dataset("params_keys", data="\n".join(SOLVER_PARAMS_SWEEP_KEYS))
    f.create_dataset("errs", data=errs)

print("Done!")

