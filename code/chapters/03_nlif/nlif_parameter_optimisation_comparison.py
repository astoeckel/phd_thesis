#!/usr/bin/env python3

#  libnlif -- Multi-compartment LIF simulator and weight solver
#  Copyright (C) 2019-2021  Andreas Stöckel
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import random
import multiprocessing

import tqdm
import numpy as np
import h5py

import nlif
from nlif.parameter_optimisation import optimise_sgd, optimise_trust_region

import env_guard

# Setup some parameters

two_comp_lif_neuron = nlif.TwoCompLIFCond()
three_comp_lif_neuron = nlif.ThreeCompLIFCond(g_c2=200e-9)
with nlif.Neuron() as four_comp_lif_neuron:
    with nlif.Soma() as soma:
        nlif.CondChan(E_rev=-65e-3, g=50e-9)
    with nlif.Compartment() as comp1:
        nlif.CondChan(E_rev=-65e-3, g=50e-9)
        nlif.CondChan(E_rev=0e-3)
        nlif.CondChan(E_rev=-75e-3)
    with nlif.Compartment() as comp2:
        nlif.CondChan(E_rev=-65e-3, g=50e-9)
        nlif.CondChan(E_rev=0e-3)
        nlif.CondChan(E_rev=-75e-3)
    with nlif.Compartment() as comp3:
        nlif.CondChan(E_rev=-65e-3, g=50e-9)
        nlif.CondChan(E_rev=0e-3)
        nlif.CondChan(E_rev=-75e-3)
    nlif.Connection(soma, comp1)
    nlif.Connection(comp1, comp2, g_c=200e-9)
    nlif.Connection(comp2, comp3, g_c=500e-9)

NEURONS = [
    two_comp_lif_neuron,
    three_comp_lif_neuron,
    four_comp_lif_neuron,
]
N_NEURONS = len(NEURONS)

N_OPTIMISERS = 2

N_PARAMS = 1
PARAMS = [
    [1e-5],
    [5e-2],
]

N_REPEAT = 1000

N_EPOCHS = 500
N_SMPLS = 300


def run_single(args):
    i, j, k, l = args

    neuron = NEURONS[i]
    assm = neuron.assemble()

    rng = np.random.RandomState(12903 * l + 21)

    gs_train = rng.uniform(0, 1000e-9, (N_SMPLS, assm.n_inputs))
    gs_test = rng.uniform(0, 1000e-9, (N_SMPLS + 1, assm.n_inputs))

    Js_train = assm.isom_empirical_from_rate(gs_train, progress=False)
    Js_test = assm.isom_empirical_from_rate(gs_test, progress=False)

    valid_train = Js_train > 1e-9
    valid_test = Js_test > 1e-9

    sys = assm.reduced_system().condition()

    try:
        if j == 0:
            _, errs_train, errs_test = optimise_trust_region(sys,
                                               gs_train[valid_train],
                                               Js_train[valid_train],
                                               gs_test[valid_test],
                                               Js_test[valid_test],
                                               alpha3=PARAMS[j][k],
                                               gamma=0.99,
                                               N_epochs=N_EPOCHS,
                                               progress=False,
                                               parallel_compile=False)
        elif j == 1:
            _, errs_train, errs_test = optimise_sgd(sys,
                                      gs_train[valid_train],
                                      Js_train[valid_train],
                                      gs_test[valid_test],
                                      Js_test[valid_test],
                                      alpha=PARAMS[j][k],
                                      N_batch=10,
                                      N_epochs=N_EPOCHS,
                                      rng=rng,
                                      progress=False)
    except:
        errs_train, errs_test = np.ones((2, N_EPOCHS + 1)) * np.nan

    return i, j, k, l, (errs_train, errs_test), (sum(valid_train), sum(valid_test))


def main():
    # Fill the parameter array
    params = [(i, j, k, l) for i in range(N_NEURONS)
              for j in range(N_OPTIMISERS) for k in range(N_PARAMS)
              for l in range(N_REPEAT)]
    random.shuffle(params)

    fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
                      'nlif_parameter_optimisation_comparison.h5')
    with h5py.File(fn, 'w') as f:
        f.create_dataset('params', data=PARAMS)

        errs = f.create_dataset(
            'errs',
            (N_NEURONS, N_OPTIMISERS, N_PARAMS, N_REPEAT, 2, N_EPOCHS + 1))

        smpls = f.create_dataset(
            'smpls',
            (N_NEURONS, N_OPTIMISERS, N_PARAMS, N_REPEAT, 2))

        with env_guard.SingleThreadEnvGuard():
            with multiprocessing.get_context('spawn').Pool() as pool:
                for i, j, k, l, E, n_smpls in tqdm.tqdm(pool.imap_unordered(
                        run_single, params),
                                                  total=len(params)):
                    errs[i, j, k, l] = E
                    smpls[i, j, k, l] = n_smpls

if __name__ == "__main__":
    main()

