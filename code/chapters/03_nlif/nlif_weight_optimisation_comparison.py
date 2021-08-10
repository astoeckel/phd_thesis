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
import nlif.parameter_optimisation as param_opt
import nlif.weight_optimisation as weight_opt

import env_guard

###########################################################################
# Micro-implementation of the NEF                                         #
###########################################################################


class LIF:
    slope = 2.0 / 3.0

    @staticmethod
    def inverse(a):
        valid = a > 0
        return 1.0 / (1.0 - np.exp(LIF.slope - (1.0 / (valid * a + 1e-6))))

    @staticmethod
    def activity(x):
        valid = x > (1.0 + 1e-6)
        return valid / (LIF.slope - np.log(1.0 - valid * (1.0 / x)))


class Ensemble:
    def __init__(self, n_neurons, n_dimensions, neuron_type=LIF):
        self.neuron_type = neuron_type

        # Randomly select the intercepts and the maximum rates
        self.intercepts = np.random.uniform(-0.95, 0.95, n_neurons)
        self.max_rates = np.random.uniform(0.5, 1.0, n_neurons)

        # Randomly select the encoders
        self.encoders = np.random.normal(0, 1, (n_neurons, n_dimensions))
        self.encoders /= np.linalg.norm(self.encoders, axis=1)[:, None]

        # Compute the current causing the maximum rate/the intercept
        J_0 = self.neuron_type.inverse(0)
        J_max_rates = self.neuron_type.inverse(self.max_rates)

        # Compute the gain and bias
        self.gain = (J_0 - J_max_rates) / (self.intercepts - 1.0)
        self.bias = J_max_rates - self.gain

    def __call__(self, x):
        return self.neuron_type.activity(self.J(x))

    def J(self, x):
        return self.gain[:, None] * self.encoders @ x + self.bias[:, None]


###########################################################################
# Neuron types to try                                                     #
###########################################################################

one_comp_lif_neuron = nlif.LIFCond()
two_comp_lif_neuron = nlif.TwoCompLIFCond()
three_comp_lif_neuron = nlif.ThreeCompLIFCond(g_c1=100e-9, g_c2=200e-9)
with nlif.Neuron() as four_comp_lif_neuron:
    with nlif.Soma() as soma:
        nlif.CondChan(E_rev=-65e-3, g=50e-9)
    with nlif.Compartment() as comp1:
        nlif.CondChan(E_rev=-65e-3, g=50e-9)
        four_comp_lif_neuron.g_E1 = nlif.CondChan(E_rev=0e-3)
        four_comp_lif_neuron.g_I1 = nlif.CondChan(E_rev=-75e-3)
    with nlif.Compartment() as comp2:
        nlif.CondChan(E_rev=-65e-3, g=50e-9)
        four_comp_lif_neuron.g_E2 = nlif.CondChan(E_rev=0e-3)
        four_comp_lif_neuron.g_I2 = nlif.CondChan(E_rev=-75e-3)
    with nlif.Compartment() as comp3:
        nlif.CondChan(E_rev=-65e-3, g=50e-9)
        four_comp_lif_neuron.g_E3 = nlif.CondChan(E_rev=0e-3)
        four_comp_lif_neuron.g_I3 = nlif.CondChan(E_rev=-75e-3)
    nlif.Connection(soma, comp1, g_c=100e-9)
    nlif.Connection(comp1, comp2, g_c=200e-9)
    nlif.Connection(comp2, comp3, g_c=500e-9)

NEURONS = [
#    one_comp_lif_neuron,
    two_comp_lif_neuron,
    three_comp_lif_neuron,
    four_comp_lif_neuron,
]
N_NEURONS = len(NEURONS)

INITIALISATIONS = ["random", "zero_point"]
N_INITIALISATIONS = len(INITIALISATIONS)

FUNCTIONS = [
#    lambda x1, x2: 0.5 * (x1 + x2),
    lambda x1, x2: 0.5 * (x1 * x2 + 1.0),
]
N_FUNCTIONS = len(FUNCTIONS)

OPTIMISERS = [
    "trust_region",
    "lbfgsb",
    "adam",
]
N_OPTIMISERS = len(OPTIMISERS)

N_REPEAT = 100

N_EPOCHS_TR = 100
N_EPOCHS_GRADIENT = 400
N_EPOCHS = max(N_EPOCHS_TR, N_EPOCHS_GRADIENT)

N_TRAIN_RES = 10
N_TEST = 100

NEURON_SYS_CACHE = [None] * N_NEURONS


def get_neuron_sys(neuron):
    # Optimise the neuron itself
    assm = neuron.assemble()
    rng = np.random.RandomState(578281)
    gs_train = rng.uniform(0, 0.5e-6, (1000, assm.n_inputs))
    As_train = assm.rate_empirical(gs_train, progress=False)
    valid_train = As_train > 12.5
    Js_train = assm.lif_rate_inv(As_train)
    sys = assm.reduced_system().condition()
    sys, _ = param_opt.optimise_trust_region(sys,
                                             gs_train[valid_train],
                                             Js_train[valid_train],
                                             alpha3=1e-5,
                                             gamma=0.99,
                                             N_epochs=100,
                                             progress=False,
                                             parallel_compile=False)

    return sys


def run_single(args):
    i_neuron, i_optimiser, i_function, i_initialisation, i_repeat = args

    # Fetch the system parameters
    if NEURON_SYS_CACHE[i_neuron] is None:
        NEURON_SYS_CACHE[i_neuron] = sys = get_neuron_sys(NEURONS[i_neuron])
    else:
        sys = NEURON_SYS_CACHE[i_neuron]

    # Generate the test samples
    # Assemble the pre-populations
    np.random.seed(5781 * i_repeat + 5713)
    ens1 = Ensemble(51, 1)
    ens2 = Ensemble(52, 1)

    # Get the activities for training
    rng = np.random.RandomState(7897 * i_repeat + 78897)
    f = FUNCTIONS[i_function]
    xs1 = np.linspace(-1, 1, N_TRAIN_RES)
    xs2 = np.linspace(-1, 1, N_TRAIN_RES)
    xss1, xss2 = np.meshgrid(xs1, xs2)
    xss1 = xss1.flatten()
    xss2 = xss2.flatten()
    As1 = ens1(xss1.reshape(1, -1)).T
    As2 = ens2(xss2.reshape(1, -1)).T
    As_train = np.concatenate((As1, As2), axis=1)
    Js_train = f(xss1, xss2) * 1e-9

    # Get the activities for testing
    xss1, xss2 = rng.uniform(-1, 1, (2, N_TEST))
    As1 = ens1(xss1.reshape(1, -1)).T
    As2 = ens2(xss2.reshape(1, -1)).T
    As_test = np.concatenate((As1, As2), axis=1)
    Js_test = f(xss1, xss2) * 1e-9

    # Generate the initial weights W0
    init = INITIALISATIONS[i_initialisation]
    if init == "random":
        W0 = rng.uniform(0.0, 1.0e-8, (sys.n_inputs, As_train.shape[-1]))
    elif init == "zero_point":
        W0 = np.zeros((sys.n_inputs, As_train.shape[-1]))
        W0, _ = weight_opt.optimise_trust_region(
            sys,
            As_train,
            Js_train * 0.0,
            W=W0,
            N_epochs=1,
            reg1=1e-1,
            progress=False,
            normalise_error=False,
            parallel_compile=False)
    else:
        raise RuntimeError("Invalid initialisation")

    # Optimise the weights
    opt = OPTIMISERS[i_optimiser]
    kwargs = dict(
            reduced_sys=sys,
            As_train=As_train,
            Js_train=Js_train,
            J_th=0.0,
            W=W0,
            As_test=As_test,
            Js_test=Js_test,
            progress=False,
    )
    if opt == "trust_region":
        _, errs_train, errs_test = weight_opt.optimise_trust_region(
            **kwargs,
            N_epochs=N_EPOCHS_TR,
            parallel_compile=False)
    elif opt == "lbfgsb":
        _, errs_train, errs_test = weight_opt.optimise_bfgs(
            **kwargs,
            N_epochs=N_EPOCHS_GRADIENT)
    elif opt == "adam":
        _, errs_train, errs_test = weight_opt.optimise_sgd(
            **kwargs,
            N_epochs=N_EPOCHS_GRADIENT)
    else:
        raise RuntimeError("Invalid optimiser")

    E = np.ones((2, N_EPOCHS + 1)) * np.nan
    E[0, :len(errs_train)] = errs_train
    E[1, :len(errs_test)] = errs_test

    return i_neuron, i_optimiser, i_function, i_initialisation, i_repeat, E


def main():
    # Fill the parameter array
    params = [(i, j, k, l, m) for i in range(N_NEURONS)
              for j in range(N_OPTIMISERS) for k in range(N_FUNCTIONS)
              for l in range(N_INITIALISATIONS) for m in range(N_REPEAT)]
    random.shuffle(params)

    fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
                      'nlif_weight_optimisation_comparison.h5')
    with h5py.File(fn, 'w') as f:
        errs = f.create_dataset('errs',
                                (N_NEURONS, N_OPTIMISERS, N_FUNCTIONS,
                                 N_INITIALISATIONS, N_REPEAT, 2, N_EPOCHS + 1))

        with env_guard.SingleThreadEnvGuard():
            with multiprocessing.get_context('spawn').Pool() as pool:
                for i, j, k, l, m, E in tqdm.tqdm(pool.imap_unordered(
                        run_single, params),
                                                  total=len(params)):
                    errs[i, j, k, l, m] = E


if __name__ == "__main__":
    main()

