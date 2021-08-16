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

from gen_2d_fun import mk_2d_flt, gen_2d_fun

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
    def __init__(self, n_neurons, n_dimensions, neuron_type=LIF, rng=None):
        self.neuron_type = neuron_type

        # Randomly select the intercepts and the maximum rates
        self.intercepts = rng.uniform(-0.95, 0.95, n_neurons)
        self.max_rates = rng.uniform(0.5, 1.0, n_neurons)

        # Randomly select the encoders
        self.encoders = rng.normal(0, 1, (n_neurons, n_dimensions))
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

    return sys, assm


###########################################################################
# Parameters                                                              #
###########################################################################

NEURONS = [
    one_comp_lif_neuron,
    two_comp_lif_neuron,
    three_comp_lif_neuron,
    four_comp_lif_neuron,
]
N_NEURONS = len(NEURONS)
NEURON_SYS_CACHE = [None] * N_NEURONS

# Number of steps on the frequency axis
N_SIGMAS = 60
SIGMAS = np.logspace(np.log10(0.075), 1, N_SIGMAS)[::-1]

# Number of repetitions
N_REPEAT = 50

# Number of epochs to use for the trust-region
N_EPOCHS_TR = 30

N_TRAIN = 256
N_TEST_RES = 101  # Must be odd

N_PRE_NEURONS1 = 100
N_PRE_NEURONS2 = 101

###########################################################################
# Actual code                                                             #
###########################################################################


def run_single(args):
    i, j, k, partition, rms_correction = args

    if not i in NEURON_SYS_CACHE:
        NEURON_SYS_CACHE[i] = get_neuron_sys(NEURONS[i])
    sys, assm = NEURON_SYS_CACHE[i]
    sigma = SIGMAS[j]
    rng = np.random.RandomState(48919 * k + 758 * partition + 4156)

    # Assemble the target function
    flt = mk_2d_flt(sigma, N_TEST_RES)
    J_tar = 0.5 * gen_2d_fun(flt, N_TEST_RES, rng).flatten() / sys.out_scale
    J_tar = J_tar.flatten()

    # Assemble the pre-ensembles
    ens1 = Ensemble(N_PRE_NEURONS1, 1, rng=rng)
    ens2 = Ensemble(N_PRE_NEURONS2, 1, rng=rng)

    # Sample the target function
    xs1 = np.linspace(-1, 1, N_TEST_RES)
    xs2 = np.linspace(-1, 1, N_TEST_RES)
    xss1, xss2 = np.meshgrid(xs1, xs2)
    As_pre1 = ens1(xss1.reshape(1, -1)).T
    As_pre2 = ens2(xss2.reshape(1, -1)).T
    As_pre = np.concatenate((As_pre1, As_pre2), axis=1)

    # Select the training samples
    idx_train = rng.randint(0, N_TEST_RES * N_TEST_RES, N_TRAIN)
    J_train = J_tar[idx_train]
    As_train = As_pre[idx_train]

    rms = np.sqrt(np.mean(np.square(J_tar * sys.out_scale)))
    if rms_correction:
        return i, j, k, rms

    # Compute the weights
    W, errs = weight_opt.optimise_trust_region(sys,
                                               As_train,
                                               J_train,
                                               J_th=0.0,
                                               N_epochs=N_EPOCHS_TR,
                                               parallel_compile=False,
                                               progress=False)

    # Compute the test error
    As_pre_test = np.clip(As_pre + 0.01 * rng.randn(*As_pre.shape), 0.0, None)
    rmse = np.sqrt(
        np.mean(
            weight_opt.loss(sys,
                            As_pre_test,
                            J_tar * sys.out_scale,
                            W * sys.in_scale,
                            J_th=0.0)))

    E = rmse / rms

    return i, j, k, E


def main():
    import sys
    rms_correction = False
    if len(sys.argv) > 1:
        partition = int(sys.argv[1])
        if len(sys.argv) > 2:
            rms_correction = sys.argv[2] == "rms"
    else:
        partition = 0

    # Fill the parameter array
    params = [(i, j, k, partition, rms_correction) for i in range(N_NEURONS)
              for j in range(N_SIGMAS) for k in range(N_REPEAT)]
    random.shuffle(params)

    if not rms_correction:
        fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
                          f'nlif_frequency_sweep_{partition}.h5')
    else:
        fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
                          f'nlif_frequency_sweep_{partition}_rms.h5')
    with h5py.File(fn, 'w') as f:
        f.create_dataset('sigmas', data=SIGMAS)
        errs = f.create_dataset('errs', (N_NEURONS, N_SIGMAS, N_REPEAT))

        with env_guard.SingleThreadEnvGuard():
            with multiprocessing.get_context('spawn').Pool() as pool:
                for i, j, k, E in tqdm.tqdm(pool.imap_unordered(
                        run_single, params),
                                            total=len(params)):
                    errs[i, j, k] = E


if __name__ == "__main__":
    main()

