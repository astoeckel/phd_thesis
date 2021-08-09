#!/usr/bin/env python3

import os

import numpy as np
import h5py

import nlif
import nlif.parameter_optimisation as param_opt
import nlif.weight_optimisation as weight_opt

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
    one_comp_lif_neuron,
    two_comp_lif_neuron,
    three_comp_lif_neuron,
    four_comp_lif_neuron,
]


def run_single(neuron,
               res_train=10,
               res_test=100,
               f=lambda x, y: (0.5 * x * y + 0.5)):
    # Optimise the neuron itself
    assm = neuron.assemble()
    rng = np.random.RandomState(578281)
    gs_train = rng.uniform(0, 1e-6, (1000, assm.n_inputs))
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

    # Assemble the pre-populations
    np.random.seed(7897)
    ens1 = Ensemble(51, 1)
    ens2 = Ensemble(52, 1)

    # Get the activities for training
    xs1 = np.linspace(-1, 1, res_train)
    xs2 = np.linspace(-1, 1, res_train)

    xss1, xss2 = np.meshgrid(xs1, xs2)
    xss1 = xss1.flatten()
    xss2 = xss2.flatten()

    As1 = ens1(xss1.reshape(1, -1)).T
    As2 = ens2(xss2.reshape(1, -1)).T
    As_train = np.concatenate((As1, As2), axis=1)

    # Assemble the target function
    Js_train = f(xss1, xss2) * 1e-9

    # Solve for synaptic weights
    W, errs = weight_opt.optimise_trust_region(sys,
                                               As_train,
                                               Js_train,
                                               J_th=0.0)

    # Get the activities for plotting
    xs1 = np.linspace(-1, 1, res_test)
    xs2 = np.linspace(-1, 1, res_test)

    xss1, xss2 = np.meshgrid(xs1, xs2)
    xss1 = xss1.flatten()
    xss2 = xss2.flatten()

    As1 = ens1(xss1.reshape(1, -1)).T
    As2 = ens2(xss2.reshape(1, -1)).T
    As_test = np.concatenate((As1, As2), axis=1)

    Js_test_tar = f(xss1, xss2) * 1e-9
    Js_test_dec = assm.i_som(As_test @ W.T, reduced_system=sys)

    return {
        "W": W,
        "res_train": res_train,
        "res_test": res_test,
        "Js_train": Js_train,
        "As_train": As_train,
        "As_test": As_test,
        "Js_test_tar": Js_test_tar,
        "Js_test_dec": Js_test_dec,
        "xs1": xs1,
        "xs2": xs2,
    }


res = {}
for i, neuron in enumerate(NEURONS):
    for key, value in run_single(NEURONS[i]).items():
        res[f"n{i}_{key}"] = value

fn = lambda fn: os.path.join(os.path.dirname(__file__), '..', '..', '..',
                             'data', fn)

np.savez(fn("nlif_decode_multiplication_example"), **res)

