#!/usr/bin/env python3

import numpy as np
import os
import nlif
from nlif.parameter_optimisation import optimise_trust_region, optimise_sgd

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


def run_analysis(neuron,
                 chan1,
                 chan2,
                 c10=0e-6,
                 c11=1e-6,
                 c20=0e-6,
                 c21=1e-6,
                 res=100):
    levels_rate = np.linspace(0, 100, 11)

    assm = neuron.assemble()
    rng = np.random.RandomState(578281)
    gs_train = rng.uniform(0, 1e-6, (1000, assm.n_inputs))
    As_train = assm.rate_empirical(gs_train)
    valid_train = As_train > 12.5
    Js_train = assm.lif_rate_inv(As_train)

    sys = assm.reduced_system().condition()

    print("a_const =", sys.a_const)
    print("A =", sys.A)
    print("b_const =", sys.b_const)
    print("B =", sys.B)
    print("L =", sys.L)
    print("c =", sys.c)

    sys, _ = optimise_trust_region(sys,
                                   gs_train[valid_train],
                                   Js_train[valid_train],
                                   alpha3=1e-5,
                                   gamma=0.99,
                                   N_epochs=100,
                                   progress=True,
                                   parallel_compile=False)

    print("a_const =", sys.a_const)
    print("A =", sys.A)
    print("b_const =", sys.b_const)
    print("B =", sys.B)
    print("L =", sys.L)
    print("c =", sys.c)

    c1s = np.linspace(c10, c11, res)
    c2s = np.linspace(c20, c21, res)
    c1ss, c2ss = np.meshgrid(c1s, c2s)

    As = assm.rate_empirical({
        chan1: c1ss,
        chan2: c2ss,
    })

    Js_pred = assm.i_som({
        chan1: c1ss,
        chan2: c2ss,
    }, reduced_system=sys)
    As_pred = assm.lif_rate(Js_pred)

    return {
        "c10": c10,
        "c11": c11,
        "c20": c20,
        "c21": c21,
        "c1s": c1s,
        "c2s": c2s,
        "As": As,
        "As_pred": As_pred
    }


fn = lambda fn: os.path.join(os.path.dirname(__file__), '..', '..', '..',
                             'data', fn)

np.savez(
    fn("nlif_params_contour_two_comp_lif"),
    **run_analysis(two_comp_lif_neuron,
                   two_comp_lif_neuron.g_E,
                   two_comp_lif_neuron.g_I,
                   c10=0.045e-6,
                   c11=0.85e-6,
                   c21=0.75e-6))

np.savez(
    fn("nlif_params_contour_three_comp_lif_1"),
    **run_analysis(three_comp_lif_neuron,
                   three_comp_lif_neuron.g_E1,
                   three_comp_lif_neuron.g_I2,
                   c10=0.05e-6,
                   c11=0.175e-6,
                   c21=0.5e-6))

np.savez(
    fn("nlif_params_contour_three_comp_lif_2"),
    **run_analysis(three_comp_lif_neuron,
                   three_comp_lif_neuron.g_E2,
                   three_comp_lif_neuron.g_I2,
                   c10=0.1e-6,
                   c11=0.5e-6,
                   c21=0.3e-6))

np.savez(
    fn("nlif_params_contour_four_comp_lif_1"),
    **run_analysis(four_comp_lif_neuron,
                   four_comp_lif_neuron.g_E1,
                   four_comp_lif_neuron.g_I3,
                   c10=0.05e-6,
                   c11=0.2e-6,
                   c21=0.5e-6))

np.savez(
    fn("nlif_params_contour_four_comp_lif_2"),
    **run_analysis(four_comp_lif_neuron,
                   four_comp_lif_neuron.g_E2,
                   four_comp_lif_neuron.g_I3,
                   c10=0.12e-6,
                   c11=0.5e-6,
                   c21=0.5e-6))

np.savez(
    fn("nlif_params_contour_four_comp_lif_3"),
    **run_analysis(four_comp_lif_neuron,
                   four_comp_lif_neuron.g_E3,
                   four_comp_lif_neuron.g_I3,
                   c10=0.175e-6,
                   c11=1e-6,
                   c21=0.45e-6))

