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

import pytest
import numpy as np

import nlif


def allclose(x, y):
    np.testing.assert_allclose(x, y, rtol=1e-5)


def test_multi_compartment_neuron():
    E_L = -65e-3
    g_L = 50e-9
    E_E = 10.0e-3
    E_I = -80e-3
    c12 = 100e-9
    c23 = 200e-9
    C_m = 1e-9

    with nlif.Neuron() as lif:
        with nlif.Soma(C_m=C_m) as soma:
            gL = nlif.CondChan(E_rev=E_L, g=g_L)

        with nlif.Compartment(C_m=C_m) as basal:
            gL = nlif.CondChan(E_rev=E_L, g=g_L)
            gE = nlif.CondChan(E_rev=E_E)
            gI = nlif.CondChan(E_rev=E_I)

        with nlif.Compartment(C_m=C_m) as apical:
            gL = nlif.CondChan(E_rev=E_L, g=g_L)
            gE = nlif.CondChan(E_rev=E_E)
            gI = nlif.CondChan(E_rev=E_I)

        nlif.Connection(soma, basal, g_c=c12)
        nlif.Connection(basal, apical, g_c=c23)

    asm = lif.assemble()

    # Make sure that the graph Laplacian is correct
    assert asm.system.L.shape == (3, 3)
    allclose(
        asm.system.L,
        np.array([[c12, -c12, 0], [-c12, c12 + c23, -c23], [0, -c23, c23]]) /
        C_m)

    assert asm.system.A.shape == (3, 4)
    allclose(asm.system.A,
             np.array([
                 [0, 0, 0, 0],
                 [1, 1, 0, 0],
                 [0, 0, 1, 1],
             ]) / C_m)
    assert asm.system.a_const.shape == (3, )
    allclose(asm.system.a_const, np.array([g_L, g_L, g_L]) / C_m)

    assert asm.system.A_mask.shape == (3, 4)
    allclose(asm.system.A_mask, np.array([
        [False, False, False, False],
        [True, True, False, False],
        [False, False, True, True],
    ]))

    assert asm.system.B.shape == (3, 4)
    allclose(
        asm.system.B,
        np.array([
            [0, 0, 0, 0],
            [E_E, E_I, 0, 0],
            [0, 0, E_E, E_I],
        ]) / C_m)
    assert asm.system.b_const.shape == (3, )
    allclose(asm.system.b_const,
             np.array([g_L * E_L, g_L * E_L, g_L * E_L]) / C_m)

    assert asm.system.B_mask.shape == (3, 4)
    allclose(asm.system.B_mask, np.array([
        [False, False, False, False],
        [True, True, False, False],
        [False, False, True, True],
    ]))

    assert asm.system.a_const_intr.shape == (3, )
    allclose(asm.system.a_const_intr, np.array([g_L, 0, 0]) / C_m)

    assert asm.system.b_const_intr.shape == (3, )
    allclose(asm.system.b_const_intr, np.array([g_L * E_L, 0, 0]) / C_m)


def test_A_and_b():
    E_L = -65.0e-3
    E_E = 0.0e-3
    g_L = 50.0e-9
    C_m = 1.0e-9
    g_C = 20.0e-9

    with nlif.Neuron() as lif:
        with nlif.Soma(C_m=C_m) as soma:
            gL = nlif.CondChan(E_rev=E_L, g=g_L)
            J = nlif.CurChan()

        with nlif.Compartment() as basal:
            gE = nlif.CondChan(E_rev=E_E)

        nlif.Connection(soma, basal, g_C)

    assm = lif.assemble()
    allclose(assm.A({
        J: 1e-9,
        gE: 10e-9
    }), np.array(((-70.0, 20.0), (20.0, -30.0))))

    allclose(assm.b({J: 1e-9, gE: 10e-9}), np.array((-2.25, 0.0)))

    allclose(assm.b({
        J: 1e-9,
        gE: 10e-9
    }, exclude_intrinsic=True), np.array((1.0, 0.0)))

    inp_J = 1e-9
    inp_gE = 2e-9

    A_actual = assm.A(np.array([inp_J, inp_gE]), exclude_intrinsic=True)
    A_expected = np.array([[-20, 20], [20, -inp_gE / C_m - 20]])
    allclose(A_actual, A_expected)

    b_actual = assm.b(np.array([inp_J, 0]), exclude_intrinsic=True)
    b_expected = np.array([1.0e9 * inp_J, 0.0])
    allclose(b_actual, b_expected)


def test_reduced_system_lif_cur():
    g_L = 50e-3
    E_L = -65e-3
    C_m = 1e-9
    v_th = -50e-3
    v_reset = -65e-3

    neuron = nlif.LIF(g_L=g_L, E_L=E_L, C_m=C_m, v_th=v_th, v_reset=v_reset)
    v_som = 0.5 * (v_th + v_reset)
    sys = neuron.assemble().reduced_system(exclude_intrinsic=False)
    allclose(sys.A, np.array([[0, 0]]))
    allclose(sys.A_mask, np.array([[False, False]]))
    allclose(sys.a_const, np.array([1]))
    allclose(sys.a_const_mask, np.array([False,]))
    allclose(sys.B, np.array([[1, -1]]))
    allclose(sys.B_mask, np.array([[True, True]]))
    allclose(sys.b_const, np.array([
        g_L * (E_L - v_som) + v_som,
    ]))
    allclose(sys.b_const_mask, np.array([True,]))
    allclose(sys.L, np.array([[0]]))
    allclose(sys.c, np.array([1]))

    allclose(sys.i_som(np.array((0.0, 0.0))), g_L * (E_L - v_som))
    allclose(sys.i_som(np.array((1e-9, 0.0))),
             1e-9 + g_L * (E_L - v_som))
    allclose(sys.i_som(np.array((1e-9, 1e-9))),
             g_L * (E_L - v_som))
    allclose(sys.i_som(np.array((0.0, 1e-9))),
             -1e-9 + g_L * (E_L - v_som))

    sys = neuron.assemble().reduced_system()
    allclose(sys.A, np.array([[0, 0]]))
    allclose(sys.a_const, np.array([1]))
    allclose(sys.B, np.array([[1, -1]]))
    allclose(sys.b_const, np.array([
        v_som,
    ]))
    allclose(sys.L, np.array([[0]]))
    allclose(sys.c, np.array([1]))

    allclose(sys.i_som(np.array((0.0, 0.0))), 0.0)
    allclose(sys.i_som(np.array((1e-9, 0.0))), 1e-9)
    allclose(sys.i_som(np.array((1e-9, 1e-9))), 0.0)
    allclose(sys.i_som(np.array((0.0, 1e-9))), -1e-9)


def test_reduced_system_lif_cond():
    g_L = 50e-3
    E_L = -65e-3
    C_m = 1e-9
    E_E = 10e-3
    E_I = -80e-3
    v_th = -50e-3
    v_reset = -65e-3

    neuron = nlif.LIFCond(g_L=g_L,
                          E_L=E_L,
                          E_E=E_E,
                          E_I=E_I,
                          C_m=C_m,
                          v_th=v_th,
                          v_reset=v_reset)

    v_som = 0.5 * (v_th + v_reset)

    sys = neuron.assemble().reduced_system(exclude_intrinsic=False)
    allclose(sys.A, np.array([[0, 0]]))
    allclose(sys.A_mask, np.array([[False, False]]))
    allclose(sys.a_const, np.array([1]))
    allclose(sys.a_const_mask, np.array([False,]))
    allclose(sys.B, np.array([[E_E - v_som, E_I - v_som]]))
    allclose(sys.B_mask, np.array([[True, True]]))
    allclose(sys.b_const, np.array([
        g_L * (E_L - v_som) + v_som,
    ]))
    allclose(sys.b_const_mask, np.array([True,]))
    allclose(sys.L, np.array([[0]]))
    allclose(sys.c, np.array([1]))

    allclose(sys.i_som(np.array((0.0, 0.0))), g_L * (E_L - v_som))
    allclose(sys.i_som(np.array((1e-9, 0.0))),
             1e-9 * (E_E - v_som) + g_L * (E_L - v_som))
    allclose(sys.i_som(np.array((1e-9, 1e-9))),
             1e-9 * (E_E - v_som) + 1e-9 * (E_I - v_som) + g_L * (E_L - v_som))
    allclose(sys.i_som(np.array((0.0, 1e-9))),
             1e-9 * (E_I - v_som) + g_L * (E_L - v_som))

    sys = neuron.assemble().reduced_system()
    allclose(sys.A, np.array([[0, 0]]))
    allclose(sys.a_const, np.array([1]))
    allclose(sys.B, np.array([[E_E - v_som, E_I - v_som]]))
    allclose(sys.b_const, np.array([
        v_som,
    ]))
    allclose(sys.L, np.array([[0]]))
    allclose(sys.c, np.array([1]))

    allclose(sys.i_som(np.array((0.0, 0.0))), 0.0)
    allclose(sys.i_som(np.array((1e-9, 0.0))),
             1e-9 * (E_E - v_som))
    allclose(sys.i_som(np.array((1e-9, 1e-9))),
             1e-9 * (E_E - v_som) + 1e-9 * (E_I - v_som))
    allclose(sys.i_som(np.array((0.0, 1e-9))),
             1e-9 * (E_I - v_som))


def test_reduced_system_two_comp_lif_cond():
    g_L = 50e-3
    E_L = -65e-3
    C_m = 1e-9
    E_E = 10e-3
    E_I = -80e-3
    c12 = 100e-9
    v_th = -50e-3
    v_reset = -65e-3

    neuron = nlif.TwoCompLIFCond(g_L=g_L,
                                 E_L=E_L,
                                 E_E=E_E,
                                 E_I=E_I,
                                 g_c=c12,
                                 C_m=C_m,
                                 v_th=v_th,
                                 v_reset=v_reset)

    v_som = 0.5 * (v_th + v_reset)

    sys = neuron.assemble().reduced_system(exclude_intrinsic=False)
    allclose(sys.A, np.array([[0, 0], [1, 1]]))
    allclose(sys.a_const, np.array([1, g_L + c12]))
    allclose(sys.B, np.array([[0, 0], [E_E, E_I]]))
    allclose(
        sys.b_const,
        np.array([
            g_L * (E_L - v_som) + v_som,
            g_L * E_L + c12 * v_som,
        ]))
    allclose(sys.L, np.array([[0, 0], [0, 0]]))
    allclose(sys.c, np.array([1, c12]))

    sys = neuron.assemble().reduced_system(exclude_intrinsic=True)
    allclose(sys.A, np.array([[0, 0], [1, 1]]))
    allclose(sys.a_const, np.array([1, g_L + c12]))
    allclose(sys.B, np.array([[0, 0], [E_E, E_I]]))
    allclose(sys.b_const, np.array([
        v_som,
        g_L * E_L + c12 * v_som,
    ]))
    allclose(sys.L, np.array([[0, 0], [0, 0]]))
    allclose(sys.c, np.array([1, c12]))


def test_reduced_system_three_comp_lif_cond():
    g_L = 50e-3
    E_L = -65e-3
    C_m = 1e-9
    E_E = 10e-3
    E_I = -80e-3
    c12 = 100e-9
    c23 = 200e-9
    v_th = -50e-3
    v_reset = -65e-3

    neuron = nlif.ThreeCompLIFCond(g_L=g_L,
                                   E_L=E_L,
                                   E_E=E_E,
                                   E_I=E_I,
                                   g_c1=c12,
                                   g_c2=c23,
                                   C_m=C_m,
                                   v_th=v_th,
                                   v_reset=v_reset)

    v_som = 0.5 * (v_th + v_reset)

    sys = neuron.assemble().reduced_system(exclude_intrinsic=False)
    allclose(sys.A, np.array([[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]]))
    allclose(sys.a_const, np.array([1, g_L + c12, g_L]))
    allclose(sys.B, np.array([[0, 0, 0, 0], [E_E, E_I, 0, 0], [0, 0, E_E,
                                                               E_I]]))
    allclose(
        sys.b_const,
        np.array(
            [g_L * (E_L - v_som) + v_som, g_L * E_L + c12 * v_som, g_L * E_L]))
    allclose(sys.L, np.array([[0, 0, 0], [0, c23, -c23], [0, -c23, c23]]))
    allclose(sys.c, np.array([1, c12, 0]))

    sys = neuron.assemble().reduced_system(exclude_intrinsic=True)
    allclose(sys.A, np.array([[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]]))
    allclose(sys.a_const, np.array([1, g_L + c12, g_L]))
    allclose(sys.B, np.array([[0, 0, 0, 0], [E_E, E_I, 0, 0], [0, 0, E_E,
                                                               E_I]]))
    allclose(sys.b_const, np.array([v_som, g_L * E_L + c12 * v_som,
                                    g_L * E_L]))
    allclose(sys.L, np.array([[0, 0, 0], [0, c23, -c23], [0, -c23, c23]]))
    allclose(sys.c, np.array([1, c12, 0]))


def test_veq():
    E_L = -65.0e-3
    E_E = 0.0e-3
    g_L = 50.0e-9
    C_m = 1.0e-9
    g_C = 20.0e-9

    with nlif.Neuron() as lif:
        with nlif.Soma(C_m=C_m) as soma:
            gL = nlif.CondChan(E_rev=E_L, g=g_L)
            J = nlif.CurChan()

        with nlif.Compartment() as basal:
            gL = nlif.CondChan(E_rev=E_L, g=g_L)
            gE = nlif.CondChan(E_rev=E_E)

        nlif.Connection(soma, basal, g_C)

    assm = lif.assemble()

    v_eq = assm.v_eq()
    allclose(v_eq, [E_L, E_L])

    v_eq = assm.v_eq([0.0, 10.0e-9])
    allclose(v_eq, [-0.0625, -0.05625])

    v_eq = assm.v_eq([[0.0, 0.0], [0.0, 10.0e-9], [1e-9, 0.0]])
    allclose(v_eq,
             [[-0.065, -0.065], [-0.0625, -0.05625], [-0.049444, -0.060556]])


def test_transform_voltage_offset():
    sys = nlif.ThreeCompLIFCond().assemble().reduced_system()
    g1, g2 = np.zeros(4), np.ones(4) * 10e-9
    v_eq_orig_1 = sys.v_eq(g1)
    v_eq_orig_2 = sys.v_eq(g2)
    i_som_orig_1 = sys.i_som(g1)
    i_som_orig_2 = sys.i_som(g2)

    v_offs = [10.0e-3, 20.0e-3, 30.0e-3]
    sys = sys.transform_voltage(v_offs=v_offs)
    v_eq_1 = sys.v_eq(g1)
    v_eq_2 = sys.v_eq(g2)
    allclose(v_eq_1 - v_offs, v_eq_orig_1)
    allclose(v_eq_2 - v_offs, v_eq_orig_2)

    i_som_1 = sys.i_som(g1)
    i_som_2 = sys.i_som(g2)
    allclose(i_som_orig_1, i_som_1)
    allclose(i_som_orig_2, i_som_2)


def test_transform_voltage_scale():
    sys = nlif.ThreeCompLIFCond().assemble().reduced_system()
    g1, g2 = np.zeros(4), np.ones(4) * 10e-9
    v_eq_orig_1 = sys.v_eq(g1)
    v_eq_orig_2 = sys.v_eq(g2)
    i_som_orig_1 = sys.i_som(g1)
    i_som_orig_2 = sys.i_som(g2)

    v_scale = [1.2, 0.5, 2.0]
    sys = sys.transform_voltage(v_scale=v_scale)
    v_eq_1 = sys.v_eq(g1)
    v_eq_2 = sys.v_eq(g2)
    allclose(v_eq_1 / v_scale, v_eq_orig_1)
    allclose(v_eq_2 / v_scale, v_eq_orig_2)

    i_som_1 = sys.i_som(g1)
    i_som_2 = sys.i_som(g2)
    allclose(i_som_orig_1, i_som_1)
    allclose(i_som_orig_2, i_som_2)

