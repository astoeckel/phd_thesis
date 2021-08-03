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

import numpy as np
import pytest

import nlif


def allclose(x, y):
    np.testing.assert_allclose(x, y, rtol=1e-5)


def test_basic_properties():
    # Construct a basic LIF neuron with a current-based input channel
    with nlif.Neuron() as lif:
        with nlif.Soma() as soma:
            gL = nlif.ConductanceChannel(E_rev=-65e-3, g=50e-3)
            J = nlif.CurrentChannel()

    # Test __contains__
    assert soma in lif
    assert gL in lif
    assert J in lif
    assert J in soma

    # Test automagic label deduction
    assert lif.label == "lif"
    assert soma.label == "soma"
    assert gL.label == "gL"
    assert J.label == "J"

    # Test parent and compartment assignment
    assert soma.parent is lif
    assert gL.parent is lif
    assert J.parent is lif
    assert gL.compartment is soma
    assert J.compartment is soma

    # Test the "is_intrinsic_current" property
    assert not J.is_intrinsic
    assert gL.is_intrinsic


def test_graph_simple():
    with nlif.Neuron() as lif:
        with nlif.Soma() as soma:
            gL = nlif.ConductanceChannel(E_rev=-65e-3, g=50e-3)
            J = nlif.CurrentChannel()

    compartments, connections, channels = lif.graph()
    assert len(compartments) == 1
    assert compartments[0] is soma

    assert len(connections) == 0

    assert len(channels) == 1
    assert len(channels[0]) == 2
    assert channels[0][0] == gL
    assert channels[0][1] == J


def test_graph_compartments_sorting():
    with nlif.Neuron() as lif:
        with nlif.Compartment() as c1:
            pass

        with nlif.Soma() as soma:
            gL = nlif.ConductanceChannel(E_rev=-65e-3, g=50e-3)
            J = nlif.CurrentChannel()

        with nlif.Compartment() as c2:
            pass

        con1 = nlif.Connection(c1, soma)
        con2 = nlif.Connection(soma, c2)

    compartments, connections, channels = lif.graph()
    assert len(compartments) == 3
    assert compartments[0] is soma
    assert compartments[1] is c1
    assert compartments[2] is c2

    assert len(connections) == 2
    assert (0, 1) in connections
    assert (0, 2) in connections
    assert len(connections[(0, 1)]) == 1
    assert len(connections[(0, 2)]) == 1
    assert connections[(0, 1)][0] is con1
    assert connections[(0, 2)][0] is con2

    assert len(channels) == 3
    assert len(channels[0]) == 2
    assert len(channels[1]) == 0
    assert len(channels[2]) == 0

    assert channels[0][0] is gL
    assert channels[0][1] is J


def test_graph_error_no_compartments():
    with nlif.Neuron() as lif:
        pass

    # This should be fine
    lif.coerce()

    with pytest.raises(nlif.NoCompartmentsError) as _:
        lif.graph()


def test_graph_error_no_soma():
    with nlif.Neuron() as lif:
        with nlif.Compartment() as comp:
            pass

    # This should be fine
    lif.coerce()

    with pytest.raises(nlif.NoSomaticCompartmentError) as _:
        lif.graph()


def test_graph_error_multiple_somas():
    with nlif.Neuron() as lif:
        with nlif.Soma() as soma1:
            pass

        with nlif.Soma() as soma2:
            pass

    # This should be fine
    lif.coerce()

    with pytest.raises(nlif.MultipleSomaticCompartmentsError) as _:
        lif.graph()


def test_graph_error_connectivity():
    with nlif.Neuron() as lif1:
        with nlif.Soma() as soma1:
            pass

    with nlif.Neuron() as lif2:
        with nlif.Soma() as soma2:
            pass

        c2 = nlif.Compartment()
        nlif.Connection(soma1, c2)

    # This should be fine
    compartments, connections, channels = lif1.coerce().graph()
    assert len(compartments) == 1
    assert len(connections) == 0
    assert len(channels) == 1
    assert len(channels[0]) == 0

    with pytest.raises(nlif.ConnectivityError) as _:
        lif2.graph()


def test_graph_error_self_connection():
    with nlif.Neuron() as lif:
        with nlif.Soma() as soma:
            pass

        nlif.Connection(soma, soma)

    with pytest.raises(nlif.ValidationError) as _:
        lif.coerce()

    with pytest.raises(nlif.SelfConnectionError) as _:
        lif.graph()


def test_graph_error_channel_connectivity():
    with nlif.Neuron() as lif1:
        soma1 = nlif.Soma()

    with nlif.Neuron() as lif2:
        soma2 = nlif.Soma()
        chan = nlif.CurChan(compartment=soma1)

    # This should be fine
    lif1.coerce()
    lif2.coerce()

    with pytest.raises(nlif.ChannelConnectivityError) as _:
        lif1.graph()

    with pytest.raises(nlif.ChannelConnectivityError) as _:
        lif2.graph()


def test_graph_error_disconnected_1():
    with nlif.Neuron() as lif:
        soma = nlif.Soma()
        comp = nlif.Compartment()

    lif.coerce()
    with pytest.raises(nlif.DisconnectedNeuronError) as _:
        lif.graph()


def test_graph_error_disconnected_2():
    with nlif.Neuron() as lif:
        soma = nlif.Soma()
        comp1 = nlif.Compartment()
        comp2 = nlif.Compartment()
        nlif.Connection(comp1, comp2)

    lif.coerce()
    with pytest.raises(nlif.DisconnectedNeuronError) as _:
        lif.graph()


def test_error_variable_intrinsic_channel():
    # This should be fine (no intrinsic current)
    with nlif.Neuron() as lif:
        with nlif.Soma():
            chan = nlif.CurChan(intrinsic=False)
    lif.coerce()

    # This should be fine (intrinsic bias current)
    with nlif.Neuron() as lif:
        with nlif.Soma():
            chan = nlif.CurChan(J=1e-9, intrinsic=True)
    lif.coerce()

    # Having an input channel that is marked as intrinsic should not be fine
    with nlif.Neuron() as lif:
        with nlif.Soma():
            chan = nlif.CurChan(intrinsic=True)
    with pytest.raises(nlif.ValidationError) as _:
        lif.coerce()


def test_graph_multiple_connections():
    with nlif.Neuron() as lif:
        soma = nlif.Soma()
        comp1 = nlif.Compartment()
        comp2 = nlif.Compartment()
        con1a = nlif.Connection(comp1, comp2)
        con1b = nlif.Connection(comp1, comp2)
        con2 = nlif.Connection(soma, comp1)

    compartments, connections, channels = lif.coerce().graph()

    assert len(compartments) == 3
    assert compartments[0] is soma
    assert compartments[1] is comp1
    assert compartments[2] is comp2

    assert len(connections) == 2
    assert (0, 1) in connections
    assert (1, 2) in connections
    assert len(connections[(1, 2)]) == 2
    assert connections[(1, 2)][0] is con1a
    assert connections[(1, 2)][1] is con1b

    assert len(connections[(0, 1)]) == 1
    assert connections[(0, 1)][0] is con2

    assert len(channels) == 3
    assert len(channels[0]) == 0
    assert len(channels[1]) == 0
    assert len(channels[2]) == 0


def test_coerce_valid():
    with nlif.Neuron() as lif:
        with nlif.Soma() as soma:
            gL = nlif.ConductanceChannel(E_rev=-65e-3, g=50e-3)
            J = nlif.CurrentChannel()

    assert lif.coerce() is lif


def test_lif_params():
    C_m = 1e-9
    v_th = -50e-3
    v_reset = -65e-3
    v_som = (v_th + v_reset) / 2
    tau_ref = 3e-3
    tau_spike = 2e-3

    with nlif.Neuron() as neuron:
        with nlif.Soma(C_m=C_m,
                       v_th=v_th,
                       v_reset=v_reset,
                       tau_ref=tau_ref,
                       tau_spike=tau_spike):
            nlif.ConductanceChannel(E_rev=-80e-3, g=50e-9)
            nlif.ConductanceChannel(E_rev=-65e-3, g=100e-9)

    params = neuron.assemble().lif_parameters()
    allclose(params["g_L"], 150e-9)
    allclose(params["E_L"], -70e-3)
    allclose(params["v_th"], v_th)
    allclose(params["v_reset"], v_reset)
    allclose(params["v_som"], v_som)
    allclose(params["tau_rc"], C_m / 150e-9)
    allclose(params["tau_ref"], tau_ref + tau_spike)
    allclose(params["C_m"], C_m)


def test_instantiate_lif():
    neuron = nlif.LIF()
    neuron.assemble()


def test_instantiate_lif_cond():
    neuron = nlif.LIFCond()
    neuron.assemble()


def test_instantiate_two_comp_lif_cond():
    neuron = nlif.TwoCompLIFCond()
    neuron.assemble()


def test_instantiate_three_comp_lif_cond():
    neuron = nlif.ThreeCompLIFCond()
    neuron.assemble()

