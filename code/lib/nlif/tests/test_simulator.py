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

def test_simulator_basic(do_plot=False):
    if do_plot:
        import matplotlib.pyplot as plt

    E_L = -65e-3
    g_L = 50e-9
    E_E = 0e-3
    E_I = -80e-3
    c12 = 100e-9
    c23 = 100e-9
    C_m = 1e-9

    with nlif.Neuron() as lif:
        with nlif.Soma(C_m=C_m) as soma:
            gL = nlif.CondChan(E_rev=E_L, g=g_L)

        with nlif.Compartment(C_m=C_m) as basal:
            gL = nlif.CondChan(E_rev=E_L, g=g_L)
            gE1 = nlif.CondChan(E_rev=E_E)
            gI1 = nlif.CondChan(E_rev=E_I)

        with nlif.Compartment(C_m=C_m) as apical:
            gL = nlif.CondChan(E_rev=E_L, g=g_L)
            gE2 = nlif.CondChan(E_rev=E_E)
            gI2 = nlif.CondChan(E_rev=E_I)

        nlif.Connection(soma, basal, g_c=c12)
        nlif.Connection(basal, apical, g_c=c23)

    assm = lif.assemble()

    n_samples = 1000
    dt = 1e-4
    ss = 1
    state0 = np.array(([assm.soma.v_reset] * (assm.n_compartments + 1)) + [0.0])
    xs = {
        gE1: np.ones(n_samples) * 100e-9,
        gE2: 200e-9,
    }

    with nlif.Simulator(lif, dt=dt, ss=ss, record_voltages=True) as sim:
        res = sim.simulate(xs)

    if do_plot:
        fig, ax = plt.subplots()
        ax.plot(res.trange(), res.v, '-+')
        plt.show()

if __name__ == "__main__":
    test_simulator_basic(True)
