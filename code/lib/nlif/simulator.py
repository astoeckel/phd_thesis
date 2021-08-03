#  libnlif -- Multi-compartment LIF simulator and weight solver
#  Copyright (C) 2017-2021  Andreas Stöckel
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

from .internal.simulator_types import (TRAFO_NONE, TRAFO_EXP10,
                                       TRAFO_EXP10_INV, Result, PoissonSource,
                                       ExponentialFilter,
                                       SampledRandomDistribution,
                                       PoissonMatrixSource)
from .internal.simulator_cpp import SimulatorCpp
from .neuron import Neuron

import numpy as np


class Simulator:
    TRAFO_NONE = TRAFO_NONE
    TRAFO_EXP10 = TRAFO_EXP10
    TRAFO_EXP10_INV = TRAFO_EXP10_INV

    def __init__(self,
                 assembled_neuron,
                 dt=1e-4,
                 ss=1,
                 record_out=True,
                 record_isom=False,
                 record_voltages=False,
                 record_spike_times=False,
                 record_in_refrac=False,
                 debug=False,
                 parallel_compile=False):
        # Make sure that the given "assembled_neuron" is actually an assembled
        # neuron
        from .assembled_neuron import AssembledNeuron
        if isinstance(assembled_neuron, Neuron):
            assembled_neuron = assembled_neuron.assemble()

        # Check the arguments for validity
        if float(dt) <= 0.0:
            raise RuntimeError("dt must be strictly positive")
        if int(ss) < 1:
            raise RuntimeError("ss must be strictly positive")

        # Copy the given system and other arguments
        self._assembled_neuron = assembled_neuron
        self._dt = float(dt)
        self._ss = int(ss)
        self._record_out = bool(record_out)
        self._record_isom = bool(record_isom)
        self._record_voltages = bool(record_voltages)
        self._record_spike_times = bool(record_spike_times)
        self._record_in_refrac = bool(record_in_refrac)
        self._debug = bool(debug)
        self._parallel_compile = bool(parallel_compile)

        # Initialise the variable holding a reference at the backend C++ simulator
        self._sim = None

    def __enter__(self):
        self._sim = SimulatorCpp(assembled_neuron=self._assembled_neuron,
                                 dt=self._dt,
                                 ss=self._ss,
                                 record_out=self._record_out,
                                 record_isom=self._record_isom,
                                 record_voltages=self._record_voltages,
                                 record_spike_times=self._record_spike_times,
                                 record_in_refrac=self._record_in_refrac,
                                 debug=self._debug,
                                 parallel_compile=self._parallel_compile)
        return self

    def __exit__(self, type, value, traceback):
        self._sim = None

    def _check_sim(self):
        if self._sim is None:
            raise RuntimeError(
                "Simulator can only be used inside a \"with\" block")

    def simulate(self, xs, state=None):
        self._check_sim()
        return self._sim.simulate(xs, state)

    def simulate_poisson(self, sources, n_samples, state=None):
        return self._sim.simulate_poisson(sources, n_samples, state)

    def simulate_noise_profile(self,
                               qs_weights,
                               qs_isi,
                               filters,
                               n_samples,
                               state=None):
        self._check_sim()
        return self._sim.simulate_noise_profile(qs_weights, qs_isi, filters,
                                                n_samples, state)

    def simulate_filtered(self, xs, filters, state=None):
        self._check_sim()
        return self._sim.simulate_filtered(xs, filters, state)

    def simulate_poisson_matrix_source(self, sources, n_samples, state=None):
        self._check_sim()
        return self._sim.simulate_poisson_matrix_source(
            source, n_samples, state)

    @property
    def n_compartments(self):
        return self._assembled_neuron.n_compartments

    @property
    def n_inputs(self):
        return self._assembled_neuron.n_inputs

    @property
    def assembled_neuron(self):
        return self._assembled_neuron

    @property
    def system(self):
        return self._assembled_neuron.system

    @property
    def dt(self):
        return self._dt

    @property
    def ss(self):
        return self._ss

    @property
    def record_out(self):
        return self._record_out

    @property
    def record_isom(self):
        return self._record_isom

    @property
    def record_voltages(self):
        return self._record_voltages

    @property
    def record_spike_times(self):
        return self._record_spike_times

    @property
    def record_in_refrac(self):
        return self._record_in_refrac

