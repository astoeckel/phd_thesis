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

import numpy as np
from ctypes import POINTER, Structure, c_double, c_uint32

c_double_p = POINTER(c_double)
c_uint32_p = POINTER(c_uint32)

TRAFO_NONE = 0
TRAFO_EXP10 = 1
TRAFO_EXP10_INV = 2


class Result:
    """
    The `Result` class holds the result of a single-neuron
    simulation, e.g., the state of the neuron and the recorded quantities.
    Note that only `state` and `out` are available per default. The other
    properties are only available if the corresponding `record_*` flag was
    specified in the call to the `simulator` function.

    Properties
    ----------

    state: neuron state after the simulation finished. This can be passed as
    "state" parameter to a subsequent call to the simulator.

    out: computed spike train expressed as an additive superposition of
    discretised delta functions. In other words, spikes are represented as
    entries with magnitude $1 / dt$ in the array, all other entries are
    zero.

    isom: recorded somatic current. The somatic current is that current,
    that if injected into a standard current-based LIF neuron, would create
    exactly the same output as the simulated neuron.

    v: recorded voltage trace. Rows in the matrix correspond to individual
    samples, columns to the individual neuron compartments.

    times: exact spike times computed down to a resolution of
    `spike_time_resolution` specified in the call to `simulator`. The
    length of this array corresponds to the number of spikes in the
    simulation period.

    in_refrac: boolean array indicating whether the neuron has been in its
    refractory period during this particular time-step.
    """
    def __init__(self, dt, ss, n_samples, state, out, isom, v, times, in_refrac):
        self.dt = dt
        self.ss = ss
        self.n_samples = n_samples
        self.state = state
        self.out = out
        self.isom = isom
        self.v = v
        self.times = np.atleast_1d(times)
        self.in_refrac = in_refrac

    def trange(self):
        return np.arange(0, self.n_samples) * (self.dt * self.ss)


class PoissonSource(Structure):
    _fields_ = [("seed", c_uint32), ("rate", c_double), ("gain_min", c_double),
                ("gain_max", c_double), ("tau", c_double), ("offs", c_double)]


class ExponentialFilter(Structure):
    _fields_ = [("tau", c_double), ("offs", c_double), ("gain", c_double)]

    @staticmethod
    def populate(struct, filter_):
        if isinstance(filter_, numbers.Number):
            struct.tau = filter_
            struct.offs = 0.0
            struct.gain = 1.0
        else:
            struct.tau = filter_.tau
            struct.offs = filter_.offs
            struct.gain = filter_.gain


class SampledRandomDistribution(Structure):
    _fields_ = [
        ("seed", c_uint32),
        ("n", c_uint32),
        ("tbl", c_double_p),
        ("trafo", c_uint32),
    ]

    @staticmethod
    def populate(struct, distr):
        struct.seed = distr.seed
        struct.n = distr.n
        struct.tbl = distr.tbl
        struct.trafo = distr.trafo

        if hasattr(distr, 'qs_'):
            setattr(struct, 'qs_', distr.qs_)

    @staticmethod
    def create(qs, trafo=TRAFO_NONE, seed=None):
        # Make sure qs has the correct format
        qs = np.array(qs, dtype=np.float64, order='C')

        struct = SampledRandomDistribution()
        struct.seed = np.random.randint((1 << 32) -
                                        1) if seed is None else seed
        struct.n = qs.size
        struct.tbl = qs.ctypes.data_as(c_double_p)
        struct.trafo = trafo

        # Make sure qs is not deleted
        setattr(struct, 'qs_', qs)

        return struct


class PoissonMatrixSource(Structure):
    _fields_ = [("seed", c_uint32), ("tau", c_double), ("offs", c_double),
                ("gain", c_double), ("n", c_uint32), ("rates", c_double_p),
                ("weights", c_double_p)]

    @staticmethod
    def populate(struct, source):
        struct.seed = source.seed
        struct.tau = source.tau
        struct.offs = source.offs
        struct.gain = source.gain
        struct.n = source.n
        struct.rates = source.rates
        struct.weights = source.weights

        # Make sure rates and weights are not deleted
        if hasattr(source, 'rates_'):
            setattr(struct, 'rates_', source.rates_)
        if hasattr(source, 'weights_'):
            setattr(struct, 'weights_', source.weights_)

    @staticmethod
    def create(tau, rates, weights, seed=None, offs=0.0, gain=1.0):
        # Make sure rates and weights have the correct format
        rates = np.array(rates, dtype=np.float64, order='C')
        weights = np.array(weights, dtype=np.float64, order='C')

        assert rates.size == weights.size
        struct = PoissonMatrixSource()
        struct.seed = np.random.randint((1 << 32) -
                                        1) if seed is None else seed
        struct.tau = tau
        struct.offs = offs
        struct.gain = gain
        struct.n = rates.size
        struct.rates = rates.ctypes.data_as(c_double_p)
        struct.weights = weights.ctypes.data_as(c_double_p)

        # Make sure rates and weights are not deleted
        setattr(struct, 'rates_', rates)
        setattr(struct, 'weights_', weights)

        return struct

