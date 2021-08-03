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

import copy

import numpy as np
import scipy.linalg

from .internal import magic
from .internal import lif_utils
from . import simulator


def pluck(dict, *keys):
    """
    Use this function to destructure a dictionary. See

    https://stackoverflow.com/a/17074606
    """
    return tuple((dict[key] for key in keys))


class System:
    """
    This class is used internally by AssembledNeuron to hold all the matrices
    that describe the neuron.
    """
    def __init__(self, n_compartments, n_inputs):
        self._n_compartments = n_compartments
        self._n_inputs = n_inputs

        n, l = n_compartments, n_inputs

        # Standard matrices describing the dynamics
        self._A = np.zeros((n, l))
        self._a_const = np.zeros(n)
        self._B = np.zeros((n, l))
        self._b_const = np.zeros((n, ))

        # Masks determining which entries in A and B are parameters that can be
        # tuned.
        self._A_mask = np.zeros((n, l), dtype=bool)
        self._a_const_mask = np.ones((n, ), dtype=bool)
        self._B_mask = np.zeros((n, l), dtype=bool)
        self._b_const_mask = np.ones((n, ), dtype=bool)

        # Matrices describing neuron-intrinsic currents. This is usually just
        # the leak channel, but could also contain things such as the bias
        # current
        self._a_const_intr = np.zeros((n, ))
        self._b_const_intr = np.zeros((n, ))

        # Graph Lagrangian
        self._L = np.zeros((n, n))

        # Vector containing all membrane capacitances
        self._C_m = np.zeros((n, ))

    def A_dyn(self, x, exclude_intrinsic=False):
        a_const = np.copy(self._a_const)
        if exclude_intrinsic:
            a_const -= self._a_const_intr
        return -(self._L + np.diag(a_const + self._A @ x))

    def b_dyn(self, x, exclude_intrinsic=False):
        b_const = np.copy(self._b_const)
        if exclude_intrinsic:
            b_const -= self._b_const_intr
        return b_const + self._B @ x

    def v_eq(self, x, exclude_intrinsic=False):
        return np.linalg.solve(-self.A_dyn(x, exclude_intrinsic),
                               self.b_dyn(x, exclude_intrinsic))

    @property
    def A(self):
        return self._A

    @property
    def A_mask(self):
        return self._A_mask

    @property
    def a_const(self):
        return self._a_const

    @property
    def a_const_mask(self):
        return self._a_const_mask

    @property
    def B(self):
        return self._B

    @property
    def B_mask(self):
        return self._B_mask

    @property
    def b_const(self):
        return self._b_const

    @property
    def b_const_mask(self):
        return self._b_const_mask

    @property
    def a_const_intr(self):
        return self._a_const_intr

    @property
    def b_const_intr(self):
        return self._b_const_intr

    @property
    def L(self):
        return self._L

    @property
    def C_m(self):
        return self._C_m

    @property
    def n_inputs(self):
        return self._n_inputs

    @property
    def n_compartments(self):
        return self._n_compartments

    def reduced_system(self, exclude_intrinsic=True):
        """
        Creates a new "ReducedSystem" instance from this system instance.
        """
        res = ReducedSystem(self.n_compartments, self.n_inputs)

        # Copy all matrices accross
        res.A[...] = self._A * self._C_m[:, None]
        res.A_mask[...] = self._A_mask

        res.a_const[...] = self._a_const * self._C_m
        res.a_const_mask[...] = self._a_const_mask

        res.B[...] = self._B * self._C_m[:, None]
        res.B_mask[...] = self._B_mask

        res.b_const[...] = self._b_const * self._C_m
        res.b_const_mask[...] = self._b_const_mask

        res.L[...] = self._L * self._C_m[:, None]

        # Exclude the intrinsic currents if so desired
        if exclude_intrinsic:
            res.a_const[...] -= self._a_const_intr * self._C_m
            res.b_const[...] -= self._b_const_intr * self._C_m

        return res


class ReducedSystem:
    """
    This class is used internally by AssembledNeuron to hold all the matrices
    that describe the neuron.
    """
    def __init__(self, n_compartments, n_inputs):
        self._n_compartments = n_compartments
        self._n_inputs = n_inputs

        n, l = n_compartments, n_inputs

        # Standard matrices describing the dynamics
        self._A = np.zeros((n, l))
        self._a_const = np.zeros((n, ))
        self._B = np.zeros((n, l))
        self._b_const = np.zeros((n, ))
        self._L = np.zeros((n, n))
        self._c = np.zeros((n))
        self._v_som = np.zeros((n))

        self._A_mask = np.zeros((n, l), dtype=bool)
        self._a_const_mask = np.zeros((n, ), dtype=bool)
        self._B_mask = np.zeros((n, l), dtype=bool)
        self._b_const_mask = np.zeros((n, ), dtype=bool)

        self._in_scale = 1.0
        self._out_scale = 1.0

    def clamp(self, idx, v):
        """
        Returns a new ReducedSystem instance where the compartment with the
        given index has been clamped to the specified voltage.
        """
        # If the first compartment is clamped, update v_som
        if idx == 0:
            self._v_som = np.ones(self.n_compartments) * v

        # Clone this ReducedSystem instance
        res = copy.deepcopy(self)

        # Disconnect the compartment from the other compartments
        res._L[idx, :] = 0
        res._L[:, idx] = 0
        for i in range(self.n_compartments):
            if i == idx:
                continue
            res._L[i, i] += self._L[idx, i]

        # Add a static conductance-based channel for each other compartment,
        # record the channel conductances in the "c" vector
        res._c[idx] = 1
        for i in range(self.n_compartments):
            if i == idx:
                continue
            res._a_const[i] -= self._L[idx, i]
            res._b_const[i] -= self._L[idx, i] * v
            res._c[i] -= self._L[idx, i]

        # Convert all conductance-based channels into current-based channels
        # in the compartment that is being clamped.
        res._a_const[idx] = 1
        res._b_const[idx] -= self._a_const[idx] * v - v
        res._A[idx, :] = 0
        res._A_mask[idx, :] = False
        res._a_const_mask[idx] = False
        res._B[idx, :] -= self._A[idx, :] * v

        return res

    def A_dyn(self, x):
        return -(self._L + np.diag(self._a_const + self._A @ x))

    def b_dyn(self, x):
        return self._b_const + self._B @ x

    def v_eq(self, x):
        return np.linalg.solve(-self.A_dyn(x), self.b_dyn(x))

    def i_som(self, x):
        return np.inner(self._c, self.v_eq(x) - self._v_som)

    def transform_voltage(self, v_offs=None, v_scale=None):
        """
        Rescales the underlying system such that all voltages are offset by
        v_offs and then scaled by v_scale, i.e.,
            v' = (v + v_offs) * v_scale
        """

        # Default v_offs and v_scale to reasonable values
        if v_offs is None:
            v_offs = np.zeros(self.n_compartments)
        if v_scale is None:
            v_scale = np.ones(self.n_compartments)

        # Coerce v_offs
        v_offs = np.asarray(v_offs)
        if v_offs.size == 1:
            v_offs = np.ones(self.n_compartments) * v_offs
        if v_offs.shape != (self.n_compartments, ):
            raise RuntimeError(
                f"v_offs must either be a scalar or an array with {self.n_compartments} entries"
            )

        # Coerce v_scale
        v_scale = np.asarray(v_scale)
        if v_scale.size == 1:
            v_scale = np.ones(self.n_compartments) * v_scale
        if v_scale.shape != (self.n_compartments, ):
            raise RuntimeError(
                f"v_scale must either be a scalar or an array with {self.n_compartments} entries"
            )

        # Copy this object
        res = copy.deepcopy(self)

        # Apply the offset
        res._b_const = self._b_const + (self._L +
                                        np.diag(self._a_const)) @ v_offs
        res._B = self._B + self._A * v_offs[:, None]
        res._v_som += v_offs

        # Apply the scaling factor to the A-side
        S = np.diag(1.0 / v_scale)
        res._L = self._L @ S
        res._a_const = S @ self._a_const
        res._A = S @ self._A
        res._v_som *= v_scale
        res._c /= v_scale

        return res

    def condition(self, in_scale=1e6, out_scale=1e9):
        # Offset all voltages such that v_som = 0 and c = 1 (where possible)
        v_scale = np.ones(self.n_compartments)
        for i in range(self.n_compartments):
            # If this compartment directly contributes to the somatic current,
            # scale the voltage of this compartment such that c[i] = 1.
            # Otherwise, try to make the corresponding a_const one.
            if self._c[i] > 1e-12:
                v_scale[i] = self._c[i]
            elif self._a_const[i] > 1e-12:
                v_scale[i] = self._a_const[i]
        res = self.transform_voltage(v_offs=-self._v_som, v_scale=v_scale)

        # Now apply the given scaling factors
        res._A *= 1.0 / in_scale
        res._B *= out_scale / in_scale
        res._b_const *= out_scale
        res._in_scale *= in_scale
        res._out_scale *= out_scale

        return res

    @property
    def A(self):
        return self._A

    @property
    def A_mask(self):
        return self._A_mask

    @property
    def a_const(self):
        return self._a_const

    @property
    def a_const_mask(self):
        return self._a_const_mask

    @property
    def B(self):
        return self._B

    @property
    def B_mask(self):
        return self._B_mask

    @property
    def b_const(self):
        return self._b_const

    @property
    def b_const_mask(self):
        return self._b_const_mask

    @property
    def L(self):
        return self._L

    @property
    def c(self):
        return self._c

    @property
    def in_scale(self):
        return self._in_scale

    @property
    def out_scale(self):
        return self._out_scale

    @property
    def v_som(self):
        return self._v_som

    @property
    def n_inputs(self):
        return self._n_inputs

    @property
    def n_compartments(self):
        return self._n_compartments


def assemble(compartments, connections, channels):
    """
    The "assemble" function takes the graph describing the neuron and generates
    the corresponding system matrices.
    """

    # Fetch the number of compartments
    n = len(compartments)

    # Fetch all input channels and create a mapping from channel onto input
    # index channel
    input_channel_map, input_compartment_map = {}, {}
    l = 0
    for i in range(n):
        for channel in channels[i]:
            if channel.is_input:
                input_channel_map[channel] = l
                input_compartment_map[l] = i
                l += 1

    # Create the target system
    sys = System(n, l)

    # Assemble the graph Laplacian
    for (i, j), conns in connections.items():
        for conn in conns:
            # Diagonal elements
            sys.L[i, i] += conn.g_c
            sys.L[j, j] += conn.g_c

            # Off-diagonal elements
            sys.L[i, j] -= conn.g_c
            sys.L[j, i] -= conn.g_c

    # Iterate over all compartments
    idx_input = 0
    for i in range(n):
        for j, chan in enumerate(channels[i]):
            if chan.is_input:
                if chan.type == "cond":
                    sys.A[i, idx_input] = 1
                    sys.A_mask[i, idx_input] = True
                    sys.B[i, idx_input] = chan.E_rev
                    sys.B_mask[i, idx_input] = True
                elif chan.type == "cur":
                    sys.B[i, idx_input] = chan.mul
                    sys.B_mask[i, idx_input] = True
                idx_input += 1
            else:
                tars = ([(sys.a_const, sys.b_const)] +
                        ([(sys.a_const_intr,
                           sys.b_const_intr)] if chan.is_intrinsic else []))
                for a_tar, b_tar in tars:
                    if chan.type == "cond":
                        a_tar[i] += chan.g
                        b_tar[i] += chan.g * chan.E_rev
                    elif chan.type == "cur":
                        b_tar[i] += chan.J * chan.mul

    # Divide all matrices by the membrane capacitances
    for i in range(n):
        sys.C_m[i] = compartments[i].C_m
        for mat in (sys.L, sys.A, sys.a_const, sys.a_const_intr, sys.B,
                    sys.b_const, sys.b_const_intr):
            mat[i] /= sys.C_m[i]

    return sys, input_channel_map, input_compartment_map


class AssembledNeuron:
    def __init__(self, neuron):
        # Copy the given neuron and coerce it
        self._obj_map = {}
        self._neuron = copy.deepcopy(neuron, memo=self._obj_map).coerce()

        # Fetch the graph describing the neuron
        compartments, connections, channels = self.neuron.graph()
        self._compartments = compartments
        self._connections = connections
        self._channels = channels

        # Assemble the system
        self._sys, self._input_channel_map, self._input_compartment_map = assemble(
            compartments, connections, channels)

    def _canonicalise_input_array(self, xs, reshape=True):
        """
        Takes an input array xs and ensures that the last dimension is the
        number of inputs and that the first dimension is the number of samples.
        Returns the original shape of the array without the number of input
        dimensions.
        """
        if xs is None:
            xs = np.zeros(self.n_inputs)
        if isinstance(xs, dict):
            xs_in = xs
            shape_orig = (0, self.n_inputs)

            # Try to find a reference array
            arr_ref = None
            for arr in xs_in.values():
                if (arr_ref is None) or (arr_ref.size <= 1):
                    arr_ref = np.asarray(arr)

            # If a reference array has been found, create the actual input array
            if not arr_ref is None:
                shape_orig = (*arr_ref.shape, self.n_inputs)
                xs = np.zeros(shape_orig, dtype=arr_ref.dtype)
                for key, arr in xs_in.items():
                    if isinstance(key, int):
                        idx = key
                    else:
                        if id(key) in self._obj_map:
                            key = self._obj_map[id(key)]
                        if not key in self._input_channel_map:
                            raise RuntimeError(
                                f"Object {key} does not refer to an input channel"
                            )
                        idx = self._input_channel_map[key]
                    xs[..., idx] = arr
        else:
            xs = np.atleast_1d(xs)
            shape_orig = xs.shape

        assert xs.shape[-1] == self.n_inputs

        if reshape:
            return xs.reshape(-1, self.n_inputs), shape_orig[:-1]
        else:
            return xs

    def canonicalise_input(self, xs):
        return self._canonicalise_input_array(xs, reshape=False)

    @property
    def compartments(self):
        return self._compartments

    @property
    def connections(self):
        return self._connections

    @property
    def channels(self):
        return self._channels

    @property
    def neuron(self):
        return self._neuron

    @property
    def n_compartments(self):
        return self._sys.n_compartments

    @property
    def n_inputs(self):
        return self._sys.n_inputs

    @property
    def soma(self):
        return self._compartments[0]

    @property
    def system(self):
        return self._sys

    def _apply_to_input_array(self,
                              xs,
                              f,
                              element_shape,
                              parallel=False,
                              progress=False):
        # Make sure that the input array has the correct shape
        xs, output_shape = self._canonicalise_input_array(xs)

        # Allocate the target array
        res = np.zeros((xs.shape[0], *element_shape))

        def loop(i):
            res[i] = f(xs[i])

        def run(map_fun):
            # Fetch the numbr of tasks
            idcs = list(range(res.shape[0]))

            # Wrap the map function in the tqdm progress bar and shuffle the
            # list of indices. This makes for a better estimation of progress
            if progress:
                import random
                import tqdm
                random.shuffle(idcs)
                iter_ = tqdm.tqdm(map_fun(loop, idcs), total=len(idcs))
            else:
                iter_ = map_fun(loop, idcs)

            # Actually run the loop!
            for _ in iter_:
                pass

        # Either process the data in parallel or serially
        if parallel:
            import multiprocessing.dummy as mp  # Uses threads instead of processes
            with mp.Pool() as pool:
                run(pool.imap_unordered)
        else:
            run(map)

        # Prevent reshape from failing when there was only a single input
        if len(output_shape) + len(element_shape) == 0:
            return res
        else:
            return res.reshape(*output_shape, *element_shape)

    def lif_parameters(self):
        """
        Returns the parameters of the somatic compartment, that acts like a LIF
        neuron.
        """

        # Compute the leak conductance and leak potential
        gLs = []
        ELs = []
        for channel in self.channels[0]:
            if channel.type == "cond" and channel.is_intrinsic:
                gLs.append(channel.g)
                ELs.append(channel.E_rev)
        if (len(gLs) == 0) or (len(ELs) == 0):
            raise RuntimeError(
                "Somatic compartment does not have a leak channel")
        gL = sum(gLs)
        EL = sum((gL * EL for gL, EL in zip(gLs, ELs))) / gL

        # Fetch all the other parameters
        soma = self.soma
        tau_rc = soma.C_m / gL
        v_th = soma.v_th
        v_reset = soma.v_reset
        v_som = (v_reset + v_th) / 2
        tau_ref = soma.tau_ref + soma.tau_spike

        return {
            'g_L': gL,
            'E_L': EL,
            'v_th': v_th,
            'v_reset': v_reset,
            'v_som': v_som,
            'tau_rc': tau_rc,
            'tau_ref': tau_ref,
            'C_m': soma.C_m
        }

    def A(self, xs, exclude_intrinsic=False):
        """
        Computes the A-matrix for the given inputs xs. The last dimension of xs
        must be equal to the number of input channels.
        """
        A_single = lambda x: self.system.A_dyn(x, exclude_intrinsic)
        return self._apply_to_input_array(
            xs, A_single, (self.n_compartments, self.n_compartments))

    def b(self, xs, exclude_intrinsic=False):
        """
        Computes the A-matrix for the given inputs xs. The last dimension of xs
        must be equal to the number of input channels.
        """
        b_single = lambda x: self.system.b_dyn(x, exclude_intrinsic)
        return self._apply_to_input_array(xs, b_single,
                                          (self.n_compartments, ))

    def v_eq(self, xs=None):
        """
        Computes the equilibrium potential of the neuron. The last dimension of
        xs must be equal to the number of input channels.
        """
        v_eq_single = lambda x: self.system.v_eq(x)
        return self._apply_to_input_array(xs, v_eq_single,
                                          (self.n_compartments, ))

    def i_som(self, xs=None, reduced_system=None, exclude_intrinsic=True):
        if reduced_system is None:
            reduced_system = self.reduced_system(
                exclude_intrinsic=exclude_intrinsic)

        return self._apply_to_input_array(xs, reduced_system.i_som, tuple())

    def rate(self, xs, reduce_system=None, exclude_intrinsic=True):
        params = self.lif_parameters()
        i_som = self.i_som(xs, reduce_system, exclude_intrinsic)
        return lif_utils.lif_detailed_rate(i_som,
                                           v_th=params["v_th"],
                                           v_reset=params["v_reset"],
                                           gL=params["g_L"],
                                           Cm=params["C_m"],
                                           EL=params["E_L"],
                                           tau_ref=params["tau_ref"])

    def _mk_poisson_sources(self, xs, noise, tau, rate, rng=np.random):
        return [
            simulator.PoissonSource(rng.randint(0, 0x7FFFFF),
                                    0 if not noise else rate[i], 0.0,
                                    0.0 if not noise else 2.0 * x, tau[i],
                                    x if not noise else 0.0)
            for i, x in enumerate(xs)
        ]

    def _mk_poisson_matrix_source(self, xs, weights, tau, rates):
        # Set all weights smaller than one millionth of the max weight to zero
        def filter_rates_weights(rs, ws):
            th = max(1e-24, np.max(np.abs(ws)) * 1e-6)
            return np.array(
                list(
                    filter(lambda x: x[0] > 0 and np.abs(x[1]) >= th,
                           zip(rs, ws)))).T.reshape(2, -1)

        # Create the PoissonMatrixSource instances
        res = []
        for i in range(weights.shape[0]):
            rs, ws = filter_rates_weights(rates, weights[i])
            source = simulator.PoissonMatrixSource.create(tau=tau[i],
                                                          rates=rs,
                                                          weights=ws,
                                                          gain=xs[i] /
                                                          (rs @ ws))
            res.append(source)
        return res

    def _simulate_noisy(self, sim, n, x, noise, tau, rate, rng=np.random):
        # Make sure "tau" and "rate" are arrays
        if not hasattr(tau, '__len__'):
            tau = tau * np.ones(len(x))
        if not hasattr(rate, '__len__'):
            rate = rate * np.ones(len(x))

        # If "noise" is a boolean flag, just simulate a poisson source
        if isinstance(noise, bool):
            sources = self._mk_poisson_sources(x, noise, tau, rate, rng=rng)
            return sim.simulate_poisson(sources, n)
        elif isinstance(noise, np.ndarray):
            # If "noise" is a matrix, it corresponds to a neural weight vector
            # for each input and "x" contains a set of pre-synaptic rates.
            assert noise.ndim == 2 and rate.ndim == 1, "\"noise\" must be a 2D weight matrix and rate must be a vector of pre-synaptic activities"
            assert noise.shape[0] == x.shape[0]
            assert noise.shape[1] == rate.shape[0], "incompatible dimensions"
            sources = self._mk_poisson_matrix_source(x, noise, tau, rate)
            return sim.simulate_poisson_matrix_source(sources, n)
        elif isinstance(noise, str):
            # Otherwise "noise" is a file containing a noise profile. A noise
            # profile describes the random distributions for the inter-spike
            # intervals of a pre-neuron
            import h5py
            arr_qs_isi, arr_qs_ws, qs_isi, qs_ws, filters = [], [], [], [], []
            with h5py.File(noise, 'r') as f:
                for i, x in enumerate(x):
                    # Load the corresponding datasets
                    gain = f['gain_{}'.format(i)][()]
                    arr_qs_ws.append(
                        np.array(f['qs_ws_{}'.format(i)],
                                 dtype=np.float64,
                                 order='C'))
                    arr_qs_isi.append(
                        np.array(f['qs_isi_{}'.format(i)],
                                 dtype=np.float64,
                                 order='C'))

                    # Create the SampledRandomDistribution objects
                    qs_ws.append(
                        simulator.SampledRandomDistribution.create(
                            arr_qs_ws[-1],
                            seed=np.random.randint(0, 0x7FFFFF),
                            trafo=simulator.TRAFO_EXP10))
                    qs_isi.append(
                        simulator.SampledRandomDistribution.create(
                            arr_qs_isi[-1],
                            seed=np.random.randint(0, 0x7FFFFF),
                            trafo=simulator.TRAFO_EXP10_INV))

                    # Create the filter objects
                    filters.append(
                        simulator.ExponentialFilter(
                            tau[i],  # tau
                            0.0,  # offs
                            x * gain  # gain
                        ))
            return sim.simulate_noise_profile(qs_ws, qs_isi, filters, n)

    def i_som_empirical(self,
                        xs=None,
                        T=1.0,
                        dt=1e-4,
                        noise=False,
                        tau=5e-3,
                        rate=1000,
                        seed=None,
                        progress=True,
                        include_refrac=False,
                        rng=np.random):
        # Compute the number of samples
        n = int(T / dt + 1e-9)

        # Construct the simulator
        with simulator.Simulator(self,
                                 dt=dt,
                                 record_out=False,
                                 record_isom=True,
                                 record_voltages=False,
                                 record_spike_times=False,
                                 record_in_refrac=not include_refrac) as sim:

            def iSom_empirical_single(x):
                res = self._simulate_noisy(sim, n, x, noise, tau, rate)
                if include_refrac:
                    iSom = np.mean(res.isom)
                else:
                    iSom = np.mean(res.isom[~res.in_refrac])
                return iSom

            return self._apply_to_input_array(xs,
                                              iSom_empirical_single,
                                              tuple(),
                                              progress=progress,
                                              parallel=True)

    def rate_empirical(self,
                       xs=None,
                       T=1.0,
                       dt=1e-4,
                       noise=False,
                       tau=5e-3,
                       rate=1000,
                       seed=None,
                       progress=True,
                       rng=np.random):
        # Compute the number of samples
        n = int(T / dt + 1e-9)

        # Construct the simulator
        with simulator.Simulator(self,
                                 dt=dt,
                                 record_out=False,
                                 record_isom=False,
                                 record_voltages=False,
                                 record_spike_times=True,
                                 record_in_refrac=False) as sim:

            def rate_empirical_single(x):
                res = self._simulate_noisy(sim, n, x, noise, tau, rate)
                return lif_utils.spike_frequency(res.times)

            return self._apply_to_input_array(xs,
                                              rate_empirical_single,
                                              tuple(),
                                              progress=progress,
                                              parallel=True)

    def isom_empirical_from_rate(self,
                                 xs=None,
                                 T=1.0,
                                 dt=1e-4,
                                 noise=False,
                                 tau=5e-3,
                                 rate=1000,
                                 seed=None,
                                 progress=True,
                                 rng=np.random):
        rates = self.rate_empirical(xs, T, dt, noise, tau, rate, seed,
                                    progress, rng)
        params = self.lif_parameters()
        return lif_utils.lif_detailed_rate_inv(rates,
                                               v_th=params["v_th"],
                                               v_reset=params["v_reset"],
                                               gL=params["g_L"],
                                               Cm=params["C_m"],
                                               EL=params["E_L"],
                                               tau_ref=params["tau_ref"])

    def impulse_response(self, xs=None, T=0.1, dt=1e-4, v0=None):
        # Canonicalise the inputs
        if xs is None:
            xs = np.copy(self.system.C_m)

        # Make sure that the initial membrane potentials, if given, have the
        # right shape
        if v0 is None:
            v0 = self.v_eq()
        else:
            v0 = np.asarray(v0)
        if v0.shape != (self.n_compartments, ):
            raise RuntimeError("")

        # Some convenient aliases
        n, k = self.n_compartments, self.n_inputs
        sys = self.system

        # Generate the time-steps
        ts = np.arange(0, T, dt)

        # Pre-compute the matrix exponentials
        eAt = np.zeros((len(ts), n + 1, n + 1))
        A = np.zeros((n + 1, n + 1))
        A[:n, :n] = self.A(np.zeros(k))
        A[:n, n] = sys.b_const
        for j, t in enumerate(ts):
            eAt[j] = scipy.linalg.expm(A * t)

        def impulse_response_single(x):
            res = np.zeros((len(ts), k, n))

            # Iterate over all inputs
            for i in range(self.n_inputs):
                # Assemble the input
                g = np.zeros(k)
                g[i] = x[i]

                # Fetch the target state
                v_eq = self.v_eq()

                # Assemble the input vector, add the impulse from the
                # conductance-based input
                b = sys.B @ g - (sys.A @ g) * v0
                b_ext = np.concatenate((-b + v_eq - v0, (0.0, )))

                # Compute the actual impulse response
                for j in range(len(ts)):
                    res[j, i] = v_eq - (eAt[j] @ b_ext)[:n]

            return res

        return ts, self._apply_to_input_array(xs, impulse_response_single,
                                              (len(ts), k, n))

    def reduced_system(self, exclude_intrinsic=True, v_som=None):
        if v_som is None:
            v_som = (self.soma.v_th + self.soma.v_reset) / 2
        return self.system.reduced_system(exclude_intrinsic).clamp(0, v_som)

    def to_dot(self):
        """
        Returns a string containing a diagram representation of the neuron in
        the graphviz "dot" format
        """
        import io
        from .internal.graphviz import generate_neuron_graph

        buf = io.StringIO()
        generate_neuron_graph(self._compartments, self._connections, buf)
        return buf.getvalue()

    def to_svg(self):
        """
        Returns a string containing a diagram representation of the neuron
        model graph.
        """
        # Run "neato" in a subprocess
        import re, subprocess
        res = subprocess.run(["neato", "-Tsvg"],
                             input=self.to_dot().encode("utf-8"),
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)

        # Handle errors
        if res.returncode != 0:
            raise Exception(str(res.stderr, "utf-8"))

        # Extract the SVG portion from the XML
        svg_match = re.compile(".*(<svg.*?</svg>).*", re.MULTILINE | re.DOTALL)
        return svg_match.match(str(res.stdout, "utf-8"))[1]

    def _repr_html_(self):
        """
        Returns an SVG diagram depicting the structure of this particular neuron
        model. The returned HTML string can directly be embedded into an HTML
        page. This function is automatically called by Jupyter notebook.
        """
        return self.to_svg()

