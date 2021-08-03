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


import copy
import io
import itertools
import numpy as np
import re


class Channel:
    """
    Base class from which the individual Channel classes are derived.
    """

    def __init__(self, name=None, type_=""):
        self.name = name
        self.type = type_


class CondChan(Channel):
    """
    The ConductanceChannel class represents a conductance-based channel inside
    a neuron compartment. The channel can either be static, in which case it has
    a constant conductance, or an input channel, in which case the conductance
    is influenced by pre-synaptic spikes.
    """

    def __init__(self, Erev, g=None, somatic_current=None, name=None):
        """
        Constructor of the ConductanceChannel class.

        Parameters
        ==========

        Erev: float describing the reversal potential of the ConductanceChannel
        in millivolts.
        g: static conductance value. If "None" is specified as a conductance
        value, the channel will be treated as an input channel.
        somatic_current: flag that indicates whether this channel is included
        in the somatic input current calculation. Per default (if
        somatic_current is None), static channels (such as the leak channel)
        are not included in the somatic input current. Explicitly setting this
        flag to true or false allows to override this behaviour.
        name: name of the condcutance channel. The name of the channel must be
        unique in each compartment, but the same neuron may have channels with
        the same name. If "None" is given, the channel is not addressable by
        name.
        """
        super().__init__(name, "cond")
        self.Erev = Erev
        self.g = g
        self.somatic_current = somatic_current

    def is_static(self):
        """
        Returns true if this channel has a static conductance
        """
        return not self.g is None

    def is_inhibitory(self, v_th=-50e-3):
        """
        Returns true if this channel is inhibitory.
        """
        return self.Erev < v_th

    def is_somatic_current(self):
        """
        Returns a boolean flag indicating whether this channel should be counted
        as a somatic current.
        """
        explicit = not self.somatic_current is None
        return self.somatic_current if explicit else (self.g is None)


class CurChan(Channel):
    """
    The CurrentChannel class represent a current-based channel inside a neuron
    compartment. The channel can either be static, in which case it acts as a
    constant current source, or it can either an input channel, in which case
    the current injected into the soma is influenced by pre-synaptic spikes.
    Currents are multiplied by the given multiplicative factor.
    """

    def __init__(self, mul=1, J=None, somatic_current=None, name=None):
        """
        Constructor of the CurrentChannel class.

        Parameters
        ==========

        mul: multiplicative factor by which input currents -- or the static
        current -- are multiplied. A factor of one corresponds to an excitatory
        synaptic channel, a factor of minus one corresponds to an inhibitory
        synaptic channel.
        somatic_current: flag that indicates whether this channel is included
        in the somatic input current calculation. Per default (if
        somatic_current is None), static current based channels -- in contrast
        to static conductance based channels --- are included in the somatic
        current calculation. Explicitly setting this flag to true or false
        allows to override this behaviour.
        """
        super().__init__(name, "cur")
        self.mul = mul
        self.J = J
        self.somatic_current = somatic_current

    def is_static(self):
        """
        Returns true if this channel has a static input current.
        """
        return not self.J is None

    def is_inhibitory(self, v_th=-50e-3):
        """
        Returns true if this channel is inhibitory.
        """
        return self.mul < 0 if (self.J is None) else self.mul * self.J < 0

    def is_somatic_current(self):
        """
        Returns a boolean flag indicating whether this channel should be counted
        as a somatic current.
        """
        explicit = not self.somatic_current is None
        return self.somatic_current if explicit else True


class Compartment:
    """
    The Compartment class represents a single compartment in the multicompartment
    LIF neuron. Per default, a compartment mainly consists of a capacitance with
    capacity Cm.
    """

    def __init__(self,
                 Cm=1e-9,
                 v_th=None,
                 v_reset=None,
                 v_spike=None,
                 tau_ref=None,
                 tau_spike=None,
                 soma=False,
                 name=None):
        self.Cm = Cm
        self.v_th = v_th if (not v_th is None) else (-50e-3 if soma else None)
        self.v_reset = v_reset if (not v_reset is None) else (-65e-3 if soma
                                                              else None)
        self.v_spike = v_spike if (not v_spike is None) else (20e-3 if soma
                                                              else None)
        self.tau_ref = tau_ref if (not tau_ref is None) else (2e-3 if soma else
                                                              None)
        self.tau_spike = tau_spike if (not tau_spike is None) else (1e-3
                                                                    if soma
                                                                    else None)
        self.soma = soma
        self.name = name
        self.channels = []
        self.connections = []

    def add_channel(self, channel=None):
        self.channels.append(channel)
        return self

    def connect(self, compartment_tar, gC):
        self.connections.append((compartment_tar, gC))
        return self


class Neuron:
    def __init__(self, name=None):
        self.name = name
        self.compartments = []
        self.connections = set()

    def add_compartment(self, compartment):
        self.compartments.append(compartment)
        return self

    def connect(self, compartment_src, compartment_tar, gC):
        self.connections.add((compartment_src, compartment_tar, gC))
        return self

    def assemble(self, symbolic=False):
        # Lookup function allowing to lookup compartments by name
        def _lookup(compartment):
            if not isinstance(compartment, Compartment):
                if compartment in compartment_names:
                    return compartment_names[compartment]
                else:
                    raise Exception(
                        "Compartment \"" + str(compartment) +
                        "\" is neither a compartment object nor a compartment specifier"
                    )
            return compartment

        # Function used for generating unique names with a common prefix
        def _make_unique_name(nameset, prefix, idx):
            while (prefix + str(idx)) in nameset:
                idx += 1
            return prefix + str(idx)

        # Import sympy if we're trying to do this symbolically
        if symbolic:
            import sympy as sp

        # Index all compartments by names, automatically add "soma" as a
        # name compartment
        channel_names = {}
        compartment_names = {}
        for compartment in self.compartments:
            if compartment.soma:
                if not compartment.name is None and compartment.name != "soma":
                    raise Exception(
                        "Somatic compartment must be named \"soma\", but got name \""
                        + compartment.name + "\" instead")
                if any(
                        map(lambda x: x is None, [
                            compartment.v_th, compartment.v_reset,
                            compartment.tau_ref
                        ])):
                    raise Exception(
                        "v_th, v_reset, or tau_ref not specified for somatic compartment"
                    )
                compartment.name = "soma"
            elif not all(
                    map(lambda x: x is None, [
                        compartment.v_th, compartment.v_reset,
                        compartment.tau_ref
                    ])):
                raise Exception(
                    "Must not specify v_th, v_reset, or tau_ref for non-somatic compartment"
                )

            if isinstance(compartment.name, str):
                if compartment.name in compartment_names:
                    raise Exception("Compartment with name \"" + compartment.
                                    name + "\" specified multiple times")
                if compartment.name == "soma" and not compartment.soma:
                    raise Exception(
                        "Compartment name \"soma\" is reserved for the somatic compartment"
                    )
                compartment_names[compartment.name] = compartment

        # Make sure there is a somatic compartment
        if not "soma" in compartment_names:
            raise Exception("Neuron requires exactly one somatic compartment.")

        # Assign an index to all compartments, make sure index zero is reserved
        # for the somatic compartment
        compartment_idcs = dict({compartment_names["soma"]: 0})
        for compartment in self.compartments:
            if not compartment in compartment_idcs:
                compartment_idcs[compartment] = len(compartment_idcs)

        # Gather all connections from the compartments
        connections = copy.copy(self.connections)  # shallow copy
        for compartment_src in self.compartments:
            for compartment_tar, gC in compartment_src.connections:
                connections.add((compartment_src, compartment_tar, gC))

        # Canonicalise the connection objects
        canonicalise = lambda c: (_lookup(c[0]), _lookup(c[1]), c[2])
        connections = list(map(canonicalise, connections))

        # Create the connection matrix
        n = len(compartment_idcs)
        C = np.zeros((n, n))
        if symbolic:
            C = sp.Matrix(C)
        for compartment_src, compartment_tar, gC in connections:
            # Fetch the source and target index
            i0 = compartment_idcs[compartment_src]
            i1 = compartment_idcs[compartment_tar]

            # Calculate the conductances after dividing by the corresponding
            # compartment membrane capacitance
            gC0 = gC / compartment_src.Cm
            gC1 = gC / compartment_tar.Cm

            # Make sure there are no self-connections and that the connection
            # has not already been specified
            if i0 == i1:
                raise Exception(
                    "Found invalid self-connection between compartments")
            if (C[i0, i1] != 0.0 and C[i0, i1] != gC0) or (C[i1, i0] != 0.0 and
                                                           C[i1, i0] != gC1):
                raise Exception(
                    "Found multiple conflicting connections between compartments"
                )

            # Symmetrically store the connection weights
            C[i0, i1] = gC0
            C[i1, i0] = gC1

        # Assemble the compartment list, generate unique names for all
        # compartments and channels that do not yet have names
        compartments = [None] * n
        for compartment, idx in compartment_idcs.items():
            if compartment.name is None:
                compartment.name = _make_unique_name(compartment_names, "comp",
                                                     idx + 1)
                compartment_names[compartment.name] = compartment

            compartment_channel_names = set()
            for channel in compartment.channels:
                if channel.name is None:
                    channel.name = _make_unique_name(channel_names, "chan", 1)
                if not channel.name in channel_names:
                    channel_names[channel.name] = set()
                channel_names[channel.name].add(channel)
                if channel.name in compartment_channel_names:
                    raise Exception("Channel with name \"" + channel.name +
                                    "\" exists multiple time in compartment \""
                                    + compartment.name + "\"")
                compartment_channel_names.add(channel.name)

            compartments[idx] = compartment

        # Make sure all channel and compartment names are valid
        name_pattern = re.compile("^[a-z][a-z_0-9]*$")
        for name in itertools.chain(compartment_names.keys(),
                                    channel_names.keys()):
            if not name_pattern.match(name):
                raise Exception(
                    "\"" + name + "\" is not a valid compartment " +
                    "or channel name; must match the following regular " +
                    "expression: [a-z][a-z_0-9]*")

        # Prune all compartments that are not connected to the soma.
        # Perform a deep copy of the compartment structures that are
        # actually passed to the Assembled neuron instance.
        visited = set()
        queue = [compartment_idcs[compartment_names["soma"]]]
        while len(queue) > 0:
            i = queue.pop()
            visited.add(i)
            for j in range(n):
                if C[i, j] != 0 and not j in visited:
                    queue.append(j)
        valid = list(sorted(visited))
        compartments = [copy.deepcopy(compartments[i]) for i in valid]
        if symbolic:
            C = C[valid, valid]
        else:
            C = C[np.ix_(valid, valid)]

        # Assemble a list of input channels
        inputs = []
        for i, compartment in enumerate(compartments):
            for j, channel in enumerate(compartment.channels):
                if not channel.is_static():
                    inputs.append((compartment.name + "." + channel.name, i,
                                   channel))

        # Fill the A, B matrices, which translate an input vector to
        # corresponding components of the linear and constant part of
        # the dynamical system equation; as well as the
        # Aconst and Bconst vectors which contain all constant
        # conductances and currents
        m, n = len(compartments), len(inputs)
        A, B, Asom, Bsom = np.zeros((4, m, n))
        if symbolic:
            A = sp.Matrix(A)
            B = sp.Matrix(B)
            Asom = sp.Matrix(Asom)
            Bsom = sp.Matrix(Bsom)
        for j in range(n):
            _, i, channel = inputs[j]
            CmInv = 1 / compartments[i].Cm
            a, b = 0, 0
            if channel.type == 'cond':
                a = -CmInv
                b = channel.Erev * CmInv
            elif channel.type == 'cur':
                b = channel.mul * CmInv
            if j > 0 or channel.is_somatic_current():
                Asom[i, j] = a
                Bsom[i, j] = b
            A[i, j] = a
            B[i, j] = b

        Aconst, Bconst, Asom_const, Bsom_const = np.zeros((4, m))
        if symbolic:
            Aconst = sp.Matrix(Aconst)
            Bconst = sp.Matrix(Bconst)
            Asom_const = sp.Matrix(Asom_const)
            Bsom_const = sp.Matrix(Bsom_const)
        for i, compartment in enumerate(compartments):
            CmInv = 1 / compartments[i].Cm
            Aconst[i] = -np.sum(C[i, :])  # Coupling conductances
            Asom_const[i] = -np.sum(C[i, :])  # Coupling conductances
            for channel in compartment.channels:
                a, b = 0, 0
                if channel.is_static():
                    if channel.type == 'cond':
                        a -= channel.g * CmInv
                        b += channel.Erev * channel.g * CmInv
                    elif channel.type == 'cur':
                        b += channel.J * channel.mul * CmInv
                if i > 0 or channel.is_somatic_current():
                    Asom_const[i] += a
                    Bsom_const[i] += b
                Aconst[i] += a
                Bconst[i] += b

        # Assemble the weight distribution matrices
        o = 2 * n + m
        WA, WB = np.zeros((2, n, o))
        WAconst = np.zeros((m, o))
        W0 = np.ones(o)
        for i in range(n):
            WA[i, i] = 1
            WB[i, n + i] = 1
        for i in range(m):
            WAconst[i, 2 * n + i] = 1
            W0[2 * n + i] = 0

        # Reduce the number of weights that are actually mapped onto non-zero
        # values
        WAvalid = np.sum(A, axis=0) != 0.0
        WBvalid = np.sum(B, axis=0) != 0.0
        WAconst_valid = np.ones(m, dtype=np.bool)
        v = np.concatenate((WAvalid, WBvalid, WAconst_valid), axis=0)
        WA, WB, WAconst, W0 = WA[:, v], WB[:, v], WAconst[:, v], W0[v]

        if symbolic:
            WA, WB = sp.Matrix(WA), sp.Matrix(WB)
            WAconst = sp.Matrix(WAconst)
            W0 = sp.Matrix(W0)

        # Throw an exception if the A matrix is singular for zero input
        if not symbolic:
            Am = C + np.diag(Aconst)
            if np.abs(np.linalg.det(Am)) < 1e-9:
                raise Exception(
                    "The specified neuron is not stable for zero input. " +
                    "Maybe add a conductance-based leak channel to the soma?")

        # Pass the canonicalised neuron description to the AssembledNeuron
        # instance
        return AssembledNeuron(symbolic, compartments, inputs, A, Aconst, B,
                               Bconst, Asom, Asom_const, Bsom, Bsom_const, C,
                               WA, WB, WAconst, W0)


def print_progress(perc):
    """
    Function which prints the progress of an operation given in percent.
    """
    import sys
    sys.stdout.write("\r{:6.2f}% done".format(perc * 100))
    if np.abs(perc - 1.0) < 1e-9:
        sys.stdout.write("\n")
    sys.stdout.flush()

def pluck(dict, *keys):
    """
    Use this function to destructure a dictionary. See

    https://stackoverflow.com/a/17074606
    """
    return [dict[key] for key in keys]
    
class AssembledNeuron:
    """
    The AssembledNeuron class represents a neuron that has been assembled from
    the descriptor classes defined above. In contrast to the descriptor classes,
    the data describing the Assembled neuron has been brought into a checked,
    canonical form.
    """

    """
    Internally used tag class.
    """
    class NoCanonWrapper:
        def __init__(self, data):
            self.data = data

    class SimulationResult:
        """
        The SimulationResult class holds the result of a single-neuron
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

        def __init__(self, state, out, isom, v, times, in_refrac):
            self.state = state
            self.out = out
            self.isom = isom
            self.v = v
            self.times = np.atleast_1d(times)
            self.in_refrac = in_refrac

    def __init__(self, symbolic, compartments, inputs, A, Aconst, B, Bconst,
                 Asom, Asom_const, Bsom, Bsom_const, C, WA, WB, WAconst, W0):
        # Make sure all matrices have the right shape
        assert len(compartments
                   ) == C.shape[0] == C.shape[1] == A.shape[0] == B.shape[0]
        assert len(inputs) == A.shape[1] == B.shape[1]
        assert WA.shape[1] == WB.shape[1] == WAconst.shape[1] == W0.shape[0]
        assert WA.shape[0] == WB.shape[0] == len(inputs)
        assert WAconst.shape[0] == Aconst.shape[0] == Bconst.shape[0] == len(
            compartments)
        assert np.prod(Aconst.shape) == np.prod(
            Bconst.shape) == len(compartments)
        assert Asom.shape == A.shape
        assert Bsom.shape == B.shape
        assert Asom_const.shape == Aconst.shape
        assert Bsom_const.shape == Bconst.shape

        # Copy the parameters to the local instance
        self._symbolic = symbolic
        self._n = len(compartments)
        self._compartments = compartments
        self._inputs = inputs
        self._A = A
        self._Aconst = Aconst
        self._B = B
        self._Bconst = Bconst
        self._Asom = Asom
        self._Asom_const = Asom_const
        self._Bsom = Bsom
        self._Bsom_const = Bsom_const
        self._C = C
        self._WA = WA
        self._WB = WB
        self._WAconst = WAconst
        self._W0 = W0

        # Set a few publicly available properties
        self.n_compartments = len(self._compartments)
        self.n_inputs = len(self._inputs)

        pass

    def inputs(self, latex=False):
        """
        Returns the names of the inidividual input channels. When supplying
        input to the AssembledNeuron instance, the individual input values
        must be specified in the order.

        Parameters
        ==========

        latex: if set to true, a compact LaTeX representation of the input
        channels is returned
        """
        if not latex:
            return list(map(lambda x: x[0], self._inputs))

        res = [None] * len(self._inputs)

        # Count the number of compartments with input channels
        n, last_j = 0, -1
        for _, j, _ in self._inputs:
            last_j, n = j, n if last_j == j else n + 1

        idx, last_j = 0, -1
        for i, channel_descr in enumerate(self._inputs):
            # Fetch information about the input, generate a compartment index
            name, j, channel = channel_descr
            last_j, idx = j, idx if last_j == j else idx + 1
            _, channel_name = name.split('.', 2)

            # Assemble the variable name
            prefix = "g" if channel.type == "cond" else "J"

            # Do not print the compartment number if there is only one
            # with inputs compartment
            res[i] = prefix + ("" if n <= 1 else "^" + str(idx))

            suffix_map = {"exc": "E", "inh": "I"}
            suffix = "" if prefix.lower() == channel_name else channel_name
            if suffix in suffix_map:
                suffix = suffix_map[suffix]
            if len(suffix) > 0:
                res[i] += "_\\mathrm{" + suffix + "}"
        return res

    def input_syms(self):
        """
        Returns a vector of sympy symbols representing each input
        """
        import sympy as sp
        return list(map(sp.Symbol, self.inputs(True)))

    def weight_syms(self):
        import sympy as sp
        n = self._W0.shape[0]
        return list(map(sp.Symbol,
                        map(lambda x: "w{}".format(x), range(n))))

    def compartments(self):
        """
        Returns the name of each compartment as a list. The order of the entires
        in the list corresponds to that adhered by recorded state variables,
        i.e. the i-th column in a recorded voltage trace corresponds to the i-th
        compartment listed in this list.
        """
        return list(map(lambda x: x.name, self._compartments))

    def soma(self):
        return self._compartments[0]

    def w0(self):
        return self._W0

    def A(self, xs, ws=None, somatic=False, symbolic=None):
        """
        Returns the A matrix of the linear dynamical system for the given input
        vector.

        Parameters
        ==========

        xs: a one-dimensional vector, where each dimension corresponds to one
        input channel as laid out by the inputs() method.

        ws: is a weight vector used for fine-tuning of the predictions of the
        neuron model to empirical measurements. The dimensionality of the weight
        vector can be obtained by calling the w0() method, which will return an
        initial weight vector that can be used in an optimization process.
        If None is given, the weight vector will be set to w0().

        somatic: If true, returns the matrix used to compute somatic input
        currents, i.e. does not include the leak channel or other channels
        explicitly marked as "non-somatic".

        """

        # Determine whether to evaluate the matrix symbolically
        symbolic = symbolic or self._symbolic

        # Use the initial weight vector if none is given
        if ws is None:
            ws = self.w0()

        # Make sure xs is a sympy matrix if we're doing symbolic computation
        if symbolic:
            import sympy as sp
            xs = sp.Matrix(xs)
            ws = sp.Matrix(ws)

        # Either fetch the normal matrix or the matrix used for somatic current
        # calculation
        A = self._Asom if somatic else self._A
        Aconst = self._Asom_const if somatic else self._Aconst

        # Either use numpy or sympy for the actual computation
        if symbolic:
            a = A * (sp.matrix_multiply_elementwise(xs, (self._WA * ws)))
            a_const = sp.Matrix(Aconst) + (self._WAconst * ws)
            return self._C + sp.diag(*(a + a_const))
        else:
            a = A @ (xs * (self._WA @ ws))
            a_const = Aconst + (self._WAconst @ ws)
            return self._C + np.diag(a + a_const)

    def b(self, xs, ws=None, somatic=False, symbolic=None):
        """
        Returns the b vector of the linear dynamical system for the given input
        vector. The b vector corresponds

        Parameters
        ==========

        xs: one-dimensional vector corresponding to the input passed to the
        individual channels. The mapping between the input dimensions and the
        corresponding channels is determined by the inputs() method.

        ws: is a weight vector used for fine-tuning of the predictions of the
        neuron model to empirical measurements. The dimensionality of the weight
        vector can be obtained by calling the w0() method, which will return an
        initial weight vector that can be used in an optimization process.
        If None is given, the weight vector will be set to w0().

        somatic: If true, returns the matrix used to compute somatic input
        currents, i.e. does not include the somatic leak channel or other
        channels explicitly marked as "non-somatic".
        """

        # Determine whether to evaluate the matrix symbolically
        symbolic = symbolic or self._symbolic

        # Use the initial weight vector if none is given
        if ws is None:
            ws = self.w0()

        # Make sure xs is a sympy matrix if we're doing symbolic computation
        if symbolic:
            import sympy as sp
            xs = sp.Matrix(xs)
            ws = sp.Matrix(ws)

        # Either fetch the normal matrix or the matrix used for somatic current
        # calculation
        B = self._Bsom if somatic else self._B
        Bconst = self._Bsom_const if somatic else self._Bconst

        # Either use numpy or sypy for the actual computation
        if symbolic:
            b = B * (sp.matrix_multiply_elementwise(xs, (self._WB * ws)))
            return b + sp.Matrix(Bconst)
        else:
            b = B @ (xs * (self._WB @ ws))
            return b + Bconst

    def _canonicalise_input_array(self, xs):
        xs = np.atleast_1d(xs)
        if xs.ndim == 1 and self.n_inputs == 1:
            xs = xs.reshape((1, -1))
        if xs.ndim == 1 and xs.size == self.n_inputs:
            xs = xs.reshape((-1, 1))
        xs = np.atleast_2d(xs)
        assert xs.shape[0] == self.n_inputs
        return xs

    def _apply_to_array(self, xs, f, res_dtype, progress, *args):
        """
        Boilerplate code used internally in the vEq and iSom methods to create
        an output array of the same shape as the input array.
        """
        import multiprocessing.dummy as mp  # Uses threads instead of processes

        if not isinstance(xs, AssembledNeuron.NoCanonWrapper):
            # Make sure the input dimension matches the number of input channels,
            # special handling for no input channels
            if self.n_inputs == 0:
                assert xs.size == 0
                return f([], ws)
            xs = self._canonicalise_input_array(xs)
        else:
            xs = xs.data

        # Prepare the output
        n = int(np.product(xs.shape[1:]))
        res = np.empty(n, dtype=res_dtype)

        # Reshape xs to a matrix containing the raw number of input samples,
        # create a flat output array. Run in as many threads as available on the
        # machine.
        xss = xs.reshape((-1, n))

        with mp.Pool() as pool:

            def loop(i):
                res[i] = f(xss[:, i], *args)

            for i, _ in enumerate(pool.imap_unordered(loop, range(n))):
                if (not progress is None) and (i % 100 == 0 or i == n - 1):
                    progress((i + 1) / n)
        return res.reshape(xs.shape[1:])

    def vEq(self, xs, ws=None):
        """
        Estimates the equilibrium potential of the neuron for constant input xs.
        This assumes that there is no spiking mechanism in the neuron.

        Parameters
        ==========

        xs: a numpy array where the first dimension is equal to the number
        of input channels. The mapping between the input dimensions and the
        corresponding channels is determined by the inputs() method.

        ws: is a weight vector used for fine-tuning of the current function
        to empirical measurements. The dimensionality of the weight vector
        can be obtained by calling the w0() method, which will return an
        initial weight vector that can be used in an optimization process.
        If None is given, the weight vector will be set to w0()

        Result
        ======

        Returns a numpy array of the same size as xs.shape[1:] (i.e. the first
        input dimension is removed).
        """

        def vEq_single(x, ws):
            """
            Calculates the equilibrium potential for a single input vector x
            """
            return np.linalg.solve(-self.A(x, ws), self.b(x, ws))[0]

        # Use no initial input weights if no weights are given
        if ws is None:
            ws = self.w0()

        return self._apply_to_array(xs, vEq_single, np.float64, None, ws)

    def iSom(self, xs, ws=None, vSom=None):
        """
        Estimates the amount of input current flowing into the somatic
        compartment for the given combination of inputs.

        Parameters
        ==========

        xs: a numpy array where the first dimension is equal to the number
        of input channels. The order of the input dimensions is determined by
        the inputs() method.

        ws: is a weight vector used for fine-tuning of the current function
        to empirical measurements. The dimensionality of the weight vector
        can be obtained by calling the w0() method, which will return an
        initial weight vector that can be used in an optimization process.
        If None is given, the weight vector will be set to w0()

        Result
        ======

        Returns a numpy array of the same size as xs.shape[1:] (i.e. the first
        input dimension is removed).
        """

        # Set vSom to the default value if no value is given
        soma = self.soma()
        vSom = (soma.v_reset + soma.v_th) * 0.5 if vSom is None else vSom

        def iSom_single(x, ws):
            """
            Calculates the synaptic current for a single input vector.
            """

            # Fetch the soma and calculate the expected average membrane
            # potential
            soma = self._compartments[0]

            # Fetch the A and b matrix determining the linear dynamical system,
            # then reduce the system
            A, b = self.A(x, ws, somatic=True), self.b(x, ws, somatic=True)
            A_red, b_red, c_red = self.reduce_system(A, b, vSom=vSom)

            # Calculate the offset current H0
            H0 = soma.Cm * (vSom * A[0, 0] + b[0])

            # Calculate the equilibrium potential
            if self.n_compartments == 1:
                vEq = None
                H1 = 0.0
            elif self.n_compartments == 2:
                vEq = np.linalg.solve(-A_red, b_red)
                H1 = soma.Cm * c_red * vEq
            else:
                vEq = np.linalg.solve(-A_red, b_red)
                H1 = soma.Cm * c_red @ vEq

            return H0 + H1

        # Use no initial input weights if no weights are given
        if ws is None:
            ws = self.w0()

        return self._apply_to_array(xs, iSom_single, np.float64, None, ws)

    def _iSom_fun_expr(self):
        import sympy as sp
        # Fetch the soma and calculate the expected average membrane
        # potential
        soma = self._compartments[0]
        vSom = (soma.v_reset + soma.v_th) * 0.5

        # Symbolic model parameters
        ws = self.weight_syms()
        xs = self.input_syms()

        # Compile the synaptic_nonlinearity into a lambda
        H = self.synaptic_nonlinearity(ws=ws, vSom=vSom)
        return H, ws, xs


    def iSom_fun(self):
        import sympy as sp
        H, ws, xs = self._iSom_fun_expr()
        H_fun = sp.lambdify(xs + ws, H, 'numpy')

        def H_fun_wrapper(xs, ws=None):
            if ws is None:
                ws = self.w0()
            return np.array(H_fun(*xs, *ws), dtype=np.float64)

        return H_fun_wrapper

    def iSom_fun_weight_jacobian(self):
        import sympy as sp
        H, ws, xs = self._iSom_fun_expr()
        H_funs = [sp.lambdify(xs + ws, sp.diff(H, w)) for w in ws]

        def H_fun_wrapper(xs, ws=None):
            if ws is None:
                ws = self.w0()
            res = np.empty((len(H_funs),) + xs.shape[1:])
            for i, H_fun in enumerate(H_funs):
                res[i] = H_fun(*xs, *ws)
            return res

        return H_fun_wrapper

    @staticmethod
    def _mk_poisson_sources(sim, xs, noise, tau, rate):
        return [sim.PoissonSource(
            np.random.randint(0, 0x7FFFFF),
            0 if not noise else rate[i],
            0.0,
            0.0 if not noise else 2.0 * x,
            tau[i],
            x if not noise else 0.0
        ) for i, x in enumerate(xs)]

    @staticmethod
    def _mk_poisson_matrix_source(sim, xs, weights, tau, rates):
        # Set all weights smaller than one millionth of the max weight to zero
        def filter_rates_weights(rs, ws):
            th = max(1e-24, np.max(np.abs(ws)) * 1e-6)
            return np.array(list(filter(lambda x: x[0] > 0 and np.abs(x[1]) >= th, zip(rs, ws)))).T.reshape(2, -1)

        # Create the PoissonMatrixSource instances
        res = []
        for i in range(weights.shape[0]):
            rs, ws = filter_rates_weights(rates, weights[i])
            source = sim.PoissonMatrixSource.create(
                tau=tau[i], rates=rs, weights=ws, gain=xs[i] / (rs @ ws)
            )
            res.append(source)
        return res

    @staticmethod
    def _simulate_noisy(sim, n, xs, noise, tau, rate):
        # Make sure "tau" and "rate" are arrays
        if not hasattr(tau, '__len__'):
            tau = tau * np.ones(len(xs))
        if not hasattr(rate, '__len__'):
            rate = rate * np.ones(len(xs))

        # If "noise" is a boolean flag, just simulate a poisson source
        if isinstance(noise, bool):
            sources = AssembledNeuron._mk_poisson_sources(sim, xs, noise, tau, rate)
            return sim.simulate_poisson(sources, n)
        elif isinstance(noise, np.ndarray):
            # If "noise" is a matrix, it corresponds to a neural weight vector
            # for each input and "xs" contains a set of pre-synaptic rates.
            assert noise.ndim == 2 and rate.ndim == 1, "\"noise\" must be a 2D weight matrix and rate must be a vector of pre-synaptic activities"
            assert noise.shape[0] == xs.shape[0]
            assert noise.shape[1] == rate.shape[0], "incompatible dimensions"
            sources = AssembledNeuron._mk_poisson_matrix_source(sim, xs, noise, tau, rate)
            return sim.simulate_poisson_matrix_source(sources, n)
        elif isinstance(noise, str):
            # Otherwise "noise" is a file containing a noise profile. A noise
            # profile describes the random distributions for the inter-spike
            # intervals of a 
            import h5py
            arr_qs_isi, arr_qs_ws, qs_isi, qs_ws, filters = [], [], [], [], []
            with h5py.File(noise, 'r') as f:
                for i, x in enumerate(xs):
                    # Load the corresponding datasets
                    gain = f['gain_{}'.format(i)][()]
                    arr_qs_ws.append(np.array(f['qs_ws_{}'.format(i)], dtype=np.float64, order='C'))
                    arr_qs_isi.append(np.array(f['qs_isi_{}'.format(i)], dtype=np.float64, order='C'))

                    # Create the SampledRandomDistribution objects
                    qs_ws.append(sim.SampledRandomDistribution.create(
                        arr_qs_ws[-1],
                        seed=np.random.randint(0, 0x7FFFFF),
                        trafo=sim.TRAFO_EXP10
                    ))
                    qs_isi.append(sim.SampledRandomDistribution.create(
                        arr_qs_isi[-1],
                        seed=np.random.randint(0, 0x7FFFFF),
                        trafo=sim.TRAFO_EXP10_INV
                    ))

                    # Create the filter objects
                    filters.append(sim.ExponentialFilter(
                        tau[i], # tau
                        0.0, # offs
                        x * gain # gain
                    ))
            return sim.simulate_noise_profile(qs_ws, qs_isi, filters, n)



    def iSom_empirical(self,
                       xs,
                       T=1.0,
                       dt=1e-4,
                       noise=False,
                       tau=5e-3,
                       rate=1000,
                       progress=print_progress,
                       include_refrac=False,
                       **kwargs):
        """
        Calculates the amount of current flowing into the somatic compartment by
        performing a numerical simulation of the neuron model.

        Parameters
        ==========

        xs: a numpy array where the first dimension is equal to the number
        of input channels. The order of the input dimensions is determined by
        the inputs() method.

        T: length of the simulation in seconds.

        dt: simulation timestep

        Result
        ======

        Returns a numpy array of the same size as xs.shape[1:] (i.e. the first
        input dimension is removed).
        """
        from .lif_utils import spike_frequency

        sim = self.simulator(
            dt=dt,
            record_isom=True,
            record_in_refrac=True,
            record_spike_times=True,
            **kwargs)
        n = int(T / dt + 1e-9)

        def iSom_empirical_single(x):
            res = self._simulate_noisy(sim, n, x, noise, tau, rate)
            if include_refrac:
                iSom = np.mean(res.isom)
            else:
                iSom = np.mean(res.isom[~res.in_refrac])
            return iSom, spike_frequency(res.times)

        return self._apply_to_array(xs, iSom_empirical_single,
                                    [('isom', 'f8'), ('rate', 'f8')], progress)


    def lif_parameters(self):
        """
        Returns the parameters of the somatic compartment, that acts like a LIF
        neuron.
        """

        # Fetch the model parameters
        soma = self.soma()
        gL = None
        EL = None
        for channel in soma.channels:
            if channel.type == "cond" and channel.is_static():
                if (not gL is None) or (not EL is None):
                    raise Exception(
                        "Somatic compartment has more than one leak channel")
                gL = channel.g
                EL = channel.Erev
        if (gL is None) or (EL is None):
            raise Exception("Somatic compartment does not have a leak channel")
        tau_rc = soma.Cm / gL
        v_th = soma.v_th
        v_reset = soma.v_reset
        tau_ref = soma.tau_ref + soma.tau_spike
        v_som = (soma.v_reset + soma.v_th) * 0.5

        return {
            'gL': gL,
            'EL': EL,
            'v_th': v_th,
            'v_reset': v_reset,
            'v_som': v_som,
            'tau_rc': tau_rc,
            'tau_ref': tau_ref,
        }

    def iTh(self):
        """
        Estimates the threshold somatic input current.
        """
        v_th, EL, gL = pluck(self.lif_parameters(), 'v_th', 'EL', 'gL')
        return (v_th - EL) * gL

    def estimate_lif_rate_from_current(self, J):
        """
        Converts the LIF-equivalent current J to the expected average spike rate
        """

        # Fetch the parameters of the intrinsic LIF neuron
        gL, EL, tau_rc, v_th, tau_ref = pluck(self.lif_parameters(),
                                              'gL', 'EL', 'tau_rc', 'v_th', 'tau_ref')

        # Compute the equilibrium potential
        v_eq = EL + J / gL
        mask = v_eq > v_th  # Return zero if the current is blow threshold

        # Estimate the spike rate
        t_spike = -np.log(mask * gL * (v_eq - v_th) /
                          (mask * J + (1.0 - mask)) + (1.0 - mask)) * tau_rc
        return mask * (1.0 / (tau_ref + t_spike))

    def estimate_lif_current_from_rate(self, rate):
        """
        Estimates the somatic current for the given spike rate.
        """

        # Fetch the parameters of the intrinsic LIF neuron
        gL, EL, tau_rc, v_th, tau_ref = pluck(self.lif_parameters(),
                                              'gL', 'EL', 'tau_rc', 'v_th', 'tau_ref')

        # Compute the underlying t_spike
        mask = rate > 0.0
        J = gL * (EL - v_th) / (np.exp(
            (rate * tau_ref - 1.0) / (rate * tau_rc + (1.0 - mask))) - 1.0)
        return mask * J

    def lif_rate_to_normalised_current(self, rate):
        from . import lif_utils

        # Fetch the model parameter tau_ref/tau_rc
        soma = self.soma()
        gLeak = 0.0
        for channel in soma.channels:
            if channel.type == "cond" and channel.is_static():
                gLeak += channel.g
        tau_rc = soma.Cm / gLeak
        tau_ref = soma.tau_ref

        return lif_utils.lif_rate_inv(rate, tau_ref, tau_rc)

    def _collect_inputs(self, expr):
        """
        Collects the input variables in the given sympy expressions, i.e.

            gE eE - v gE + gI eI - v gI

        is rewritten to

            gE (eE - v) + gI (eI - v)
        """
        import sympy as sp

        expr = sp.expand(expr)  # Expand the expression before simplifying it
        for sym in self.input_syms():
            if isinstance(expr, sp.Matrix):
                for i in range(expr.shape[0]):
                    for j in range(expr.shape[1]):
                        expr[i, j] = self._collect_inputs(expr[i, j])
            else:
                expr = sp.collect(expr, sym)
        return expr

    def system(self, ws=None, somatic=False):
        """
        Returns a symbolic representation of the A and b matrices defining the
        dynamics of the neuron model.
        """
        xs = self.input_syms()
        A = self.A(xs, ws, somatic=somatic, symbolic=True)
        b = self.b(xs, ws, somatic=somatic, symbolic=True)
        return A, b

    def reduce_system_general(self, A, b, i, v, symbolic=None):
        """
        Reduces given the system by clamping a certain compartment i to the
        given potential v. Returns three matrices corresponding to the reduced
        system A_red, b_red and the vector required to compute the current
        flowing into the clamped compartment.
        """

        # Determine whether to evaluate the matrix symbolically
        symbolic = symbolic or self._symbolic

        # Abort if this is a one-compartment neuron
        n = A.shape[0]
        if n <= 1:
            return 0, 0, 0

        # Fetch the A sub-matrix, add the current caused by the clamped somatic
        # compartment
        sel = list(range(0, i)) + list(range(i + 1, n))
        if symbolic:
            import sympy as sp
            A_red = sp.Matrix(A.extract(sel, sel))
            b_red = sp.Matrix(b.extract(sel, [0])) + v * sp.Matrix(A.extract(sel, [i]))
            c_red = sp.Matrix(A.extract([i], sel))
        else:
            A_red = A[np.ix_(sel, sel)]
            b_red = b[sel] + v * A[sel, i]
            c_red = A[i, sel]
        return A_red, b_red, c_red

    def reduce_system(self, A, b, vSom=None, symbolic=None):
        """
        Reduces the system returned by the system() function to a reduced system
        """

        # Determine whether to evaluate the matrix symbolically
        symbolic = symbolic or self._symbolic

        # If no specifc value for vSom is given, replace it with a default value
        if vSom is None:
            if symbolic:
                vSom = sp.Symbol("v_{\\mathrm{som}}")
            else:
                soma = self.soma()
                vSom = (soma.v_reset + soma.v_th) * 0.5

        return self.reduce_system_general(A, b, 0, vSom, symbolic)

    def synaptic_nonlinearity(self, vSom=None, ws=None, nsimplify=True):
        """
        Returns the synaptic nonlinearity function as a symbolic expression.
        Implements the exact equation that is in the paper.
        """
        import sympy as sp

        # If no specifc value for vSom is given, replace it with a symbol
        if vSom is None:
            vSom = sp.Symbol("v_{\\mathrm{som}}")

        # Fetch both the full and the reduced dynamical system
        A, b = self.system(ws, somatic=True)
        A_red, b_red, c_red = self.reduce_system(A, b, vSom, symbolic=True)

        # Calculate the offset current H0
        H0 = self.soma().Cm * (vSom * A[0, 0] + b[0])

        # Calculate the equilibrium potential
        vEq = 0
        if self.n_compartments > 1:
            vEq = -A_red.inv() * b_red
        H1 = self.soma().Cm * c_red * vEq
        if isinstance(H1, sp.Matrix):
            assert H1.shape == (1, 1)
            H1 = H1[0, 0]

        # Add H0 and H1, simplify the resulting expression
        H = sp.simplify(H0 + H1)

        # If nsimplify is set to true, try to simplify all floating point
        # numbers to rational numbers
        if nsimplify:
            H = sp.simplify(sp.nsimplify(H, tolerance=1e-12))
        return self._collect_inputs(H)

    def vEq(self, ws=None, nsimplify=True):
        """
        Provides a symbolic expression for the equilibrium potential.
        """
        import sympy as sp

        A, b = self.system(ws)
        vEq = sp.simplify(-(A.inv() * b))
        if nsimplify:
            vEq = sp.simplify(sp.nsimplify(vEq, tolerance=1e-12))
        return vEq

    def _fit_model_weights_two_comp(self, Js, samples, compute_internal_weights=True, old_w=None, lambda_old_w=None):
        import sympy as sp
        from .multi_compartment_lif_solver_qp import solve_linearly_constrained_quadratic_loss

        # Filter for samples with super-threshold currents
        valid = Js > self.iTh()

        # Assemble the quadratic loss function
        scale_cur = 1e9
        scale_cond = 1e6
        n_samples = samples.shape[1]
        n_valid = int(np.sum(valid))

        if not old_w is None:
            old_w = old_w / old_w[1]
            if lambda_old_w is None:
                lambda_old_w = 1e-4
            lambda_old_w = np.sqrt(lambda_old_w * n_valid)

        n_cstr = n_valid + (0 if old_w is None else 5)
        n_vars = 5
        C, d = np.zeros((n_cstr, n_vars)), np.zeros(n_cstr)
        C[:n_valid, 0] = 1
        C[:n_valid, 1] = samples[1, valid] * scale_cond
        C[:n_valid, 2] = -Js[valid] * scale_cur
        C[:n_valid, 3] = -Js[valid] * samples[0, valid] * scale_cur * scale_cond
        C[:n_valid, 4] = -Js[valid] * samples[1, valid] * scale_cur * scale_cond
        d[:n_valid] = -samples[0, valid] * scale_cond

        if not old_w is None:
            old_w_c = np.copy(old_w)
            old_w_c[np.abs(old_w) < 1e-20] = 1e-20
            C[n_valid + 0, 0] = lambda_old_w / (old_w_c[0] * scale_cond)
            C[n_valid + 1, 1] = lambda_old_w / old_w_c[2]
            C[n_valid + 2, 2] = lambda_old_w * scale_cur / (old_w_c[3] * scale_cond)
            C[n_valid + 3, 3] = lambda_old_w * scale_cur / old_w_c[4]
            C[n_valid + 4, 4] = lambda_old_w * scale_cur / old_w_c[5]

            d[n_valid:] = lambda_old_w

        G, h = np.zeros((3, n_vars)), np.zeros(3)
        G[0, 2] = -1
        G[1, 3] = -1
        G[2, 4] = -1

        # Solve for model weights
        ws = solve_linearly_constrained_quadratic_loss(C=C, d=d, G=G, h=h, tol=1e-12)[:, 0]

        # Rescale the output
        ws = np.array((
            ws[0] / scale_cond,
            1.0,
            ws[1],
            ws[2] * scale_cur / scale_cond,
            ws[3] * scale_cur,
            ws[4] * scale_cur,
        ))

#        import matplotlib.pyplot as plt

        def H(w, gE, gI):
            return (w[0] + w[1] * gE + w[2] * gI) / (w[3] + w[4] * gE + w[5] * gI)

        gE, gI = samples
        gEs = np.linspace(np.min(gE), np.max(gE))
        gIs = np.linspace(np.min(gI), np.max(gI))
        gEss, gIss = np.meshgrid(gEs, gIs)
        Js_model = H(ws, gEss, gIss)

#        fig, ax = plt.subplots()
#        ax.scatter(gE, gI, c=Js)
#        ax.contour(gEs, gIs, Js_model, levels=np.linspace(0, np.max(Js), 100), colors=['white'])
#        ax.contour(gEs, gIs, Js_model, levels=np.linspace(0, np.max(Js), 100), colors=['r'], linestyles='--')
#        ax.set_xlim(np.min(gE), np.max(gE))
#        ax.set_ylim(np.min(gI), np.max(gI))

        # Transform the computed weights into the parameterisation used by
        # the rest of the code. While this could be done in closed form, I'm
        # lazy and will just let sypy do the job. We force model parameter w0
        # to one.

        # Fetch the fully parameterised dendritic nonlinearity canonicalise it
        if compute_internal_weights:
            # Assume a0 = 1
            ws /= ws[3]

            H, ws_syms, xs_syms = self._iSom_fun_expr()
            H = sp.cancel(sp.together(H)).subs(ws_syms[0], 1.0) # => w0 = 0
            H_num, H_den = sp.fraction(H)
            H_num_p, H_den_p = sp.Poly(H_num, *xs_syms), sp.Poly(H_den, *xs_syms)
            b1, b2, b0 = H_num_p.coeffs()
            a1, a2, a0 = H_den_p.coeffs()

            # Compute the model weights
            sols = sp.nonlinsolve([
                sp.Eq(b0 / a0, ws[0]),
                sp.Eq(b1 / a0, ws[1]),
                sp.Eq(b2 / a0, ws[2]),
                sp.Eq(a1 / a0, ws[4]),
                sp.Eq(a2 / a0, ws[5]),
            ], ws_syms[1:])
            sol = next(iter(sols))
            print(sol)
            ws_opt = np.array(list(map(lambda x: x.evalf(), sol)), dtype=np.float64)
            ws_opt = np.concatenate(((1.0,), ws_opt))
        else:
            ws_opt = None

        return ws_opt, ws

    def _fit_model_weights_generic(self, Js, samples):
        """
        Uses the Conjugate Gradient method to fit model weights to the given
        currents and samples.
        """
        import scipy.optimize

        # Optimise the model parameter for RMSE
        print("Computing H and H'...")
        H = self.iSom_fun()
        H_jacobian = self.iSom_fun_weight_jacobian()

        print("Fitting parameters...")
        valid = Js > self.iTh()

        norm = [1.0]
        n_samples = samples.shape[1]

        def err(w):
            return norm[0] * 0.5 * np.mean((H(samples[:, valid], w) - Js[valid]) ** 2)

        def jac(w):
            return norm[0] * H_jacobian(samples[:, valid], w) @ (H(samples[:, valid], w) - Js[valid]) / n_samples

        # Set the initial error to one during the optimization by dividing by
        # err0
        w0 = self.w0()
        norm[0] = 1.0 / err(w0)
        res = scipy.optimize.minimize(err, w0, method='CG', jac=jac, options={'disp':True})
        return res.x

    def fit_model_weights(self,
                          xMin=None,
                          xMax=None,
                          n_samples_per_dim=100,
                          fit_rate=True,
                          fit_relu=False,
                          fit_lif=False,
                          noise=False,
                          T=1.0,
                          dt=1e-5,
                          tau=5e-3,
                          rate=1000,
                          progress=print_progress,
                          random=None,
                          xs=None,
                          compute_internal_weights=True,
                          old_w=None,
                          lambda_old_w=None):
        """
        The fit_weights function fits the synaptic nonlinearity model weights to
        data obtained from numerical simulation.

        xMin: vector containing the minimum value for each inidividual input
        dimension.

        xMax: vector containing the maximum value for each inidividual input
        dimension.

        n_samples_per_dim: number of samples per input dimension.

        fit_rate: if True, fits for the output spike rate instead of currents.
        This is usually more useful.

        fit_relu: Instead of assuming a standard LIF somatic nonlinearity,
        assume that the neuron nonlinearity is a rectified linear unit (ReLU).
        This parameter only has an effect if fit_rate is True. The ReLU
        parameters (bias and slope) are automatically derived to fit the LIF
        response curve.

        fit_lif: Fit the LIF response curve to the actually measured
        relationship between J and Jth. This will return additional values
        values w, G, GInv. Where G is defined as

                G[J] = 1 / (tau_ref + t_spike(J))

        where t_spike(J) = -np.log(((w[0] + w[1] * J) / (w[2] + J)) * tau_rc).
        GInv is the inverse.

        noise: if True, adds artificial first-order low-pass filtered spike
        noise to the individual numerical simulations.

        T: time in seconds that should be simulated.

        dt: timestep that should be used for the simulation.

        tau: exponential filter applied to the input spike noise.

        rate: spike rate of the Poisson noise source used to generate the noisy
        input.

        Returns the optimized model parameters as well as the estimated RMSE.
        """
        import scipy.optimize

        assert (not fit_relu) or fit_rate, "fit_relu implies fit_rate"
        assert (not fit_lif) or fit_rate, "fit_lif implies fir_rate"
        assert not (fit_lif and fit_relu), "fit_lif and fit_relu are exclusive"

        assert ((xMin is None) == (xMax is None)) and ((xMin is None) != (xs is None)), "Either both xMin and xMax must be provided or xs must be set"

        # Step 1: Gather samples

        # Select the default random number generator
        if random is None:
            random = np.random

        # Make sure xMax has the correct dimensionality
        if xs is None:
            xMin = np.atleast_1d(xMin).flatten()
            xMax = np.atleast_1d(xMax).flatten()
            assert (len(xMax) == len(xMin) == self.n_inputs)

            # Sample the input space
            n_samples = n_samples_per_dim * self.n_inputs
            samples = np.empty((self.n_inputs, n_samples))
            for i in range(self.n_inputs):
                samples[i] = random.uniform(xMin[i], xMax[i], n_samples)
        else:
            assert xs.shape[0] == self.n_inputs
            n_samples = xs.shape[1]
            samples = np.copy(xs)

        if isinstance(noise, np.ndarray):
            assert not xs is None, "must provide \"xs\" in PoissonMatrixSource mode"
            assert isinstance(rate, np.ndarray), "\"rate\" must be an array of firing rates"
            assert noise.ndim == 2, "\"noise\" must be a 2D-array"
            assert rate.ndim == 1, "\"rate\" must be a 1D-array"
            assert rate.shape[0] == noise.shape[1], "number of pre-synaptic neurons does not match"
            assert noise.shape[0] == self.n_inputs, "weight matrix dimension does not match"

        # Measure the average spike rate for each of the samples
        record_isom = ((not fit_rate) or fit_relu or fit_lif)
        record_rate = fit_rate

        sim = self.simulator(
            dt=dt,
            record_isom=record_isom,
            record_in_refrac=record_isom,
            record_spike_times=record_rate)

        def iSom_empirical_single_noise(x):
            from .lif_utils import spike_frequency

            # Run the simulation
            n = int(T / dt + 1e-9)
            res = self._simulate_noisy(sim, n, x, noise, tau, rate)


            # Compute the somatic current, either according to the output rate
            # or according to the actual current
            isom, output_rate = None, None
            if record_rate:
                output_rate = spike_frequency(res.times)
            if record_isom:
                isom = np.mean(res.isom[~res.in_refrac])

            return isom, output_rate

        # Fetch the somatic current at the individual samples
        res = self._apply_to_array(samples, iSom_empirical_single_noise,
                [('isom', 'f8'), ('rate', 'f8')], progress)

        # Convert the simulation result to currents
        if fit_rate:
            if fit_relu:
                # Fit a ReLu to the isom/rate relationship
                valid = res['rate'] > np.percentile(res['rate'], 50)
                relu_slope, relu_bias = \
                    np.polyfit(res['isom'][valid], res['rate'][valid], 1)
                Js = np.maximum(0, (res['rate'] - relu_bias) / relu_slope)[valid]
                samples = samples[:, valid]
            elif fit_lif:
                # Fit the LIF response curve to the isom/rate relationship
                valid = res['rate'] > 0
                tau_rc, tau_ref = pluck(self.lif_parameters(), 'tau_rc', 'tau_ref')
                T = np.exp(-(1.0 / res['rate'][valid] - tau_ref) / tau_rc)
                n = int(np.sum(valid))
                C, d = np.zeros((n, 3)), np.zeros(n)
                C[:, 0] = 1
                C[:, 1] = res['isom'][valid]
                C[:, 2] = -T
                d = T * res['isom'][valid]
                w = np.linalg.lstsq(C, d, rcond=None)[0]

                def G(J):
                    mask = 1.0 * (np.equal(w[2] + J > 0, w[0] + w[1] * J > 0))
                    t_spike = -np.log(mask * (w[0] + w[1] * J) / (mask * (w[2] + J) + (1.0 - mask)) + (1.0 - mask)) * tau_rc
                    return np.maximum(0.0, mask * (1.0 / (tau_ref + t_spike)))

                def GInv(rate):
                    mask = 1.0 * (rate > 0.0)
                    T = np.exp(-(1.0 / ((1.0 - mask) + mask * rate) - tau_ref) / tau_rc)
                    return mask * (T * w[2] - w[0]) / ((1.0 - mask) + mask * (w[1] - T))

                Js = GInv(res['rate']) # Estimate the LIF current from the rate
            else:
                Js = self.estimate_lif_current_from_rate(res['rate'])
        else:
            Js = res['isom']

        import h5py
        with h5py.File('data.h5', 'w') as f:
            f.create_dataset('res_isom', data=res["isom"])
            f.create_dataset('res_rate', data=res["rate"])
            f.create_dataset('samples', data=samples)
            f.create_dataset('weights', data=noise)
            f.create_dataset('rates', data=rate)
            f.create_dataset('Js', data=Js)


        # Step 2: Solve for weights
        if self.n_compartments == 2:
            ws, ws_opt = self._fit_model_weights_two_comp(Js, samples, compute_internal_weights=compute_internal_weights, old_w=old_w, lambda_old_w=lambda_old_w)
            if not compute_internal_weights:
                return ws_opt
        else:
            ws = self._fit_model_weights_generic(Js, samples)

        # Step 3: Compute the initial and final error for evaluation purposes
        valid = Js > self.iTh()
        def err(ws):
            return 0.5 * np.mean((self.iSom(samples[:, valid], ws) - Js[valid]) ** 2)
        err_init = err(self.w0())
        err_opt = err(ws)

        if fit_relu:
            return ws, err_init, err_opt, relu_slope, relu_bias
        elif fit_lif:
            return ws, err_init, err_opt, w, G, GInv
        else:
            return ws, err_init, err_opt

    def connectivity_graph(self):
        """
        Returns the adjacency matrix containing the conductances between
        individual compartments.
        """
        n = self.n_compartments
        C = np.copy(self._C)
        for i in range(n):
            C[i] *= self._compartments[i].Cm
        return C

    def dynamics_model(self, ws=None, somatic=True):
        """
        Returns the matrices used to describe the dynamics of the neuron model.
        """

        # Apply the model weights
        if ws is None:
            ws = self.w0()

        # Either fetch the model used to calculate the LIF-equivalent somatic
        # current or the full model
        _A = self._Asom if somatic else self._A
        _Aconst = self._Asom_const if somatic else self._Aconst
        _B = self._Bsom if somatic else self._B
        _Bconst = self._Bsom_const if somatic else self._Bconst

        # Compute a new A, Aconst
        A = _A * (self._WA @ ws)[None, :]
        Aconst = _Aconst + self._WAconst @ ws

        # Compute a new B, pass Bconst along
        B = _B * (self._WB @ ws)[None, :]
        Bconst = np.copy(_Bconst)

        # Pass C along as well
        C = np.copy(self._C)

        return A, Aconst, B, Bconst, C

    def solver_model(self, ws=None, somatic=True):
        """
        Returns the matrices MA, MAconst, MB, MBconst, MC. The matrices must be
        passed to the solver routines.

        In particular, the equation

        J = diag(MAconst + MA @ x) @ v + MB @ x + MBconst

        computes the external currents flowing into each compartment (where
        static conductance/current channels count as "external" as well).

        Parameters
        ----------

        ws: Model parameters. If set, the model matrices will be adapted to take
        the given model weights into account.
        """

        # Fetch the number of compartments and input channels
        n = self.n_compartments
        m = self.n_inputs

        # Fetch the weight-adjusted dynamics model
        MA, MAconst, MB, MBconst, MC = self.dynamics_model(ws=ws, somatic=somatic)

        j = 0 # Input index
        for i in range(n):
            # Fetch the membrance capacitance for this compartment
            Cm = self._compartments[i].Cm

            # Remove current/conductance terms steming from the coupling
            #MAconst[i] += np.sum(self._C[i])

            # Undo the multiplication with CmInv
            MA[i] *= Cm
            MB[i] *= Cm
            MAconst[i] *= Cm
            MBconst[i] *= Cm
            MC[i] *= Cm

        return MA, MAconst, MB, MBconst, MC


    def current_graph(self):
        """
        Returns an affine transformation which allows to compute the current
        flowing through the connections of a multicompartment neuron model given
        the individual compartment potentials. It holds

        j = D * v + e

        where D, e are returned from this function, v is a vector of membrane
        potentials for each compartment, and j is a vector of compartments for
        each connection. This function furthermore returns a list of tuples that
        maps from index in the j-vector to the associated connection pair.

        Connections may include "self-connections" between nodes, which
        correspond to currents flowing in/out of this compartment due to static
        current- or conductance-based channels.
        """
        # Variables storing the number of compartments and maximum number of
        # pairs
        n = self.n_compartments
        m = n * (n + 1) // 2

        # "pairs" is used to provide feedback as to which vector entry
        # corresponds to which current. "rows" selects non-zero rows from the
        # result matrix D and vector e
        pairs, rows = [], []
        D = np.zeros((m, n))
        e = np.zeros((m))

        # Current from/to compartment indices
        ci, cj = 0, 0
        for i in range(m):
            Cm = self._compartments[ci].Cm
            if ci == cj:
                sel = list(range(0, ci)) + list(range(ci + 1, n))
                D[i, ci] = Cm * (self._Aconst[ci] + np.sum(self._C[ci, sel]))
                e[i] = Cm * (self._Bconst[ci])
                rows.append(i)
                pairs.append((ci, cj))
            elif self._C[ci, cj] != 0.0:
                D[i, ci] =  Cm * self._C[ci, cj]
                D[i, cj] = -Cm * self._C[ci, cj]
                rows.append(i)
                pairs.append((ci, cj))
            cj += 1
            if cj >= n:
                ci = ci + 1
                cj = ci
        return D[rows], e[rows], pairs

    @staticmethod
    def clamp_current_graph_potential(D, e, i, v):
        """
        Given the current-graph described by D, e returns a new current graph
        D', e' where the i-th compartment has been clamped to the given voltage
        v.
        """
        m, n = D.shape
        Dp, ep = np.copy(D), np.copy(e)
        sel = list(range(0, i)) + list(range(i + 1, n))
        for j in range(m):
            ep[j] += D[j, i] * v
            Dp[j, i] = 0.0
        return Dp[:, sel], ep

    def vEq_extreme(self):
        """
        For each compartment, computes the maximum and minimum potential that
        these compartments are ever going to have. Note that for neurons with
        current source inputs this is typically unbounded.
        """
        vMin =  np.ones(self.n_compartments) * np.inf
        vMax = -np.ones(self.n_compartments) * np.inf

        # Fetch the dynamical system description for zero input and compute vEq
        A, b = self.A(np.zeros(self.n_inputs)), self.b(np.zeros(self.n_inputs))
        vEq = -np.linalg.solve(A, b)
        vMin = np.minimum(vMin, vEq)
        vMax = np.maximum(vMax, vEq)

        for m in range(0, self.n_compartments):
            sel = list(range(0, m)) + list(range(m + 1, self.n_compartments))
            compartment = self._compartments[m]
            for channel in compartment.channels:
                if channel.type == "cond" and not channel.is_static():
                    Erev = channel.Erev
                    A_red, b_red, _ = self.reduce_system_general(A, b, m, Erev)
                    vMin[m] = np.min((Erev, vMin[m]))
                    vMax[m] = np.max((Erev, vMax[m]))
                    if not np.all(A_red == 0.0):
                        vEq = -np.linalg.solve(A_red, b_red)
                        vMin[sel] = np.minimum(vMin[sel], vEq)
                        vMax[sel] = np.maximum(vMax[sel], vEq)
        return vMin, vMax

    @staticmethod
    def make_noise(tau, rate, T=1.0, dt=1e-4, random=None):
        import scipy.signal as signal

        if random is None:
            random = np.random

        n = int(T / dt + 1e-9)
        xs = np.zeros(n, dtype=np.float64)
        f0 = 1.0 - dt / tau
        f1 = 1.0 / (tau * rate)

        # Create the unfiltered spike train
        curT = 0
        while True:
            curT += random.exponential(1.0 / rate)
            i = int(curT / dt)
            if i < n:
                xs[i] = f1 * random.uniform(0.0, 2.0)
            else:
                break

        # Filter the spike train
        return signal.lfilter([1.0], [1.0, -f0], xs)

    def simulator(self,
                  dt=1e-4,
                  ss=1,
                  record_out=True,
                  record_isom=False,
                  record_voltages=False,
                  record_spike_times=False,
                  record_in_refrac=False):
        """
        Returns a class with methods "simulate" and "simulate_poisson" that can
        be used to simulate the model neuron while recording the specified
        modalities. Per default returns the output over time, i.e. a discretised
        sum of delta functions, as well as the final neuron state.

        Parameters
        ----------

        dt: timestep in seconds. Per default, a one millisecond timestep is
        used. This should be sufficient in most cases, since the system is
        solved using an implicit Euler method and spike times are computed
        exactly.

        record_isom: If true, additionally records the current flowing into the
        soma at any point in time. Note that this does not include the

        record_voltages: If true, additionally records the voltage in each
        compartment.

        record_spike_times: If true, records the precise spike times in
        addition to returning the discretised sum of delta functions (with a
        resolution restricted to dt). Result will be a list of spike times for
        each output spikes. This feature allows precise spike frequence
        measurements if the neuron is firing regularly.

        record_in_refrac: If true, records the refractory state of the neuron.
        This is an array of boolean flags, indicating whether the neuron was in
        the refractory period at the corresponding timestep. This data is useful
        to mask out currents flowing while the neuron is in the refractory
        period.
        """

        from .multi_compartment_lif_cpp import compile_simulator_cpp, supports_cpp

        if not supports_cpp():
            raise Exception("Cannot create the C++ simulator backend. "
              +  "Make sure a recent version of g++ as well as the Eigen C++ "
              +  "library are installed on your system.")

        return compile_simulator_cpp(self,
                    dt=dt,
                    ss=ss,
                    record_out=record_out,
                    record_isom=record_isom,
                    record_voltages=record_voltages,
                    record_spike_times=record_spike_times,
                    record_in_refrac=record_in_refrac)

    def to_svg(self):
        """
        Returns a string containing a diagram representation of the neuron
        model graph.
        """
        import subprocess
        from .multi_compartment_lif_graphviz import generate_neuron_graph

        # Create the dot file
        buf = io.StringIO()
        generate_neuron_graph(self._compartments, self._C, buf)

        # Run the subprocess
        res = subprocess.run(
            ["neato", "-Tsvg"],
            input=buf.getvalue().encode("utf-8"),
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

