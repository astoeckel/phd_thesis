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

# Some of the code in this file is adapted from pykinsim (see https://github.com/astoeckel/pykinsim)

from collections import deque, OrderedDict
import threading

from .internal import magic
from .internal import lif_utils
from .internal.errors import *

# Initialize thread-local storage -- this is needed to properly implement the
# "with" magic of the Neuron and Compartment object
_thread_local = threading.local()

DEFAULT_G_C = 50.0e-9
DEFAULT_E_E = 0.0e-3
DEFAULT_E_I = -75.0e-3
DEFAULT_G_L = 50.0e-9
DEFAULT_E_L = -65.0e-3
DEFAULT_C_M = 1.0e-9
DEFAULT_V_TH = -50.0e-3
DEFAULT_V_RESET = -65.0e-3
DEFAULT_V_SPIKE = 20.0e-3
DEFAULT_TAU_REF = 2.0e-3
DEFAULT_TAU_SPIKE = 1.0e-3


class Labeled:
    """
    A "Labeled" object is simply an object with a "label" property. This label
    will be printed as the representation of the object. libnlif automagically™
    deduces object labels from variable names whenever the scope of a
    "with Neuron..." statement is left.
    """
    def __init__(self, label=None):
        self.label = label

    def __repr__(self):
        fmt = {
            "label": str(self.label),
            "class": self.__class__.__name__,
            "addr": id(self),
        }
        if self.label is None:
            return "<{class} @ 0x{addr:02x}>".format(**fmt)
        else:
            return "<{label} ({class} @ 0x{addr:02x})>".format(**fmt)

    def coerce(self):
        if not self.label is None:
            self.label = str(self.label)
        return self


class Neuron(Labeled):
    def __init__(self, label=None):
        """
        Constructor of the Neuron object.

        Parameters
        ==========

        label: An (optional) name describing this object.
        """
        super().__init__(label)

        # We use an OrderedDict here to maintain the order of the inputs,
        # although this technically isn't necessary in newer versions of Python.
        # Also note that we use this container as a set, that is, we only use
        # the keys, and simply assign "True" to every key that should be in
        # the set.
        self._objs = OrderedDict()

    def __enter__(self):
        """
        Marks this Neuron instance as the "currently active" neuron.
        Newly created Compartment, Connection, and Channel instances are
        automatically inserted into the channel.
        """

        if not hasattr(_thread_local, 'active_neuron'):
            _thread_local.active_neuron = []
        _thread_local.active_neuron.append(self)

        return self

    def __exit__(self, type, value, traceback):
        _thread_local.active_neuron.pop()

        # Try to assign labels to objects that do not have labels yet
        names = magic.variables(skip=1,
                                flt=lambda obj: isinstance(obj, Labeled))
        if (self in names) and (self.label is None):
            self.label = names[self]
        for obj in self._objs:
            if (obj in names) and (obj.label is None):
                obj.label = names[obj]

    def __contains__(self, key):
        return key in self._objs

    def graph(self):
        """
        Constructs and validates the graph underpinning this neuron instance.
        Returns three objects: `compartments`, `connections`, and `channels`.

        `compartments` is a list of `Compartment` objects. The soma will always
        be the first compartment. The other compartments are listed in their
        construction order.

        `connections` is a map between tuples of compartment indices and lists
        of `Connection` objects. The indices in the tuples (i, j) are sorted such
        that i < j. This is possible because all connections are unidirectional.
        Multiple connections are supported.

        `channels` is a list of lists. The outer list corresponds to each
        compartment, the inner lists contain the channels belonging to each
        compartment.
        """

        # List all compartments, make sure that the somatic compartment comes
        # first
        _compartments_with_indices = list(
            enumerate((x for x in self._objs if isinstance(x, Compartment))))
        compartments = [
            x[1] for x in sorted(_compartments_with_indices,
                                 key=lambda x: (not x[1].is_soma, x[0]))
        ]

        # Make sure that there is at least one compartment and that the first
        # compartment is the only compartment
        if len(compartments) == 0:
            raise NoCompartmentsError(
                "Neuron object must contain at least one compartment",
                obj=self)
        if not isinstance(compartments[0], Soma):
            raise NoSomaticCompartmentError(
                "Neuron object must contain exactly one Soma object", obj=self)
        for i, compartment in enumerate(compartments):
            if (i == 0) != isinstance(compartment, Soma):
                raise MultipleSomaticCompartmentsError(
                    "Neuron object must possess exactly one Soma object",
                    obj=self)

        # Assemble some helper maps
        _compartment_to_idx_map = {}
        _idx_to_compartment_map = {}
        for i, compartment in enumerate(compartments):
            _compartment_to_idx_map[compartment] = i
            _idx_to_compartment_map[i] = compartment

        # Assemble the map of connections
        connections = {}
        for connection in (x for x in self._objs if isinstance(x, Connection)):
            # Make sure that the connection targets are valid
            if not connection.comp1 in _compartment_to_idx_map:
                raise ConnectivityError(
                    f"Compartment {connection.comp1} in connection {connection} is not part {self}"
                )
            if not connection.comp2 in _compartment_to_idx_map:
                raise ConnectivityError(
                    f"Compartment {connection.comp2} in connection {connection} not part of {self}"
                )

            # Fetch the connection indices
            i = _compartment_to_idx_map[connection.comp1]
            j = _compartment_to_idx_map[connection.comp2]

            # Disallow self-connections
            if i == j:
                raise SelfConnectionError(f"Self-connections are not allowed")

            # Add the connection to the connections map
            key = (min(i, j), max(i, j))
            if not key in connections:
                connections[key] = []
            connections[key].append(connection)

        # Assemble all channels, make sure that no compartment has a channel
        # it should not have
        channels = [[] for _ in range(len(compartments))]
        _input_channel_set, _channel_set = set(), set()
        n_inputs = 0
        for channel in (x for x in self._objs if isinstance(x, Channel)):
            # Add the channel to the correct spot in the channels list
            if channel.is_input:
                _input_channel_set.add(channel)
                n_inputs += 1
            if not channel.compartment in self:
                raise ChannelConnectivityError(
                    f"Channel {channel} connected to an invalid compartment")
            i = _compartment_to_idx_map[channel.compartment]
            channels[i].append(channel)
            _channel_set.add(channel)

        # Make sure that there is one channel object for each input channel
        if len(_input_channel_set) != n_inputs:
            raise ChannelConnectivityError(
                f"Each input channel instance must be used exactly once")

        # Make sure that the channel actually belongs to one of the
        # compartments
        _channel_set_ref = set()
        for compartment in compartments:
            for channel in compartment.channels:
                _channel_set_ref.add(channel)
                if not channel in _channel_set:
                    raise ChannelConnectivityError(
                        f"Compartment {compartment} contains channel that does not belong to the neuron"
                    )

        # For a final time ensure that all the channels attached to the Neuron
        # instance and the channels attached to the compartments are the same
        if _channel_set_ref != _channel_set:
            raise ChannelConnectivityError(
                "Channels are connected incorrectly")

        # Ensure that all compartments can be reached from the soma
        visited = set()
        queue = deque((compartments[0], ))
        while len(queue) > 0:
            cur = queue.popleft()
            visited.add(cur)
            i = _compartment_to_idx_map[cur]
            for ii, jj in connections:
                if ii == i:
                    if not _idx_to_compartment_map[jj] in visited:
                        queue.append(_idx_to_compartment_map[jj])
                if jj == i:
                    if not _idx_to_compartment_map[ii] in visited:
                        queue.append(_idx_to_compartment_map[ii])
        if len(visited) != len(compartments):
            raise DisconnectedNeuronError(
                "Connectivity graph is not a single connected component")

        return compartments, connections, channels

    def assemble(self):
        from .assembled_neuron import AssembledNeuron
        return AssembledNeuron(self)

    def coerce(self):
        # Call the inherited coerce function
        super().coerce()

        # Make sure that all child objects are of the correct type
        for obj in self._objs:
            if not isinstance(obj,
                              (Compartment, Connection, CurChan, CondChan)):
                raise ValidationError(
                    f"{obj} is not a Compartment, Connection, CurChan or CondChan",
                    obj=self)
            obj.coerce()

        return self

    def to_dot(self):
        return self.assemble().to_dot()

    def to_svg(self):
        return self.assemble().to_svg()

    def _repr_html_(self):
        return self.assemble()._repr_html_()


class NeuronPart(Labeled):
    """
    The NeuronPart base-class is used for all objects that can be owened by a
    Neuron instance.
    """
    def __init__(self, label=None, parent=None):
        # Pass the label to the Labeled super-constructor
        super().__init__(label)

        # If no parent object has been given, make sure the constructor has been
        # called while a Chain object is being constructed. Otherwise just use
        # the given parent.
        if parent is None:
            if len(getattr(_thread_local, 'active_neuron', [])) == 0:
                raise RuntimeError(
                    "{} instances must be constructed within an active neuron".
                    format(self.__class__.__name__))
            self._parent = _thread_local.active_neuron[-1]
        else:
            self._parent = parent

        # Add this object the parent object set
        self._parent._objs[self] = True

    def coerce(self):
        """
        The coerce method checks and normalises all parameters; returns a
        reference at this, or a copied instance.
        """
        # Call the inherited coerce function
        super().coerce()

        # Make sure the parent object is correct
        if (self._parent is None) or (not self in self._parent):
            raise ValidationError("Invalid parent", obj=self)

        return self

    @property
    def parent(self):
        return self._parent


class Connection(NeuronPart):
    """
    The connection class is used to connect two Compartment instances.
    Connections are bi-directional. The only connections that are not allowed
    are self-connections, i.e., a compartment cannot be connected to itself.
    """
    def __init__(self, comp1, comp2, g_c=DEFAULT_G_C, label=None, parent=None):
        """
        Creates a new Connection between two compartments of a neuron.
        Note that these connections are bidirectional. Correspondingly, it does
        not matter which compartment is provided as a first and as a second
        argument.

        Parameters
        ==========

        comp1, comp2: The two Compartment instances to connect.

        gc:   The coupling conductance. Must be a non-negative number.

        label: Name of the compartment. If None, the name will be automagically
              determined from the variable name this instance has been assigned
              to.
        """
        super().__init__(label, parent)

        self.comp1 = comp1
        self.comp2 = comp2
        self.g_c = g_c

    def coerce(self):
        # Call the inherited implementation of "coerce"
        super().coerce()

        # Make sure that the coupling conductance is a scalar
        self.g_c = magic.scalar(self.g_c)
        if (type(self.g_c) is float) and (self.g_c < 0):
            raise ValidationError("Coupling conductance must be non-negative",
                                  obj=self)

        # Make sure that the two compartments are Compartment instances and
        # point at different compartments.
        if not (isinstance(self.comp1, Compartment)
                and isinstance(self.comp2, Compartment)):
            raise ValidationError(
                "Connections can only connect instances of the class Compartment",
                obj=self)

        if self.comp1 is self.comp2:
            raise ValidationError("Self-connections are not allowed", obj=self)

        return self


class Compartment(NeuronPart):
    """
    A passive compartment characterised by its membrane capacitance.
    """
    def __init__(self, C_m=DEFAULT_C_M, label=None, parent=None):
        """
        Creates a new passive compartment.

        Parameters
        ==========

        C_m:  A positive number describing the membrane capacitance in Farad.
        """

        # Call the inherited constructor
        super().__init__(label, parent)

        # Copy the given membrane capacitance
        self.C_m = C_m

        # Initialise the set of objects belonging to this compartment
        self._objs = OrderedDict()

    def __enter__(self):
        """
        Marks this Compartment instance as the "currently active" compartment.
        Newly created Channel objects are automatically assigned to this
        compartment.
        """

        if not hasattr(_thread_local, 'active_compartment'):
            _thread_local.active_compartment = []
        _thread_local.active_compartment.append(self)

        return self

    def __exit__(self, type, value, traceback):
        """
        Removes this compartment from the active compartment stack.
        """
        _thread_local.active_compartment.pop()

        # Try to assign labels to objects that do not have labels yet
        names = magic.variables(skip=1,
                                flt=lambda obj: isinstance(obj, Labeled))
        if (self in names) and (self.label is None):
            self.label = names[self]
        for obj in self._objs:
            if (obj in names) and (obj.label is None):
                obj.label = names[obj]

    def __contains__(self, key):
        return key in self._objs

    @property
    def channels(self):
        return self._objs.keys()

    def coerce(self):
        # Call the inherited implementation of "coerce"
        super().coerce()

        # Make sure that the membrane capacitance is positive
        self.C_m = magic.scalar(self.C_m)
        if isinstance(self.C_m, magic.Number) and self.C_m <= 0:
            raise ValidationError(
                "Membrane capacitance must be strictly positive", obj=self)

        # Make sure that all child objects are either CurChan or CondChan
        # instances
        for obj in self._objs:
            if not isinstance(obj, (CurChan, CondChan)):
                raise ValidationError(f"{obj} is not a CurChan or CondChan",
                                      obj=self)

        return self

    @property
    def is_soma(self):
        return False


class Soma(Compartment):
    def __init__(self,
                 C_m=1e-9,
                 v_th=-50e-3,
                 v_reset=-65e-3,
                 v_spike=20e-3,
                 tau_ref=2e-3,
                 tau_spike=1e-3,
                 label=None,
                 parent=None):
        """
        Creates a new active compartment. In n-LIF neurons, there can only be
        a single active compartment and this compartment must be the somatic
        compartment.

        Parameters
        ==========

        C_m:  A positive number containing the membrane capacitance in Farad.

        v_th: The threshold potential in volt. Must be larger than the reset
              potential.

        v_reset: The reset potential in volt. Must be smaller than the threshold
              potential.

        v_spike: The spike potential. Must be larger than the threshold
              potential.

        tau_ref: The length of the refractory period. If zero, this neuron will
              not have a refractory period.

        tau_spike: The length of the spike period. If zero, this neuron will
              not have a spike period.

        label: The name of the neuron; will be automagically determined if not
              given.

        parent: The parent Neuron object. Will be automagically determined if
              not given.
        """

        # Call the inherited constructor
        super().__init__(C_m, label, parent)

        # Copy all given arguments
        self.v_th = v_th
        self.v_reset = v_reset
        self.v_spike = v_spike
        self.tau_ref = tau_ref
        self.tau_spike = tau_spike

    def coerce(self):
        # Call the inherited coerce implementation
        super().coerce()

        # Convert all parameter values into scalars
        self.v_th = magic.scalar(self.v_th)
        self.v_reset = magic.scalar(self.v_reset)
        self.v_spike = magic.scalar(self.v_spike)
        self.tau_ref = magic.scalar(self.tau_ref)
        self.tau_spike = magic.scalar(self.tau_spike)

        # Make sure that some basic constraints are met. Note that all checks
        # are necessary (transitivity does not necessarily hold)
        if isinstance(self.v_th, magic.Number) and isinstance(
                self.v_reset, magic.Number):
            if self.v_th <= self.v_reset:
                raise ValidationError(
                    "Threshold potential must be strictly larger than the reset potential",
                    obj=self)
        if isinstance(self.v_th, magic.Number) and isinstance(
                self.v_spike, magic.Number):
            if self.v_spike <= self.v_th:
                raise ValidationError(
                    "Spike potential must be strictly larger than the threshold potential",
                    obj=self)
        if isinstance(self.v_reset, magic.Number) and isinstance(
                self.v_spike, magic.Number):
            if (self.v_spike <= self.v_reset):
                raise ValidationError(
                    "Spike potential must be strictly larger than the reset potential",
                    obj=self)

        # Make sure that the time-constants are non-negative
        if isinstance(self.tau_ref, magic.Number) and (self.tau_ref < 0):
            raise ValidationError("Refractory period must be non-negative",
                                  obj=self)
        if isinstance(self.tau_spike, magic.Number) and (self.tau_spike < 0):
            raise ValidationError("Spike period must be non-negative",
                                  obj=self)

        return self

    @property
    def is_soma(self):
        return True

    def is_excitatory(self, v_th=DEFAULT_V_TH):
        return not self.is_inhibitory(v_th)


class Channel(NeuronPart):
    def __init__(self,
                 intrinsic=None,
                 compartment=None,
                 label=None,
                 parent=None):
        """
        Creates a new Channel instance. Do not use this class directly, use
        CondChan or CurChan instead.

        Parameters
        ==========

        compartment: Compartment this channel belongs to. If `None` is given,
            the compartment is automagically determined from the surrounding
            `with Compartment()...` statement.

        intrinsic: flag that indicates whether this channel is included
              in the somatic input current calculation.
              
              If explicitly set to `True`, this channel is assumed to be
              already included in the nonlinearity.
              
              If explicitly set to `False`, this channel is assumed to not be
              part of the somatic nonlinearity.
              
              The default value of `None` applies a heuristic. If this channel
              is part of the somatic compartment, and is a conductance-based
              channel, then this channel is assumed to be a leak channel that
              is already accounted for in the somatic nonlinearity.
        """

        # If no compartment object has been given, try to automatically fetch
        # the compartment object form the thread local compartment stack.
        if compartment is None:
            if len(getattr(_thread_local, 'active_compartment', [])) == 0:
                raise RuntimeError(
                    "{} instances must be constructed within an active neuron".
                    format(self.__class__.__name__))
            self._compartment = _thread_local.active_compartment[-1]
        else:
            self._compartment = compartment

        # Add this object the parent object set
        self._compartment._objs[self] = True

        # Call the inherited constructor and copy given arguments
        super().__init__(label, parent)
        self.intrinsic = intrinsic

    def coerce(self):
        super().coerce()

        if not isinstance(self._compartment, Compartment):
            raise ValidationError(
                "Channel must belong to a compartment object", obj=self)

        if not any((self.intrinsic is x for x in (None, False, True))):
            raise ValidationError(
                "The intrinsic flag must either be None, False, True",
                obj=self)

        if (self.intrinsic is True) and self.is_input:
            raise ValidationError(
                "A channel cannot both be an input and an intrinsic!",
                obj=self)

        return self

    @property
    def compartment(self):
        return self._compartment

    @property
    def is_intrinsic(self):
        """
        Returns a boolean flag indicating whether this channel should be counted
        as a somatic current.
        """
        if not self.intrinsic is None:
            return bool(self.intrinsic)
        return (self._compartment.is_soma and self.is_static
                and self.type == "cond")

    @property
    def is_input(self):
        return not self.is_static


class CondChan(Channel):
    """
    The ConductanceChannel class represents a conductance-based channel inside
    a neuron compartment. The channel can either be static, in which case it has
    a constant conductance, or an input channel, in which case the conductance
    is influenced by pre-synaptic spikes.
    """
    def __init__(self,
                 E_rev,
                 g=None,
                 intrinsic=None,
                 compartment=None,
                 label=None,
                 parent=None):
        """
        Constructor of the ConductanceChannel class.

        Parameters
        ==========

        E_rev: float describing the reversal potential of the ConductanceChannel
              in millivolts.

        g: static conductance value. If "None" is specified as a conductance
              value, the channel will be treated as an input channel.
        """
        # Call the inherited constructor
        super().__init__(intrinsic, compartment, label, parent)

        # Copy the given arguments
        self.E_rev = E_rev
        self.g = g

    def coerce(self):
        # Call the inherited coerce function
        super().coerce()

        # Make sure that the reversal potential is a float
        self.E_rev = magic.scalar(self.E_rev)

        # Make sure that g is either None or a float
        self.g = None if self.g is None else magic.scalar(self.g)
        if isinstance(self.g, float) and (self.g < 0):
            raise ValidationError(
                "Static channel conductance must be non-negative", obj=self)

        return self

    @property
    def type(self):
        return "cond"

    @property
    def is_static(self):
        return not self.g is None

    def is_inhibitory(self, v_th=DEFAULT_V_TH):
        return self.E_rev < v_th


class CurChan(Channel):
    """
    The CurChan class represents a current-based channel within a compartment.
    Current channels can either be static, in which case the channel provides
    a bias to the neuron, or an input channel, in which case the current
    can be fed-in by the user.
    """
    def __init__(self,
                 mul=1,
                 J=None,
                 intrinsic=None,
                 compartment=None,
                 label=None,
                 parent=None):
        """
        Constructor of the CurrentChannel class.

        Parameters
        ==========

        mul: multiplicative factor by which input currents -- or the static
             current -- are multiplied. A factor of one corresponds to an
             excitatory synaptic channel, a factor of minus one corresponds to
             an inhibitory synaptic channel.

        J:   If not set to `None`, corresponds to a static current induced by
             this channel. The current must be given in ampere.
        """
        # Call the inherited constructor
        super().__init__(intrinsic, compartment, label, parent)

        # Copy the given arguments
        self.mul = mul
        self.J = J

    def coerce(self):
        # Call the inherited coerce function
        super().coerce()

        # Make sure that all parameters are floats
        self.mul = magic.scalar(self.mul)
        self.J = None if self.J is None else magic.scalar(self.J)

    @property
    def type(self):
        return "cur"

    @property
    def is_static(self):
        return not self.J is None

    def is_inhibitory(self, v_th=DEFAULT_V_TH):
        return ((self.mul < 0) if (self.J is None) else
                (self.mul * self.J < 0))


# Aliases
CurrentChannel = CurChan
ConductanceChannel = CondChan


class LIF(Neuron):
    def __init__(self,
                 mul_E=1.0,
                 mul_I=-1.0,
                 g_L=DEFAULT_G_L,
                 E_L=DEFAULT_E_L,
                 C_m=DEFAULT_C_M,
                 v_th=DEFAULT_V_TH,
                 v_reset=DEFAULT_V_RESET,
                 v_spike=DEFAULT_V_SPIKE,
                 tau_ref=DEFAULT_TAU_REF,
                 tau_spike=DEFAULT_TAU_SPIKE,
                 label=None):
        super().__init__(label)

        with self:
            with Soma(C_m=C_m,
                      v_th=v_th,
                      v_reset=v_reset,
                      v_spike=v_spike,
                      tau_ref=tau_ref,
                      tau_spike=tau_spike) as self.soma:
                self.g_L = CondChan(E_rev=E_L, g=g_L)
                self.J_E = CurChan(mul=mul_E)
                self.J_I = CurChan(mul=mul_I)


class LIFCond(Neuron):
    def __init__(self,
                 E_E=DEFAULT_E_E,
                 E_I=DEFAULT_E_I,
                 g_L=DEFAULT_G_L,
                 E_L=DEFAULT_E_L,
                 C_m=DEFAULT_C_M,
                 v_th=DEFAULT_V_TH,
                 v_reset=DEFAULT_V_RESET,
                 v_spike=DEFAULT_V_SPIKE,
                 tau_ref=DEFAULT_TAU_REF,
                 tau_spike=DEFAULT_TAU_SPIKE,
                 label=None):
        super().__init__(label)

        with self:
            with Soma(C_m=C_m,
                      v_th=v_th,
                      v_reset=v_reset,
                      v_spike=v_spike,
                      tau_ref=tau_ref,
                      tau_spike=tau_spike) as self.soma:
                self.g_L = CondChan(E_rev=E_L, g=g_L)
                self.g_E = CondChan(E_rev=E_E)
                self.g_I = CondChan(E_rev=E_I)


class TwoCompLIFCond(Neuron):
    def __init__(self,
                 g_c=DEFAULT_G_C,
                 E_E=DEFAULT_E_E,
                 E_I=DEFAULT_E_I,
                 g_L=DEFAULT_G_L,
                 E_L=DEFAULT_E_L,
                 C_m=DEFAULT_C_M,
                 v_th=DEFAULT_V_TH,
                 v_reset=DEFAULT_V_RESET,
                 v_spike=DEFAULT_V_SPIKE,
                 tau_ref=DEFAULT_TAU_REF,
                 tau_spike=DEFAULT_TAU_SPIKE,
                 label=None):
        super().__init__(label)

        with self:
            with Soma(C_m=C_m,
                      v_th=v_th,
                      v_reset=v_reset,
                      v_spike=v_spike,
                      tau_ref=tau_ref,
                      tau_spike=tau_spike) as self.soma:
                self.g_L = CondChan(E_rev=E_L, g=g_L)

            with Compartment(C_m=C_m) as self.dendrites:
                self.g_L = CondChan(E_rev=E_L, g=g_L)
                self.g_E = CondChan(E_rev=E_E)
                self.g_I = CondChan(E_rev=E_I)

            Connection(self.soma, self.dendrites, g_c=g_c)


class ThreeCompLIFCond(Neuron):
    def __init__(self,
                 g_c1=DEFAULT_G_C,
                 g_c2=DEFAULT_G_C,
                 E_E=DEFAULT_E_E,
                 E_I=DEFAULT_E_I,
                 g_L=DEFAULT_G_L,
                 E_L=DEFAULT_E_L,
                 C_m=DEFAULT_C_M,
                 v_th=DEFAULT_V_TH,
                 v_reset=DEFAULT_V_RESET,
                 v_spike=DEFAULT_V_SPIKE,
                 tau_ref=DEFAULT_TAU_REF,
                 tau_spike=DEFAULT_TAU_SPIKE,
                 label=None):
        super().__init__(label)

        with self:
            with Soma(C_m=C_m,
                      v_th=v_th,
                      v_reset=v_reset,
                      v_spike=v_spike,
                      tau_ref=tau_ref,
                      tau_spike=tau_spike) as self.soma:
                self.g_L = CondChan(E_rev=E_L, g=g_L)

            with Compartment(C_m=C_m) as self.basal:
                self.g_L = CondChan(E_rev=E_L, g=g_L)
                self.g_E1 = CondChan(E_rev=E_E)
                self.g_I1 = CondChan(E_rev=E_I)

            with Compartment(C_m=C_m) as self.apical:
                self.g_L = CondChan(E_rev=E_L, g=g_L)
                self.g_E2 = CondChan(E_rev=E_E)
                self.g_I2 = CondChan(E_rev=E_I)

            Connection(self.soma, self.basal, g_c=g_c1)
            Connection(self.basal, self.apical, g_c=g_c2)

