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


class ValidationError(ValueError):
    def __init__(self, msg, obj=None):
        if not obj is None:
            msg = "Error while validating object {}: {}".format(repr(obj), msg)
        super().__init__(msg)


class NoCompartmentsError(ValidationError):
    pass


class NoSomaticCompartmentError(ValidationError):
    pass


class DisconnectedNeuronError(ValidationError):
    pass


class MultipleSomaticCompartmentsError(ValidationError):
    pass

class ConnectivityError(ValidationError):
    pass

class SelfConnectionError(ValidationError):
    pass

class ChannelConnectivityError(ValidationError):
    pass
