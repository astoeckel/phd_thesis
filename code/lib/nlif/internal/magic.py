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

# Try to import the traceback module. This is used for automagic component
# label extraction
try:
    import traceback
except ModuleNotFoundError:
    traceback = None

Number = (int, float)

def variables(skip=0, flt=None):
    """
    Returns a map from object id to the corresponding variable names over all
    stack frames. The innermost name of an object is returned.

    skip: Number of stack frames to skip.
    flt:  Optional filter function that is applied to each object before its
          name is recorded.
    """

    # If we have the "traceback" module, dump the stack and list all variables
    # matching the filter
    names = {}
    if not traceback is None:
        for i, (frame, _) in enumerate(traceback.walk_stack(None)):
            if i < skip:
                continue
            for key, value in frame.f_locals.items():
                if (flt is None) or (flt(value)):
                    if (not (value in names)):  # innermost name
                        names[value] = key

    return names


def scalar(x):
    """
    Converts the given variable "x" into either a floating point number or an
    integer.
    """
    if (isinstance(x, np.ndarray) and (x.size == 1)):
        return float(next(iter(a.flat)))
    elif (isinstance(x, int)):
        return x
    else:
        try:
            return float(x)
        except (ValueError, TypeError):
            raise ValueError("Expected scalar, but got {}".format(repr(x)))

