#   This file is part of NEF Synaptic Computation
#   (c) Andreas Stöckel 2017, 2018
#
#   NEF Synaptic Computation is free software: you can redistribute it and/or
#   modify it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   NEF Synaptic Computation is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License along with
#   NEF Synaptic Computation.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from .lif_utils import lif_rate_inv

def first_order_tuning_curve_parameters(max_rate, x_intercept, tau_rc=20e-3, tau_ref=2e-3):
    Jmax = lif_rate_inv(max_rate)
    a0 = (1.0 - Jmax * x_intercept) / (1.0 - x_intercept)
    a1 = Jmax - a0
    return np.array((a1, a0))

def second_order_tuning_curve_parameters(max_rate, x_max, spread, tau_rc=20e-3, tau_ref=2e-3):
    Jmax = lif_rate_inv(max_rate)

    scale = 1.0 / (spread ** 2)
    a2 = -(Jmax - 1) * scale
    a1 = 2 * x_max * (Jmax - 1) * scale
    a0 = (Jmax * (spread ** 2 - x_max ** 2) + x_max ** 2) * scale
    return np.array((a2, a1, a0))

def box_tuning_curve_parameters(min, max, n, i):
    w = (max - min) / n
    c = w * (i + 0.5) + min
    return c

def box_tuning_curve(min, max, n, p, x):
    w = (max - min) / n
    norm = 0.5 * (x[:, None] - p[None, :]) / w
    return np.logical_and(norm > -1, norm < 1)