#!/usr/bin/env python3

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
#   You should have received a copy of the GNU General Public License
#   along with soft_cond_lif.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np


def _rate(g_tot, e_eq, tau_ref=2e-3, v_th=1.0):
    """
    Function used internally by lif_cond_rate and lif_rate. Takes the total
    conductance and the current equilibrium potential and calculates the
    corresponding spike rate. All parameters may be numpy arrays.

    g_tot: total conductance.
    e_eq: equilibrium potential.
    tau_ref: refractory period in second.
    v_th: threshold potential.
    """

    # Mask out invalid values
    mask = 1 * np.logical_and(e_eq > v_th, g_tot > 0)
    t_spike = -np.log1p(-mask * v_th / (mask * e_eq + (1 - mask))) / g_tot
    return mask / (tau_ref + t_spike)


def lif_cond_rate(gL,
                  gE,
                  gI,
                  e_rev_E=4.33,
                  e_rev_I=-0.33,
                  tau_ref=2e-3,
                  v_th=1.0):
    """
    Calculates the firing rate of a LIF neuron with conductance based synapses.
    Input to the model are conductances gE and gI. All parameters may be numpy
    arrays.

    gL: leak conductance.
    gE: excitatory conductance.
    gI: inhibitory conductance.
    e_rev_E: excitatory synapse reversal potential.
    e_rev_I: inhibitory synapse reversal potential.
    tau_ref: refractory period in second.
    v_th: threshold potential.
    """

    # Calculate the total conductance and the equilibrium potential
    g_tot = gL + gE + gI
    e_eq = (gE * e_rev_E + gI * e_rev_I) / g_tot

    return _rate(g_tot, e_eq, tau_ref, v_th)


def lif_rate(J, tau_ref=2e-3, tau_rc=20e-3):
    """
    Calculates the firing rate of a LIF neuron with current based synapses.
    Input to the model is the current J. All parameters may be numpy arrays.

    J: input current
    tau_ref: refractory period in seconds.
    tau_rc: membrane time constant in seconds.
    """

    mask = 1 * (J > 1)
    t_spike = -np.log1p(-mask * 1.0 / (mask * J + (1 - mask))) * tau_rc
    return mask / (tau_ref + t_spike)


def lif_rate_inv(r, tau_ref=2e-3, tau_rc=20e-3):
    """
    Calculates the firing rate of a LIF neuron with current based synapses.
    Input to the model is the current J. All parameters may be numpy arrays.

    r: input rate
    tau_ref: refractory period in seconds.
    tau_rc: membrane time constant in seconds.
    """
    mask = 1.0 * (r > 1e-6)
    return -mask / (np.exp(
        (r * tau_ref - 1) / ((1.0 - mask) + r * tau_rc)) - 1)


def lif_detailed_rate(J,
                      v_th=-35e-3,
                      v_reset=-80e-3,
                      gL=50e-9,
                      Cm=1e-9,
                      EL=-65e-3,
                      tau_ref=2e-3):
    """
    Calculates the firing rate of a LIF neuron with current-based synapses for
    detailed neuron parameters.
    """
    tau_rc = Cm / gL
    J_th = (v_th - EL) * gL
    valid = 1.0 * (J > J_th)
    t_spike = valid * (tau_rc * np.log1p(-valid * ((v_th - v_reset) * gL) /
                                         (valid * ((EL - v_reset) * gL + J) + (1.0 - valid))))
    return valid / (tau_ref - t_spike)


def lif_detailed_rate_inv(r,
                          v_th=-35e-3,
                          v_reset=-80e-3,
                          gL=50e-9,
                          Cm=1e-9,
                          EL=-65e-3,
                          tau_ref=2e-3):
    """
    Calculates 
    """
    tau_rc = Cm / gL
    valid = 1.0 * (r > 1e-6)
    exp = np.exp((r * tau_ref - 1) / (valid * r * tau_rc + (1.0 - valid)))
    return -valid * (((EL - v_reset) * exp - EL + v_th) * gL) / (exp - 1.0)


def spike_frequency(spikes):
    """
    Computes the spike frequency by taking the median distance between spikes in
    the given input array.

    spikes: array containing the individual spike times in seconds
    """
    n_spikes = len(spikes)
    spikes = np.array(spikes)
    if n_spikes > 2:
        return 1 / np.mean(spikes[1:] - spikes[:-1])
    return 0.0

