#!/usr/bin/env python3

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--T',
                    type=float,
                    help='Simulation time in seconds',
                    default=100.0)
parser.add_argument('--J',
                    type=float,
                    help='Maximum ramp current in nA',
                    default=2.0)
parser.add_argument('--neuron',
                    type=str,
                    help='Neuron type; one off "HH", "LIF"',
                    default="HH")
parser.add_argument('--tar',
                    type=str,
                    help='Target filename in the data directory',
                    default='hodgkin_huxley_ramp.npz')
parser.add_argument('--no_skew',
                    action='store_true',
                    default=False)

args = parser.parse_args()

###############################################################################
# Neuron Model                                                                #
###############################################################################

from brian2 import *

T = args.T * second
max_I = args.J * nA
point_area = 20000 * umetre**2
Cm = 1 * ufarad * cm**-2 * point_area
gL = 5e-5 * siemens * cm**-2 * point_area
EL = -65 * mV

print(Cm, gL)

if args.no_skew:
    ramp = '''(t / T) * max_I'''
else:
    ramp = '''(exp(2.0 * t / T) - 1) / (exp(2.0) - 1) * max_I'''

if args.neuron.lower() == "hh":
    # HH model adapted from the following Brian example:
    # https://brian2.readthedocs.io/en/stable/examples/IF_curve_Hodgkin_Huxley.html

    # Parameters
    EK = -90 * mV
    ENa = 50 * mV
    g_na = 100 * msiemens * cm**-2 * point_area
    g_kd = 30 * msiemens * cm**-2 * point_area
    VT = -63 * mV

    # HH model with exponentially ramping up input current
    eqns = '''
    dv/dt = (gL*(EL-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I)/Cm : volt
    dm/dt = 0.32*(mV**-1)*4*mV/exprel((13.*mV-v+VT)/(4*mV))/ms*(1-m)-0.28*(mV**-1)*5*mV/exprel((v-VT-40.*mV)/(5*mV))/ms*m : 1
    dn/dt = 0.032*(mV**-1)*5*mV/exprel((15.*mV-v+VT)/(5*mV))/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
    dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
    I = ''' + ramp + ''' : amp
    '''
    input_neuron = NeuronGroup(N=1,
                               model=eqns,
                               threshold='v > -20*mV',
                               refractory='v > -40*mV',
                               method='exponential_euler')
elif args.neuron.lower() == "lif":
    tauRef = 2 * ms
    vTh = -55 * mV
    vReset = -85 * mV
    # HH model with exponentially ramping up input current
    eqns = '''
    dv/dt = (gL*(EL-v) + I)/Cm : volt (unless refractory)
    I = ''' + ramp + ''' : amp
    '''
    reset = '''
        v = vReset
    '''
    input_neuron = NeuronGroup(N=1,
                               model=eqns,
                               reset=reset,
                               threshold='v > vTh',
                               refractory=tauRef,
                               method='exponential_euler')
else:
    raise RuntimeError("Unknown neuron type")

input_neuron.v = EL  # Reset the neuron

input_voltage_monitor = StateMonitor(input_neuron,
                                     'v',
                                     record=True,
                                     dt=0.1 * ms)
input_current_monitor = StateMonitor(input_neuron,
                                     'I',
                                     record=True,
                                     dt=0.1 * ms)
spike_mon = SpikeMonitor(input_neuron, record=True)

defaultclock.dt = 10 * us
run(T, report='stdout')

ts = input_voltage_monitor.t / ms
us = input_voltage_monitor.v[0] / mV
Js = input_current_monitor.I[0] / nA
spikes = spike_mon.t / ms

import os

np.savez(os.path.join('data', args.tar),
         ts=ts,
         us=us,
         Js=Js,
         spikes=spikes)

