#!/usr/bin/env python3

# This file generates the membrane potential traces embedded in figure 2.2

from brian2 import *

###############################################################################
# Neuron Model                                                                #
###############################################################################

# HH model adapted from the following Brian example:
# https://brian2.readthedocs.io/en/stable/examples/IF_curve_Hodgkin_Huxley.html

# Parameters
point_area = 20000 * umetre**2
Cm = 1 * ufarad * cm**-2 * point_area
gl = 5e-5 * siemens * cm**-2 * point_area
El = -65 * mV
EK = -90 * mV
ENa = 50 * mV
Esyn = ENa
g_na = 100 * msiemens * cm**-2 * point_area
g_kd = 30 * msiemens * cm**-2 * point_area
VT = -63 * mV

tau_syn = 5 * ms

# HH model with current input
eqns_hh = Equations('''
dv/dt = (gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I)/Cm : volt
dm/dt = 0.32*(mV**-1)*4*mV/exprel((13.*mV-v+VT)/(4*mV))/ms*(1-m)-0.28*(mV**-1)*5*mV/exprel((v-VT-40.*mV)/(5*mV))/ms*m : 1
dn/dt = 0.032*(mV**-1)*5*mV/exprel((15.*mV-v+VT)/(5*mV))/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
I : amp
''')

eqns_hh_spatial = Equations('''
Im = (gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK)) / point_area : amp/meter**2
I : amp (point current)
dm/dt = 0.32*(mV**-1)*4*mV/exprel((13.*mV-v+VT)/(4*mV))/ms*(1-m)-0.28*(mV**-1)*5*mV/exprel((v-VT-40.*mV)/(5*mV))/ms*m : 1
dn/dt = 0.032*(mV**-1)*5*mV/exprel((15.*mV-v+VT)/(5*mV))/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
''')


# Leaky integrator
eqns_leaky_integrator = Equations('''
dv/dt = (gl * (El - v) - (v - Esyn) * gsyn + I)/Cm : volt
I : amp
gsyn : siemens
''')

# First-order synapse
eqns_synapse = Equations('''
dg/dt =  -(1 / tau_syn) * g : 1 (clock-driven)
gsyn_post = g * siemens : siemens (summed)
w : siemens
''')

# Gap junction synapse. We use this to couple the dendritic, somatic, and
# axonal section of the neuron
eqns_junction = Equations('''
I_post = g * (v_pre - v_post) : amp (summed)
g : siemens
''')

###############################################################################
# Network setup                                                               #
###############################################################################

# Input neuron. We solely use this neuron as a spike source.
input_neuron = NeuronGroup(N=1,
                           model=eqns_hh,
                           threshold='v > -20*mV',
                           refractory='v > -40*mV',
                           method='exponential_euler')
input_neuron.v = El

# Target neuron. This is the neuron we actually analyse. This neuron consists
# of a synapse and a long axon.
dendrite = NeuronGroup(N=1, model=eqns_leaky_integrator, method='exponential_euler')
dendrite.v = El

# Somatic and axonal part of the neuron
morpho = Cylinder(length=66 * cm, diameter=2 * 238 * um, n=1000)
neuron = SpatialNeuron(morphology=morpho,
                       model=eqns_hh_spatial,
                       threshold='v > -20*mV',
                       refractory='v > -40*mV',
                       method='exponential_euler',
                       Cm=1*uF/cm**2,
                       Ri=35.4*ohm*cm)
neuron.v = El

# Target neuron
target = NeuronGroup(N=1, model=eqns_leaky_integrator, method='exponential_euler')
target.v = El

# Connect the network
syn_in_den = Synapses(*(input_neuron, dendrite), # Input to dendrite
                      model=eqns_synapse,
                      on_pre='g += w / siemens',
                      method='exponential_euler')
syn_den_som = Synapses(*(dendrite, neuron), # Dendrite to soma coupling
                       model=eqns_junction,
                       method='exponential_euler')
syn_som_den = Synapses(*(neuron, dendrite), # Soma to dendrite coupling
                       model=eqns_junction,
                       method='exponential_euler')
syn_som_tar = Synapses(*(neuron, target), # Soma to target coupling
                       model=eqns_synapse,
                       on_pre='g += w / siemens',
                       method='exponential_euler')
syn_in_den.connect()
syn_in_den.w[0, 0] = 7.0 * nS

# Coupling conductance (for some reason the units don't match properly after switching to a spatial neurons)
syn_den_som.connect(i=0, j=0)
syn_som_den.connect(i=0, j=0)
syn_den_som.g[0, 0] = 15.0 * uS
syn_som_den.g[0, 0] = 15.0 * nS

syn_som_tar.connect(i=0, j=0) # Connecting to a different compartment does not seem to work
syn_som_tar.w[0, 0] = 10.0 * nS

###############################################################################
# Simulation                                                                  #
###############################################################################

# Monitor spikes and voltages of all neurons
input_spike_monitor = SpikeMonitor(input_neuron)
input_voltage_monitor = StateMonitor(input_neuron,
                                     'v',
                                     record=True,
                                     dt=0.1 * ms)
synapse_conductance_monitor = StateMonitor(syn_in_den,
                                           'g',
                                           record=syn_in_den[0, 0],
                                           dt=0.1 * ms)
dendrite_voltage_monitor = StateMonitor(dendrite,
                                        'v',
                                        record=True,
                                        dt=0.1 * ms)
neuron_voltage_monitor = StateMonitor(neuron,
                                      'v',
                                      record=True,
                                      dt=0.1 * ms)
target_voltage_monitor = StateMonitor(target,
                                      'v',
                                      record=True,
                                      dt=0.1 * ms)

# Use a relatively small timestep
defaultclock.dt = 10 * us

# Create a random number generator for the current source
rng = np.random.RandomState(47281)
run(10 * ms)
for _ in range(50):
    input_neuron.I = rng.uniform(0.2, 0.8) * nA
    run(1 * ms)
input_neuron.I = 0.0 * nA
run(50 * ms)

fig, ax = subplots()
ax.plot(input_voltage_monitor.t / ms, input_voltage_monitor.v[0] / mV)
ax.plot(dendrite_voltage_monitor.t / ms, dendrite_voltage_monitor.v[0] / mV)
for i in np.linspace(0, 999, 3, dtype=int):
    ax.plot(neuron_voltage_monitor.t / ms, neuron_voltage_monitor.v[i] / mV)
ax.plot(target_voltage_monitor.t / ms, target_voltage_monitor.v[0] / mV)
fig.savefig('../../../data/example_neuron_voltage_traces.svg')
