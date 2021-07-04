# This file generates the membrane potential traces embedded in figure 2.2

from brian2.units import *


def execute_model(time_and_current_tuples):
    import brian2

    # Set the default timestep
    brian2.defaultclock.dt = 10 * brian2.units.us

    # Disable magic stuff
    brian2.start_scope()

    # HH model adapted from the following Brian example:
    # https://brian2.readthedocs.io/en/stable/examples/IF_curve_Hodgkin_Huxley.html

    # Uses Traub and Miles Channel dynamics.

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
    VT = -50 * mV

    tau_syn = 5 * ms

    # HH model with current input
    eqns_hh = brian2.Equations('''
    dv/dt = (gl*(El-v) - gNa *(v-ENa) - gK * (v-EK) + I)/Cm : volt
    dm/dt = 0.32*(mV**-1)*4*mV/exprel((13.*mV-v+VT)/(4*mV))/ms*(1-m)-0.28*(mV**-1)*5*mV/exprel((v-VT-40.*mV)/(5*mV))/ms*m : 1
    dn/dt = 0.032*(mV**-1)*5*mV/exprel((15.*mV-v+VT)/(5*mV))/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
    dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
    gNa = g_na*(m*m*m)*h : siemens
    gK = g_kd*(n*n*n*n) : siemens
    I : amp
    ''')

    neuron = brian2.NeuronGroup(N=1,
                                model=eqns_hh,
                                threshold='v > -20*mV',
                                refractory='v > -40*mV',
                                method='exponential_euler')
    neuron.v = El

    monitor = brian2.StateMonitor(neuron, {"v", "I", "gNa", "gK"}, record=True)

    # Create a network
    net = brian2.Network([neuron, monitor])
    for duration, J in time_and_current_tuples:
        neuron.I = J * amp
        net.run(duration * second)

    # Fetch the data
    ts = np.array(monitor.t / second)
    us = np.array(monitor.v[0] / volt)
    Js = np.array(monitor.I[0] / amp)
    gNas = np.array(monitor.gNa[0] / siemens)
    gKs = np.array(monitor.gK[0] / siemens)

    return ts, us, Js, gNas, gKs


fig, (ax1, ax2, ax3) = plt.subplots(3,
                        1,
                        figsize=(2.5, 2.25),
                        gridspec_kw={
                            "hspace": 0.75,
                            "wspace": 0.125,
                            "height_ratios": [2, 1, 1]
                        })

ts, us, Js, gNas, gKs = execute_model([(5e-3, 0.0e-9), (1e-3, 5e-9), (9e-3, 0.0e-9)])
cs1 = utils.blues[::-1]
cs2 = ["#c0c0c0", "#505050", "k"] #utils.oranges[::-1]

I = np.logical_and(ts > 4e-3, ts < 9e-3)

ax1.axhline(-65, color='k', linestyle=(0, (1, 1)), zorder=-1, linewidth=0.5)
ax1.plot(ts[I] * 1e3, us[I] * 1e3, color=utils.blues[0], clip_on=False)
ax1.set_ylabel("$v$ (mV)")
ax1.spines["bottom"].set_visible(False)
utils.annotate(ax1, 6.9, 0, 6.5, 20, "\\textit{Membrane}\n\\textit{potential}", ha="right")
ax1.set_yticks(np.arange(-80, 41, 40))
ax1.set_yticks(np.arange(-80, 41, 20), minor=True)
ax1.set_xlim(4, 9)
ax1.set_xticks([])

ax2.plot(ts[I] * 1e3, gNas[I] * 1e6, color=utils.reds[0], clip_on=False)
ax2.set_ylabel("$g_\\mathrm{Na^+}$ (µS)", labelpad=9)
ax2.spines["bottom"].set_visible(False)
ax2.set_xticks([])
ax2.set_xlim(4, 9)
ax2.set_ylim(0, 10)
ax2.set_yticks([0, 10])
ax2.set_yticks([0, 5, 10], minor=True)
utils.annotate(ax2, 6.9, 4, 6.5, 7, "\\textit{Sodium $(\\mathrm{Na^+})$}\n\\textit{conductance}", ha="right")

ax3.plot(ts[I] * 1e3, gKs[I] * 1e6,  color=utils.oranges[0], clip_on=False)
ax3.set_ylabel("$g_\\mathrm{K^+}$ (µS)", labelpad=13)
ax3.set_xlim(4, 9)
ax3.set_ylim(0, 2)
ax3.set_yticks([0, 2])
ax3.set_yticks([0, 1, 2], minor=True)
ax3.set_xticks(np.arange(4, 9.1, 1))
ax3.set_xticks(np.arange(4, 9.1, 0.5), minor=True)
ax3.spines["bottom"].set_position(("outward", 10))
utils.annotate(ax3, 7.25, 1, 6.5, 1.5, "\\textit{Potassium $(\\mathrm{K^+})$}\n\\textit{conductance}", ha="right")
ax3.set_xlabel("Time $t$ (ms)")

utils.save(fig)

