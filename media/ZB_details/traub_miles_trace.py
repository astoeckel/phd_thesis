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
    g_na = 100 * msiemens * cm**-2 * point_area
    g_kd = 30 * msiemens * cm**-2 * point_area
    VT = -50 * mV

    print(Cm, gl, g_na, g_kd)

    # HH model with current input
    eqns_hh = brian2.Equations('''
    dv/dt = (IL +  INa + IK + I) / Cm : volt
    dm/dt = 0.32*(mV**-1)*4*mV/exprel((13.*mV-v+VT)/(4*mV))/ms*(1-m)-0.28*(mV**-1)*5*mV/exprel((v-VT-40.*mV)/(5*mV))/ms*m : 1
    dn/dt = 0.032*(mV**-1)*5*mV/exprel((15.*mV-v+VT)/(5*mV))/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
    dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
    IL = gl*(El-v) : amp
    INa = gNa *(ENa-v) : amp
    IK = gK * (EK-v) : amp
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

    monitor = brian2.StateMonitor(neuron, {"v", "I", "IL", "INa", "IK", "gNa", "gK", "m", "h", "n"}, record=True)

    # Create a network
    net = brian2.Network([neuron, monitor])
    for duration, J in time_and_current_tuples:
        neuron.I = J * amp
        net.run(duration * second)

    # Fetch the data; hackily scope this so we do not override unit definitions
    def result():
        ts = np.array(monitor.t / second)
        us = np.array(monitor.v[0] / volt)
        Js = np.array(monitor.I[0] / amp)
        JLs = np.array(monitor.IL[0] / amp)
        JNas = np.array(monitor.INa[0] / amp)
        JKs = np.array(monitor.IK[0] / amp)
        gNas = np.array(monitor.gNa[0] / siemens)
        gKs = np.array(monitor.gK[0] / siemens)
        ms = np.array(monitor.m[0])
        hs = np.array(monitor.h[0])
        ns = np.array(monitor.n[0])
        return ts, us, Js, JLs, JNas, JKs, gNas, gKs, ms, hs, ns

    return result()


ts, us, Js, JLs, JNas, JKs, gNas, gKs, ms, hs, ns = execute_model([(0.1e-3, 0.0e-9), (1e-3, 5e-9), (5.9e-3, 0.0e-9)])

fig, axs = plt.subplots(10, 1, figsize=(7.2, 8.5), gridspec_kw={
    "height_ratios": [
        2,
        0.5,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
})

axs[0].axhline(-65, linestyle=(0, (1, 1)), linewidth=0.75, color='k')
axs[0].plot(ts * 1e3, us * 1e3, 'k', color=utils.blues[0])
axs[0].set_ylabel("$v$ (mV)", labelpad=9)
utils.annotate(axs[0], 3.5, 20, 4.3, 35, 'Membrane potential $v(t)$', va="center", ha="left")
axs[0].plot([0.1, 1.1], [-20, -20], 'k-', linewidth=2.0, solid_capstyle='butt')
axs[0].text(0.55, -15.0, '$1\,\mathrm{ms}$', va="bottom", ha="center")

axs[1].axhline(0, linestyle=(0, (1, 1)), linewidth=0.75, color='k')
axs[1].plot(ts * 1e3, Js * 1e9, 'k')
axs[1].set_ylabel("$J$ (nA)", labelpad=18)
utils.annotate(axs[1], 1.2, 2.6, 1.6, 2.5, "Input current $J(t)$", va="center", ha="left")

axs[2].axhline(0, linestyle=(0, (1, 1)), linewidth=0.75, color='k')
axs[2].plot(ts * 1e3, JLs * 1e9, 'k')
axs[2].set_ylabel("$J_\\mathrm{L}$ (nA)", labelpad=13)
utils.annotate(axs[2], 3.7, -0.5, 4.3, -0.75, "Leak current $J_\\mathrm{L}(t)$", va="center", ha="left")

axs[3].axhline(0, linestyle=(0, (1, 1)), linewidth=0.75, color='k')
axs[3].plot(ts * 1e3, JNas * 1e9, color=utils.reds[0])
axs[3].set_ylabel("$J_\\mathrm{Na^+}$ (nA)", labelpad=10)
utils.annotate(axs[3], 3.8, 50, 4.3, 75, "Sodium current $J_\\mathrm{Na^+}(t)$", va="center", ha="left")

axs[4].axhline(0, linestyle=(0, (1, 1)), linewidth=0.75, color='k')
axs[4].plot(ts * 1e3, gNas * 1e6, color=utils.reds[0])
axs[4].set_ylabel("$g_\mathrm{Na^+}$ (µS)", labelpad=18.5)
utils.annotate(axs[4], 3.5, 2.5, 4.3, 3.5, "Sodium conductance $g_\\mathrm{Na^+}(t)$", va="center", ha="left")

axs[5].axhline(0, linestyle=(0, (1, 1)), linewidth=0.75, color='k')
axs[5].plot(ts * 1e3, ms, color=utils.reds[0], clip_on=False)
axs[5].set_ylim(0, 1)
axs[5].set_yticks([0, 1])
axs[5].set_ylabel("$m$", labelpad=18)
utils.annotate(axs[5], 3.9, 0.25, 4.3, 0.5, "Sodium gating variable $m(t)$", va="center", ha="left")

axs[6].axhline(0, linestyle=(0, (1, 1)), linewidth=0.75, color='k')
axs[6].plot(ts * 1e3, hs, color=utils.reds[0], clip_on=False)
axs[6].set_ylim(0, 1)
axs[6].set_yticks([0, 1])
axs[6].set_ylabel("$h$", labelpad=18)
utils.annotate(axs[6], 3.6, 0.15, 3.1, 0.8, "Sodium shutoff $h(t)$", va="center", ha="right")


axs[7].axhline(0, linestyle=(0, (1, 1)), linewidth=0.75, color='k')
axs[7].plot(ts * 1e3, JKs * 1e9, color=utils.oranges[0])
axs[7].set_ylabel("$J_\\mathrm{K^+}$ (nA)")
utils.annotate(axs[7], 3.8, -75, 4.3, -50, "Potassium current $J_\\mathrm{K^+}(t)$", va="center", ha="left")

axs[8].axhline(0, linestyle=(0, (1, 1)), linewidth=0.75, color='k')
axs[8].plot(ts * 1e3, gKs * 1e6, color=utils.oranges[0])
axs[8].set_ylabel("$g_\mathrm{K^+}$ (µS)", labelpad=18)
utils.annotate(axs[8], 3.95, 0.9, 4.3, 1.2, "Potassium conductance $g_\\mathrm{K^+}(t)$", va="center", ha="left")

axs[9].axhline(0, linestyle=(0, (1, 1)), linewidth=0.75, color='k')
axs[9].plot(ts * 1e3, ns, color=utils.oranges[0], clip_on=False)
axs[9].set_ylim(0, 1)
axs[9].set_yticks([0, 1])
axs[9].set_ylabel("$n$", labelpad=18)
utils.annotate(axs[9], 3.95, 0.65, 4.3, 0.8, "Potassium gating variable $n(t)$", va="center", ha="left")

for ax in axs:
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.set_xlim(ts[0] * 1e3 - 0.1, ts[-1] * 1e3 + 0.1)


utils.save(fig)

