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


def plot_data(ax0,
              ax1,
              ts,
              us,
              Js,
              ylabels=False,
              color1=utils.blues[0],
              color2="k",
              letter="A",
              final=True,
              title=""):
    ax0.plot(ts * 1e3, us * 1e3, color=color1)
    ax1.plot(ts * 1e3, Js * 1e9, clip_on=False, color=color2)

    if final:
        ax0.axhline(-65, color='k', linestyle=(0, (1, 1)), zorder=-1, linewidth=0.5)
        ax0.set_xlim(0, 15)
        ax0.set_xticks(np.arange(0, 16, 5))
        ax0.set_xticks(np.arange(0, 16, 2.5), minor=True)
        ax0.set_ylim(-90, 50)
        ax0.set_yticks(np.arange(-80, 51, 40))
        ax0.set_yticks(np.arange(-90, 51, 10), minor=True)
        ax0.set_xticklabels([])
        ax1.set_xlim(0, 15)
        ax1.set_xticks(np.arange(0, 16, 5))
        ax1.set_xticks(np.arange(0, 16, 2.5), minor=True)
        ax1.set_ylim(0, 10)
        ax1.set_yticks([0, 10])
        ax1.set_yticks([0, 5, 10], minor=True)
        ax1.set_xlabel("Time $t$ (ms)")

        if ylabels:
            ax0.set_ylabel('Potential $v(t)$ (mV)', labelpad=1.0)
            ax1.set_ylabel('$J(t)$ (nA)', labelpad=5.6)
        else:
            ax0.set_yticklabels([])
            ax1.set_yticklabels([])

        ax0.text(-0.2125 if ylabels else -0.0125,
                 1.0275,
                 "\\textbf{{{}}}".format(letter),
                 va="bottom",
                 ha="left",
                 transform=ax0.transAxes,
                 fontdict={"size": 12})
        ax0.set_title(title)



fig, axs = plt.subplots(2,
                        3,
                        figsize=(7.4, 3.5),
                        gridspec_kw={
                            "hspace": 0.25,
                            "wspace": 0.125,
                            "height_ratios": [9, 2]
                        })

#cs1 = utils.blues[::-1]
#cs2 = utils.oranges[::-1]
cs1 = utils.blues[::-1]
cs2 = ["#a0a0a0", "#606060", "k"] #utils.oranges[::-1]

for i, J in enumerate(np.linspace(0.0e-9, 4e-9, 4)[1:]):
    ts, us, Js, gNas, gKs = execute_model([(5e-3, 0.0e-9), (1e-3, J), (9e-3, 0.0e-9)])
    plot_data(*axs[:, 0], ts, us, Js, ylabels=True, color1=cs1[i], color2=cs2[i], final=i==2, letter="A", title="Subthreshold")
utils.annotate(axs[0, 0], 2.5, -61, 3.5, -35, "\\textit{Resting state}")
utils.annotate(axs[0, 0], 7.5, -42, 8.5, -15, "\\textit{Small depolarisation}")

ts, us, Js, gNas, gKs = execute_model([(5e-3, 0.0e-9), (1e-3, 5e-9), (9e-3, 0.0e-9)])
plot_data(*axs[:, 1], ts, us, Js, letter="B", title="Superthreshold")
utils.annotate(axs[0, 1], 7.75, 40, 8.75, 45, "\\textit{Depolarisation}", ha="left")
utils.annotate(axs[0, 1], 7.7, -78.5, 6.6, -77.5, "\\textit{Hyperpolarisation}", ha="right", va="center")

ts, us, Js, gNas, gKs = execute_model([(5e-3, 0.0e-9), (1e-3, 5e-9), (1.75e-3, 0.0e-9),
                            (1e-3, 10e-9), (6.25e-3, 0.0e-9)])
plot_data(*axs[:, 2], ts, us, Js, letter="C", title="Refractory period")
utils.annotate(axs[0, 2], 8.25, -55.5, 11.25, 0, "\\textit{Suppressed}\n\\textit{action potential}", ha="center", va="center")

utils.save(fig)

