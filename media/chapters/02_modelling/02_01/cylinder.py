#!/usr/bin/env python3


def simulate_model(J=1e-9, Toff=0.25e-3, T=2e-3, Tpulse=1e-3, N=200, L=1e-3):
    # Adapted from https://brian2.readthedocs.io/en/stable/examples/compartmental.cylinder.html

    import brian2

    # Set the default timestep
    brian2.defaultclock.dt = 10 * brian2.units.us

    # Disable magic stuff
    brian2.start_scope()

    # Morphology
    diameter = 1 * brian2.um
    length = L * brian2.meter
    Cm = 1 * brian2.uF / brian2.cm**2
    Ri = 150 * brian2.ohm * brian2.cm
    N = N
    morpho = brian2.Cylinder(diameter=diameter, length=length, n=N)

    # Passive channels
    gL = 5e-4 * brian2.siemens / brian2.cm**2
    EL = -65 * brian2.mV
    eqs = '''
    Im = gL * (EL - v) : amp/meter**2
    I : amp (point current)
    '''

    neuron = brian2.SpatialNeuron(morphology=morpho,
                                  model=eqs,
                                  Cm=Cm,
                                  Ri=Ri,
                                  method='exponential_euler')
    neuron.v = EL
    monitor = brian2.StateMonitor(neuron, {'v'}, record=True)

    net = brian2.Network([neuron, monitor])
    net.run(Toff * brian2.second)
    neuron.I[N // 2] = J * brian2.amp
    net.run(Tpulse * brian2.second)
    neuron.I[N // 2] = 0 * brian2.amp
    net.run((T - Tpulse - Toff) * brian2.second)

    return monitor.t / brian2.second, (
        neuron.distance - length / 2) / brian2.meter, monitor.v / brian2.volt


ts, xs, us = simulate_model()
dt = (ts[-1] - ts[0]) / (len(ts) - 1)
dx = (xs[-1] - xs[0]) / (len(xs) - 1)

fig, axs = plt.subplots(1,
                        2,
                        figsize=(7.4, 1.2),
                        gridspec_kw={
                            "width_ratios": [5, 2],
                            "wspace": 0.35,
                        })

t_idcs = list(range(us.shape[1] // 20, us.shape[1], 20))
for i in t_idcs:
    colour = mpl.cm.get_cmap('viridis')(i / (us.shape[1] - 1))
    axs[0].plot(xs * 1e3, us[:, i] * 1e3, color=colour)
axs[0].set_xlabel('Distance from the injection site $x$ (mm)')
axs[0].set_ylabel('$v(t)$ (mV)')
axs[0].set_xlim((xs[0] - dx / 2) * 1e3, (xs[-1] + dx / 2) * 1e3)

yticks = np.arange(-70, 51, 30)
axs[0].set_yticks(yticks)
for ytick in yticks:
    axs[0].plot([-0.5, 0.56], [ytick, ytick],
                linestyle=':',
                linewidth=0.5,
                color='k',
                zorder=-100,
                clip_on=False)

yticks2 = np.arange(-70, 51, 15)
for i in range(len(yticks2) - 1):
    v = 0.5 * (yticks2[i] + yticks2[i + 1])
    colour = mpl.cm.get_cmap('Blues')((v + 70) / (135))
    axs[0].plot([0.55, 0.55], [yticks2[i], yticks2[i + 1]],
                clip_on=False,
                solid_capstyle='butt',
                linewidth=5.0,
                color=colour,
                zorder=-101)
axs[0].set_ylim(-70, 50)
axs[0].set_xlim(-0.5, 0.5)
axs[0].text(-0.12, 1.05, '\\textbf{A}', ha='left', va='top', transform=axs[0].transAxes, size=12)

axs[1].imshow(us,
              cmap='Blues',
              vmin=-70e-3,
              vmax=65e-3,
              extent=((ts[0] + dt) * 1e3, (ts[-1] + dt) * 1e3,
                      (xs[0] + dx) * 1e3, (xs[-1] + dx) * 1e3))
axs[1].set_aspect('auto')
axs[1].set_xlim(ts[0] * 1e3, (ts[-1] + dt) * 1e3)
axs[1].set_ylim((xs[0] - dx / 2) * 1e3, (xs[-1] + dx / 2) * 1e3)
axs[1].set_xlabel('Time $t$ (ms)')
axs[1].set_ylabel('$x$ (mm)')
axs[1].text(-0.31, 1.05, '\\textbf{B}', ha='left', va='top', transform=axs[1].transAxes, size=12)

for i in t_idcs:
    colour = mpl.cm.get_cmap('viridis')(i / (us.shape[1] - 1))
    axs[1].plot(ts[i] * 1e3,
                -0.5,
                'o',
                color=colour,
                clip_on=False,
                zorder=1000,
                markersize=5)
    axs[1].axvline(ts[i] * 1e3, linestyle=':', linewidth=0.5, color='k')

axs[1].plot([0.25, 1.25], [0.2, 0.2],
            'k-',
            solid_capstyle='butt',
            linewidth=2.0)
axs[1].text(
    0.75,
    0.25,
    'Stimulus',
    ha='center',
    va='bottom',
    bbox={
        "pad": 1.0,
        "color": "#eff6fc",
        "linewidth": 0.0,
    },
)

utils.save(fig)

