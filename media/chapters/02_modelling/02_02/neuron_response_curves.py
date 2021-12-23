import os


# Loads voltages, spikes and currents
def load_ramp_data(filename):
    data = np.load(os.path.join(utils.datadir, filename))
    ts = data["ts"] * 1e-3  # ms to sec
    us = data["us"] * 1e-3  # mV to V
    Js = data["Js"] * 1e-9  # nA to A
    spikes = data["spikes"] * 1e-3  # ms to sec
    dt = (ts[-1] - ts[0]) / (len(ts) - 1)

    return ts, us, Js, spikes, dt


# Processes the ramp data into an IF-curve
def load_and_process_ramp_data(filename, wnd=2, ss=50):
    ts, us, Js, spikes, dt = load_ramp_data(utils.datafile(filename))

    # Estimate the rates by computing the inverse of the inter-spike-intervals
    # over wnd spikes
    rates = wnd * 1.0 / (spikes[wnd:] - spikes[:-wnd])

    # Compute the times corresponding to the estimated rates
    rates_ts = 0.5 * (spikes[wnd:] + spikes[:-wnd])

    # Compute the indices corresponding to the estimates rates
    rates_idcs = np.array(rates_ts / dt, dtype=int)

    # Fetch the currents corresponding to those indices
    rates_Js = Js[rates_idcs]

    # Return sub-sampled versions of the estimates
    return np.concatenate(((-0.5e-9, rates_Js[0]), rates_Js[::ss])), \
           np.concatenate(((0.0, 0.0), rates[::ss]))


def do_plot(ax2, fn2, letter, title, ref_fn=None):
    Js, As = load_and_process_ramp_data(fn2)

    ax2.plot(Js * 1e9, As, 'k', linewidth=2.0, color='k', clip_on=False)
    #if ref_fn:
    #    ax2.plot(Js * 1e-9, ref_fn(As))
    ax2.set_xlim(-0.5, 2.0)
    ax2.set_xticks(np.arange(-0.5, 2.1, 0.5))
    ax2.set_xticks(np.arange(-0.5, 2.0, 0.25), minor=True)
    ax2.set_yticks(np.arange(0, 250, 100))
    ax2.set_yticks(np.arange(0, 251, 50), minor=True)
    ax2.set_yticks([0, 100, 200])
    ax2.set_yticks([25, 50, 75, 125, 150, 175, 225, 250], minor=True)
    ax2.set_ylim(0.0, 250.0)
    ax2.set_xlabel('Input current $J$ (nA)')
    ax2.set_ylabel('$G[J]$ ($\\mathrm{s}^{-1}$)', labelpad=1.0)
    ax2.set_title(title)
    ax2.text(-0.1075,
             1.0275,
             '\\textbf{' + letter + '}',
             fontdict={'size': 12},
             va='bottom',
             ha='right',
             transform=ax2.transAxes)

def lif_rate(Js, tau_ref=2e-3, Cm=200.0e-12, gL=10e-9, vTh=-55e-3, EL=-65e-3, vReset=-85e-3):
    tauRC = Cm / gL
    Jth = (vTh - EL) * gL
    valid = Js > Jth
    t_spike = -tauRC * np.log1p((~valid) * 1.0 - valid * (((vTh - vReset) * gL) / ((EL - vReset) * gL + Js)))
    As = valid / ((~valid) * 1.0 + valid * (tau_ref + t_spike))
    return As

fig = plt.figure(figsize=(7.2, 2.95))
gs1 = fig.add_gridspec(1, 2, wspace=0.4)
ax02 = fig.add_subplot(gs1[0])
ax12 = fig.add_subplot(gs1[1])

do_plot(ax02, 'lif_ramp.npz', 'A', 'LIF neuron', lif_rate)
do_plot(ax12, 'hodgkin_huxley_ramp.npz', 'B', 'Hodgkin-Huxley neuron')

Js = np.linspace(-0.5, 2.0, 1000) * 1e-9
ax02.plot(Js * 1e9, lif_rate(Js), ':', color='white', clip_on=False)

l1 = mpl.lines.Line2D([], [], linewidth=2, color='k')
l2 = mpl.lines.Line2D([], [], linewidth=1, linestyle=':', color='k')
ax02.legend([l1, l2], ['Empirical measurement', 'Rate model'], loc='upper left')

utils.save(fig)

