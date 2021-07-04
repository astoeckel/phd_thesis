#!/usr/bin/env python3

from nef_synaptic_computation.multi_compartment_lif import *

# One-compartment, current-based LIF
LIF_cur = (Neuron().add_compartment(
    Compartment(soma=True).add_channel(CurChan(mul=1, name="j")).add_channel(
        CondChan(Erev=-65e-3, g=50e-9, name="leak"))).assemble())


# Generates a mapping between the given rates and the other
# value "xs" -- this function is necessary since we do not
# directly observe "xs" as a function of
def map_onto_rates(rates, xs, bins=100):
    rmin = 2
    rmax = 200
    rbins = np.linspace(rmin, rmax + (rmax - rmin) / (bins - 1), bins + 1)
    res = [[] for _ in range(len(rbins) - 1)]

    rates_flat = rates.flatten()
    xs_flat = xs.flatten()
    for j, r in enumerate(rates_flat):
        i = int((r - rmin) * bins / (rmax - rmin))
        if i >= 0 and i < bins:
            res[i].append(xs_flat[j])

    return rbins[:-1], res


# Plots an individual result
def do_plot(ax, rs, xs, color="k", **kwargs):
    def nanwrap(f):
        return lambda x: np.NaN if len(x) == 0 else f(x)

    xs_mean = np.array(list(map(nanwrap(np.mean), xs)))
    xs_perc_25 = np.array(
        list(map(nanwrap(lambda x: np.percentile(x, 75)), xs)))
    xs_perc_75 = np.array(
        list(map(nanwrap(lambda x: np.percentile(x, 25)), xs)))

    artist = ax.plot(rs,
                     xs_mean * 1e3,
                     color=color,
                     **kwargs,
                     linewidth=1,
                     zorder=2)
    ax.fill_between(rs,
                    xs_perc_25 * 1e3,
                    xs_perc_75 * 1e3,
                    color=color,
                    alpha=0.25,
                    linewidth=0,
                    zorder=1)

    return artist[0]


def do_setup(ax, rs):
    v_th = LIF_cur.soma().v_th
    v_reset = LIF_cur.soma().v_reset
    ax.plot(rs,
            np.ones_like(rs) * v_th * 1e3,
            color="k",
            linestyle=(0, (1, 5)),
            linewidth=0.5,
            zorder=0)
    ax.plot(rs,
            np.ones_like(rs) * v_reset * 1e3,
            color="k",
            linestyle=(0, (1, 5)),
            linewidth=0.5,
            zorder=0)
    ax.set_xlim(0, 200)
    ax.set_xticks(np.arange(0, 201, 50))
    ax.set_xticks(np.arange(0, 201, 25), minor=True)
    ax.set_ylim(-65, -45)
    ax.set_yticks(np.arange(-65, -44, 5))
    ax.set_yticks(np.arange(-65, -44, 2.5), minor=True)
    ax.set_xlabel("Output rate ($\\mathrm{s}^{-1}$)")
    ax.set_ylabel("$\\bar v$ ($\\mathrm{mV}$)")


with h5py.File(utils.datafile("average_som_pot_large.h5"), "r") as f:
    Js = f["Js"][()]
    v_som = f["v_som"][()]
    v_som_no_ref = f["v_som_no_ref"][()]
    rates = f["rates"][()]

# Number of pre-population spikes for which to perform the experiment
# Larger values/none correspond to smaller input noise
n_pre_spikes_list = [50, 100, 500, None]

fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.5))
ax0, ax1 = axs

cmap = mpl.cm.get_cmap('viridis')
artists = []
for i, n_pre_spikes in enumerate(n_pre_spikes_list):
    if n_pre_spikes is None:
        color = "k"
    else:
        color = cmap(0.75 * (i / (len(n_pre_spikes_list) - 2)))

    rs, xs = map_onto_rates(rates[i], v_som[i])
    rs, xs_no_ref = map_onto_rates(rates[i], v_som_no_ref[i])

    if i == 0:
        do_setup(ax0, rs)
        do_setup(ax1, rs)

    artists.append(do_plot(ax0, rs, xs, color=color))
    do_plot(ax1, rs, xs_no_ref, color=color)

# Plot some simple models
v_th = LIF_cur.soma().v_th
v_reset = LIF_cur.soma().v_reset
v_spike = LIF_cur.soma().v_spike

dr = 1 / rs
pred0 = ((v_th + v_reset) * 0.5 * (dr - 3e-3) / dr + (v_spike) * 1.0 *
         (1e-3) * rs + (v_reset) * 1.0 * (2e-3) * rs) * 1e3
pred1 = (v_th + v_reset) * 0.5 * 1e3 * np.ones_like(rs)

ax0.plot(rs, pred0, color='k', linewidth=0.5, zorder=0)
artists.append(
    mpl.lines.Line2D([], [], color='k', linewidth=0.5, linestyle=(0, (1, 1))))
ax0.plot(rs, pred0, color='w', linewidth=0.5, linestyle=(0, (1, 1)), zorder=3)
ax1.plot(rs, pred1, color='k', linewidth=0.5, zorder=0)
ax1.plot(rs, pred1, color='w', linewidth=0.5, linestyle=(0, (1, 1)), zorder=3)

for i in range(2):
    axs[i].text(-0.155,
                1.13,
                '\\textbf{{{}}}'.format(chr(ord('A') + i)),
                va='top',
                ha='left',
                size=12,
                transform=axs[i].transAxes)

ax0.set_title("{With refractory and spike period}")
ax1.set_title("{Without refractory and spike period}")

fig.legend(artists, [
    "50 $\\mathrm{s}^{-1}$", "100 $\\mathrm{s}^{-1}$",
    "500 $\\mathrm{s}^{-1}$", "No spike noise", "Linear model"
],
           ncol=len(n_pre_spikes_list) + 1,
           loc="upper center",
           bbox_to_anchor=(0.5, 1.15))

utils.save(fig)

