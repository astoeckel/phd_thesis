#   Code for the PhD Thesis
#   "Harnessing Neural Dynamics as a Computational Resource: Building Blocks
#   for Computational Neuroscience and Artificial Agents"
#   Copyright (C) 2021  Andreas Stöckel
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Run code/cerebellum/generate_cerebellum_detailed_neurons_example.py first

def hide_spines(ax):
    for side in ['top', 'right', 'bottom', 'left']:
        ax.spines[side].set_visible(False)
    return ax


def rasterplot(ax, ts, A, **style):
    N, n = A.shape
    for i in range(n):
        for t in ts[np.where(A[:, i] != 0)]:
            ax.plot([t, t], [i + 0.5, i + 1.5],
                    zorder=-100,
                    solid_capstyle="butt",
                    **style)
    ax.set_ylim(0.5, n + 0.5)


def plot_decoding_example(axs,
                          ts,
                          xs,
                          ys,
                          spikes,
                          delays,
                          xs_tars,
                          ys_hats,
                          theta,
                          name,
                          ax1_ylim,
                          ax2_ylim,
                          ax3_ylim,
                          show_t_on=False):

    dt = (ts[-1] - ts[0]) / (len(ts) - 1)
    T = ts[-1] + dt

    ax1 = hide_spines(axs[0])
    ax1.plot(ts, xs, color='k', clip_on=False, linewidth=1.0)
    ax1.set_ylim(*ax1_ylim)
    ax1.set_yticks([])
    ax1.set_xlim(0, T)
    ax1.set_xticks([])
    ax1.text(
        1.0,
        1.8,
        'Input $u(t)$',
        ha='right',
        va='top',
        fontsize=8,
        transform=ax1.transAxes,
    )

    ax1.plot([0.2, 0.6], [-0.5, -0.5],
             color='k',
             linewidth=1.5,
             clip_on=False,
             solid_capstyle="butt")
    ax1.text(0.4,
             ax1_ylim[0] - 0.7,
             '$\\theta = 0.4\\,\\mathrm{s}$',
             ha='center',
             va='top',
             fontsize=8)

    if show_t_on:
        ax1.plot([0.6, 0.8], [1.25, 1.25],
                 color='k',
                 linewidth=1.5,
                 clip_on=False,
                 solid_capstyle="butt")
        ax1.text(0.7,
                 1.30,
                 '$t_\\mathrm{on} = 0.2\\,\\mathrm{s}$',
                 ha='center',
                 va='bottom',
                 fontsize=8)
    else:
        ax1.text(1.75,
                 1.30,
                 'Bandwidth $B = 5.0\\,\\mathrm{Hz}$',
                 ha='center',
                 va='bottom',
                 fontsize=8)

    height = 1  #int(np.round(0.75 * (ax1_ylim[1] - ax1_ylim[0])))
    ax1.plot([0.0, 0.0], [0.0, height],
             color='k',
             linewidth=1.5,
             clip_on=False,
             solid_capstyle="butt")
    ax1.text(0.025,
             0.6,
             '${}$'.format(height),
             ha='left',
             va='center',
             fontsize=8)
    ax1.axhline(0, linestyle='--', linewidth=0.5, color='black', zorder=100)

    ax1.text(
        0.0,
        2.0,
        '$\\mathbf{{{}}}$'.format(name),
        ha='left',
        va='top',
        fontsize=12,
        transform=ax1.transAxes,
    )

    ax2 = hide_spines(axs[1])
    ax2.set_ylim(*ax2_ylim)
    ax2.set_yticks([])
    ax2.set_xlim(0, T)

    #    ax2.axhline(0, linestyle='--', linewidth=0.5, color='black', zorder=100)
    #    ax2.plot(ts, ys)

    ax2.text(
        1.0,
        0.8,
        'Granule cell activities $\\mathbf{a}(t)$',
        ha='right',
        va='top',
        fontsize=8,
        transform=ax2.transAxes,
    )

    # Randomly select n_neurons neurons
    n_neurons = spikes.shape[1]
    rasterplot(ax2, ts, spikes, color='k', linewidth=0.5)
    ax2.set_ylim(int(-n_neurons * 0.1), int(n_neurons * 1.5))

    cmap = cm.get_cmap('viridis')
    ax3 = hide_spines(axs[2])
    ax3.axhline(0, linestyle='--', linewidth=0.5, color='black', zorder=100)
    ax3.set_ylim(*ax3_ylim)
    ax3.set_yticks([])
    for i, delay in enumerate(delays):
        #color = cmap(0.9 * (1.0 - delays[i]))
        color = [cmap(0.1), cmap(0.5), cmap(0.9)][i]
        #color = ["#ce5c00", "#204a87", "#a40000"][i]
        ax3.plot(ts,
                 ys_hats[i],
                 color=color,
                 zorder=i,
                 label='${:0.2g}$'.format(delays[i]))
        ax3.plot(ts,
                 xs_tars[i],
                 color='white',
                 linewidth=0.75,
                 linestyle=(0, (1, 1)),
                 zorder=2 * (i + 1))
        ax3.plot(ts,
                 xs_tars[i],
                 color=color,
                 linewidth=0.75,
                 linestyle=(1, (1, 1)),
                 zorder=2 * (i + 1))
    ax3.text(1.0,
             1.05,
             'Decoded delays ${\\hat u}(t - \\theta\')$',
             ha='right',
             va='top',
             fontsize=8,
             transform=ax3.transAxes)

    ax3.plot([0.0, 0.0], [0.0, 1.0],
             color='k',
             linewidth=1.5,
             clip_on=False,
             solid_capstyle="butt",
             zorder=10)
    ax3.text(0.025, 0.7, '$1$', ha='left', va='center', fontsize=8)
    ax3.legend(
        loc='lower right',
        bbox_to_anchor=(1.025, -0.2),
        ncol=len(delays),
        fontsize=7,
        columnspacing=1.0,
        handlelength=0.9,
        handletextpad=0.5,
    )
    ax3.text(
        0.65,
        -0.06,
        'Delay $\\theta\'/\\theta$',
        ha='right',
        va='center',
        fontsize=7,
        transform=ax3.transAxes,
    )

    plt.subplots_adjust(left=None,
                        bottom=None,
                        right=None,
                        top=None,
                        wspace=0.05,
                        hspace=0.1)

    return fig


with h5py.File(utils.datafile("cerebellum_detailed_neurons_example.h5")) as f:
    fig = plt.figure(figsize=(7.7, 2.75))
    ax11 = plt.subplot2grid((9, 2), (0, 0))
    ax12 = plt.subplot2grid((9, 2), (1, 0), sharex=ax11, rowspan=4)
    ax13 = plt.subplot2grid((9, 2), (5, 0), sharex=ax11, rowspan=4)
    ax21 = plt.subplot2grid((9, 2), (0, 1))
    ax22 = plt.subplot2grid((9, 2), (1, 1), sharex=ax21, rowspan=4)
    ax23 = plt.subplot2grid((9, 2), (5, 1), sharex=ax21, rowspan=4)

    keys = ["ts", "xs", "ys", "spikes", "delays", "xs_tars", "ys_hats", "theta"]
    res_pulse = (f["pulse_input"][key][()]
                 for key in keys)
    res_white = (f["white_input"][key][()]
                 for key in keys)

    plot_decoding_example((ax11, ax12, ax13),
                          *res_pulse,
                          name="A",
                          ax1_ylim=(-0.1, 1),
                          ax2_ylim=(-1.0, 1.25),
                          ax3_ylim=(-0.3, 1.5),
                          show_t_on=True)

    plot_decoding_example((ax21, ax22, ax23),
                          *res_white,
                          name="B",
                          ax1_ylim=(-0.1, 1),
                          ax2_ylim=(-1.0, 1.0),
                          ax3_ylim=(-1.4, 1.75))

    utils.save(fig)

