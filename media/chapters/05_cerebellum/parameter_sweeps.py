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

from benchmark_plots_common import *

#
# PARAMETER SWEEPS
#

# Plot the figures for the paper
files = [
    utils.datafile("sweep_tau_wn_two_populations_dales_principle_detailed.h5"),
    utils.datafile(
        "sweep_n_pcn_granule_convergence_wn_two_populations_dales_principle_detailed.h5"
    ),
    utils.datafile(
        "sweep_n_golgi_granule_convergence_wn_two_populations_dales_principle_detailed.h5"
    ),
]

default_values = [6, 3, 3]

fig, axs = plt.subplots(1, 3, figsize=(5, 2.0))

for i, fn in enumerate(files):
    ax = axs[i]
    plot_benchmark_result(fn, ax=ax, show_title=False, small=True)
    if i > 0:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    utils.outside_ticks(ax)

    ax.arrow(default_values[i],
             -0.055,
             0.0,
             0.02,
             width=0.2,
             head_width=0.5,
             head_length=0.04,
             linewidth=0.5,
             edgecolor='white',
             facecolor='k',
             clip_on=False,
             zorder=100)

    ax.text(-0.025,
            1.15,
            "$\\textbf{{{}}}$".format(chr(ord('A') + i)),
            va='top',
            ha='left',
            fontsize=12,
            color='black',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', linewidth=0, pad=0.4))

utils.save(fig)

