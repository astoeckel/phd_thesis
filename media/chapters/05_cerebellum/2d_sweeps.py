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

def plot_2d_sweep_grid(dirname):
    import os

    def remove_spines(ax):
        for spine in ['right', 'left', 'top', 'bottom']:
            ax.spines[spine].set_visible(False)

    descrs = {}
    for file in sorted(os.listdir(dirname)):
        if file.endswith(".h5"):
            try:
                fn = os.path.join(dirname, file)

                _, _, descr, _ = plot_benchmark_result(fn, show_title=False, do_plot=False)

                # Skip sweeps
                if descr["is_sweep"]:
                    continue

                key = (descr["mode"] + ("_detailed" if ("detailed" in file) else ""), descr["input_type"])
                descrs[key] = fn
            except OSError:
                pass # Files may be locked
            except Exception as e:
                raise e

    fig, axs = plt.subplots(2, 5, figsize=(7.5, 3.3))
    for i, mode in enumerate(['direct', 'single_population', 'two_populations', 'two_populations_dales_principle', 'two_populations_dales_principle_detailed']):
        for j, input_type in enumerate(['pulse', 'white_noise']):
            ax = axs[j, i]
            remove_spines(ax)
            utils.outside_ticks(ax)
            key = (mode, input_type)
            fn = descrs[key]
            if j == 0:
                levels = np.linspace(0.2, 0.5, 11)
                cmap = 'inferno'
            else:
                levels = np.linspace(0.0, 1.0, 11)
                cmap = 'viridis'
            _, _, _, C = plot_benchmark_result(fn, ax=ax, show_title=False, small=True, levels=levels, cmap=cmap)

            if i == 0:
                if j == 0:
#                    cax = fig.add_axes([0.125, 0.48, 0.375, 0.03])
#                    cax = fig.add_axes([0.125, 0.95, 0.375, 0.03])
                    cax = fig.add_axes([0.125, -0.05, 0.375, 0.03])
                    cax.set_title('Pulse experiment error $E$', fontsize=8)
                elif j == 1:
#                    cax = fig.add_axes([0.525, 0.48, 0.375, 0.03])
#                    cax = fig.add_axes([0.525, 0.95, 0.375, 0.03])
                    cax = fig.add_axes([0.525, -0.05, 0.375, 0.03])
                    cax.set_title('Noise experiment error $E$', fontsize=8)
                cb = plt.colorbar(C, cax=cax, orientation='horizontal')
                cb.outline.set_visible(False)
            
            if i == 0:
                axs[j, i].set_yticks([0, 0.5, 1])
                axs[j, i].set_yticklabels(["0", "", "1"])
                axs[j, i].set_ylabel('Delay $\\theta\'/\\theta$', labelpad=-3.0)
            else:
                axs[j, i].set_yticks([])
                axs[j, i].set_ylabel('')
            if j == 0:
                axs[j, i].set_xticks([10, 50, 90])
                axs[j, i].set_xticklabels([])
                axs[j, i].text(9, -0.16, "0.1", fontsize=9.0, ha='left', va='bottom')
                axs[j, i].text(91, -0.16, "0.9", fontsize=9.0, ha='right', va='bottom')
                axs[j, i].set_xlabel('Pulse width', labelpad=-2.0)
            elif j == 1:
                axs[j, i].set_xticks([0.1, 5, 10])
                axs[j, i].set_xticklabels([])
                axs[j, i].text(0.0, -0.16, "0.1", fontsize=9.0, ha='left', va='bottom')
                axs[j, i].text(10.1, -0.16, "10", fontsize=9.0, ha='right', va='bottom')
                axs[j, i].set_xlabel('Bandwidth $B$', labelpad=-2.0)
            axs[j, i].text(0.055, 0.945, "$\\mathbf{{{}}}_{}$".format(chr(ord('A') + i), j + 1), va='top', ha='left', fontsize=12, color='black', transform=ax.transAxes)
            axs[j, i].text(0.05, 0.95, "$\\mathbf{{{}}}_{}$".format(chr(ord('A') + i), j + 1), va='top', ha='left', fontsize=12, color='white', transform=ax.transAxes)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.25)


    return fig

utils.save(plot_2d_sweep_grid(utils.datafile('cerebellum_benchmark_data')))

