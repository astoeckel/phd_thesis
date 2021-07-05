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

import numpy as np
import h5py
import json
import matplotlib.pyplot as plt

# CONSTANTS

SWEEP_KEY_TO_LABEL = {
    "q": "State dimensionality $q$",
    "tau": "Time-constant $\\tau$ (ms)",
    "n_pcn_granule_convergence": "PCN $\\to$ Granule Conv.",
    "n_golgi_granule_convergence": "Golgi $\\to$ Granule Conv.",
    "bias_mode": None,
    "go_gr_ratio": "Golgi to Granule Ratio (%)",
    "solver_reg": "Regularisation factor",
    "solver_relax": "Subthreshold relaxation",
    "spatial_sigma": "Locality ($\\sigma$)",
    "pcn_xi_lower_bound": "PCN $x$-icpt. lower bound",
    "golgi_inh_bias_mode": "Golgi Bias Mode",
}

SWEEP_KEY_SORT_ORDER = {
    None: 0,
    "q": 1,
    "tau": 2,
    "bias_mode": 3,
    "go_gr_ratio": 4,
    "n_pcn_granule_convergence": 5,
    "n_golgi_granule_convergence": 6,
    "pcn_xi_lower_bound": 7,
    "solver_reg": 8,
    "solver_relax": 9,
    "spatial_sigma": 10,
    "golgi_inh_bias_mode": 11,
}

SWEEP_KEY_TRAFO = {
    "tau": {
        "mul": 1000.0,
    },
    "go_gr_ratio": {
        "mul": 100.0,
    },
    "bias_mode": {
        "map": {
            "uniform_pcn_intercepts": "Unif.\n",
            "realistic_pcn_intercepts": "Real.\n",
            "lugaro_uniform_pcn_intercepts": "Unif.\nLg",
            "lugaro_realistic_pcn_intercepts": "Real.\nLg",
            "jbias_uniform_pcn_intercepts": "Unif.\n$J_\\mathrm{bias}$",
            "jbias_realistic_pcn_intercepts": "Real.\n$J_\\mathrm{bias}$",
            "exc_jbias_uniform_pcn_intercepts": "Unif.\n$J_\\mathrm{bias}^+$",
            "exc_jbias_realistic_pcn_intercepts": "Real.\n$J_\\mathrm{bias}^+$",
            "inh_jbias_uniform_pcn_intercepts": "Unif.\n$J_\\mathrm{bias}^-$",
            "inh_jbias_realistic_pcn_intercepts": "Real.\n$J_\\mathrm{bias}^-$",
        },
    },
    "golgi_inh_bias_mode": {
          "map": {
              "none": "None",
              "recurrent": "Rec.",
              "lugaro": "Lg.",
              "recurrent_and_lugaro": "Rec. & Lg.",
          }
    },
    "solver_relax": {
        "map": {
            #False: "No relax.",
            #True: "With relax."
        }
    },
    "pcn_xi_lower_bound": {
        "map": lambda p: ('\n' if p[0] % 2 == 1 else '') + '{:0.2f}'.format(p[1]),
    }
}

MODE_TO_TITLE = {
    "direct": "Direct implementation",
    "single_population": "Single population",
    "two_populations": "Two populations",
    "two_populations_dales_principle": "Two populations\n(Dale's Principle)"
}

INPUT_TYPE_TO_LABEL = {
    "white_noise": "White noise",
    "pulse": "Rectangle pulses",
}

def plot_benchmark_result(filename, ax=None, title=None, show_title=True, do_plot=True, small=False, levels=None, cmap=None):

    # HELPER FUNCTIONS

    def unique(cback, iterable, expect_one=True):
        """Unique, converts many into one."""
        try:
            lst = list(map(cback, iterable))
        except KeyError:
            lst = []
        except Exception as e:
            raise e

        res = []
        if (len(lst) > 0):
            res.append(lst[0])
            for i in range(1, len(lst)):
                if not lst[i] in res:
                    res.append(lst[i])

        if expect_one:
            assert len(res) == 1, "Expected exactly one element"
            return res[0]
        else:
            return res

    def params(param):
        """Redistributes the parameters from a table into a dict."""
        idx, seed, delays, kwargs = param
        return {
            "idx": idx,
            "seed": seed,
            "delays": delays,
            "kwargs": kwargs
        }

    def is_lin(xs):
        return np.polyfit(np.arange(len(xs)), xs, 1, full=True)[1][0] < 1e-6

    def is_log(xs):
        return is_lin(np.log(xs))

    def is_numeric(xs):
        return all(map(lambda x: isinstance(x, (int, float, bool)), xs))

    # LOAD FILE

    # Load the dataset
    with h5py.File(filename, 'r') as f:
        Es = f["errors"][()]
        Ps = json.loads(f["errors"].attrs["params"])

    # RECONSTRUCT THE TYPE OF EXPERIMENT

    # Determine the network mode
    mode = unique(lambda param: params(param)["kwargs"]["mode"], Ps)

    # Determine the input type
    input_type = unique(lambda param: params(param)["kwargs"]["input_descr"][0], Ps)

    # Determine the list of delays
    delays = np.array(unique(lambda param: params(param)["delays"], Ps))

    # Get the list of input frequencies
    if input_type == "white_noise":
        ys = np.array(
            unique(lambda param: params(param)["kwargs"]["input_descr"][1], Ps, False))
    elif input_type == "pulse":
        ys = 100.0 * np.array(
            unique(lambda param: params(param)["kwargs"]["input_descr"][2], Ps, False))

    # Check for any of the other sweeps we may have been doing.
    # If no sweep is performed, make xs a one-element dummy array
    # in order to simplify the following code
    sweep_key = None
    xs = np.array((0,))
    for key in SWEEP_KEY_TO_LABEL.keys():
        xs_test = np.array(unique(lambda param: params(param)["kwargs"][key], Ps, False))
        if xs_test.size > 1:
            if not sweep_key is None:
                raise RuntimeError("Not implemented: sweeping over multiple parameters")
            sweep_key = key
            xs = xs_test

    # Make sure the Es matrix has the right shape
    assert Es.shape[0] == (xs.size * delays.size), filename
    Es = Es.reshape(xs.size, delays.size, ys.size, 2, -1)

    # Fetch the RMSE and the RMS
    Es_rmse = Es[:, :, :, 0, :]
    Es_rms = Es[:, :, :, 1, :]

    # ACTUAL PLOTTING CODE

#    if input_type == "white_noise":
#        # If we're in "white_noise" input mode, compute the NRMSE
#        errs = (Es_rmse / Es_rms).mean(axis=3)
#    else:
#        # Otherise scale by the mean of Es_rms accross all entries
#        errs = Es_rmse.mean(axis=3) / np.mean(Es_rms)
    errs = Es_rmse.mean(axis=3) / np.mean(Es_rms)

    # Create a new matplotlib axis if no target axis was given
    own_fig = False
    C = None
    if do_plot:
        if ax is None:    
            _, ax = plt.subplots(figsize=(3, 2.75))
            own_fig = True
        fig = ax.get_figure()
    else:
        fig, ax = None, None

    # No sweep, just a single experiment
    if xs.size == 1:
        if levels is None:
            levels = np.linspace(0, 1, 13)
        if do_plot:
            lw = 1.0 if small else 2.0
            C = ax.contourf(ys, delays, errs[0].T, vmin=np.min(levels), vmax=np.max(levels), levels=levels, cmap=cmap)
            ax.contour(ys, delays, errs[0].T, vmin=np.min(levels), vmax=np.max(levels),
                       colors=['white'], linestyles=['--'], linewidths=[lw], levels=levels)
            if is_log(ys):
                ax.set_xscale('log')

            if input_type == "white_noise":
                ax.set_xlabel('White Noise Bandwidth $B$')
            elif input_type == "pulse":
                ax.set_xlabel('Relative pulse width $t_\\mathrm{on}/\\theta$ (%)')
            ax.set_ylabel('Delay $\\theta\'/\\theta$')

        # Print the total error
        rmse = np.mean(errs)
        if do_plot:
            if small:
                plt.text(0.95, 0.95, "$\\langle E \\rangle = {:0.2f}$".format(rmse),
                         ha="right", va="top", bbox={
                             "facecolor": "white", "linewidth": 0, "pad": 2.0},
                         transform=ax.transAxes, fontsize=8)
            else:
                plt.text(0.05, 0.95, "$\langle E \\rangle = {:0.2f}$".format(rmse),
                         ha="left", va="top", bbox={
                             "facecolor": "white", "linewidth": 0},
                         transform=ax.transAxes)
    else:
        x_ticks_mul = 1.0
        x_ticks_map = {}
        if sweep_key in SWEEP_KEY_TRAFO:
            trafo = SWEEP_KEY_TRAFO[sweep_key]
            if "mul" in trafo:
                x_ticks_mul = trafo["mul"]
            if "map" in trafo:
                x_ticks_map = trafo["map"]

        if do_plot:
            ax.boxplot(errs.reshape(xs.size, -1).T,
                       showfliers=False)

        x_tick_idcs = np.arange(1, len(xs) + 1, dtype=np.int)
        if x_ticks_mul != 1.0:
            x_ticks = xs * x_ticks_mul
        else:
            x_ticks = xs
        if callable(x_ticks_map):
            x_ticks = list(map(x_ticks_map, enumerate(x_ticks)))
        if is_numeric(x_ticks): # Convert to string
            if np.median(x_ticks) < 1 and is_log(x_ticks):
                x_ticks = np.log10(x_ticks)
                x_ticks = list(map(lambda x: "$10^{{{:0.1f}}}$".format(x[1]) if x[0] % 2 == 0 else "", enumerate(x_ticks)))
            else:
                x_ticks = list(map(lambda x: str(int(x)), x_ticks))
        for i, tick in enumerate(x_ticks):
            if (not callable(x_ticks_map)) and (tick in x_ticks_map):
                x_ticks[i] = x_ticks_map[tick]
        while len(x_ticks) > 5:
            x_tick_idcs = x_tick_idcs[::2]
            x_ticks = x_ticks[::2]

        if do_plot:
            ax.set_xticks(x_tick_idcs)
            ax.set_xticklabels(x_ticks)
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel('Normalised RMSE $E$')
            ax.set_xlabel(SWEEP_KEY_TO_LABEL[sweep_key])

            plt.text(0.95, 0.95, INPUT_TYPE_TO_LABEL[input_type],
                     ha="right", va="top", bbox={
                         "facecolor": "white", "linewidth": 0},
                     fontsize=8,
                     transform=ax.transAxes)


    if (title is None) and (mode in MODE_TO_TITLE):
        title = MODE_TO_TITLE[mode]
        if ("detailed" in filename) and mode == "two_populations_dales_principle":
            title = title.split(")")[0] + "; detailed)"
    if show_title and (not title is None) and do_plot:
        ax.set_title(title)

    if own_fig:
        fig.tight_layout()

    descr = {
        "title": title,
        "input_type": input_type,
        "is_sweep": not sweep_key is None,
        "sweep_key": sweep_key,
        "mode": mode,
    }

    return fig, ax, descr, C

