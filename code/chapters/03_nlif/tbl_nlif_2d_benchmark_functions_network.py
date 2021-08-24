#!/usr/bin/env python3

#    Code for the "Nonlinear Synaptic Interaction" Paper
#    Copyright (C) 2017-2020   Andreas Stöckel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
Manually run this code.
"""

import sys
import numpy as np
import h5py


def decode_all(xs):
    return [str(x, "utf-8") for x in xs]


FORMULAS = {
    "addition": "x_1 + x_2",
    "multiplication": "x_1 \\times x_2",
    "multiplication_limited": "x_1 \\times x_2",
    "sqrt-multiplication": "\\sqrt{x_1 \\times x_2}",
    "sqr-multiplication": "(x_1 \\times x_2) ^ 2",
    "shunting": "x_1 / (1 + x_2)",
    "norm": "\\|(x_1, x_2)\\|",
    "arctan": "\\mathrm{atan}(x_1, x_2)",
    "half-max": "x_1 (x_1 > x_2)",
    "max": "\\max(x_1, x_2)",
    "softmax": "\\mathrm{smax}(x_1, x_2)",
    "bell": "e^{-\\|\\vec x\\|}",
}

LIMITED_DOMAIN = {
    "multiplication_limited", "sqrt-multiplication", "shunting"
}

FUNCTIONS_SORT = [
    "addition",
    "shunting",
    "sqrt-multiplication",
    "multiplication_limited",
    "multiplication",
    "sqr-multiplication",
    "norm",
    "arctan",
    "max",
]

TITLES = {
    "lif": ["LIF", "standard"],
    "lif_2d": ["LIF", "two layers"],
    "two_comp": ["$n$-LIF", "$n = 2$"],
    "three_comp": ["$n$-LIF", "$n = 3$"],
    "four_comp": ["$n$-LIF", "$n = 4$"],
}

SUBTITLES = [
    "\\textbf{Standard parameters} ($\\lambda = 10^{-3}$; $\\xi_0 \\in [-0.95, 0.95]$; with Dale's principle, $p_\\mathrm{inh} = 30\%$)",
    "\\textbf{Adapted parameters} ($\\lambda = 10^{-6}$; $\\xi_0 \\in [-0.95, 0]$; no Dale's principle)",
]

errs, n_repeat = None, 0
for fn in sys.argv[1:]:
    with h5py.File(sys.argv[1], 'r') as f:
        benchmark_functions = decode_all(f["function_keys"][()].split(b"\n"))
        neurons = decode_all(f["neurons"][()].split(b"\n"))
        if errs is None:
            errs = f['errs'][()]
        else:
            errs = np.concatenate((errs, f['errs'][()]), axis=-2)
        n_neurons, n_params, n_funs, n_repeat_, n_errs = errs.shape
        n_repeat += n_repeat_


def print_header():
    print("\\documentclass[10pt,a3paper]{article}")
    print("\\usepackage[top=1cm,left=1.5cm,bottom=1cm,right=1.5cm]{geometry}")
    print("\\usepackage{siunitx}")
    print("\\usepackage{booktabs}")
    print("\\usepackage{stix2}")
    print("\\usepackage{multirow}")
    print("\\usepackage[svgnames,table]{xcolor}")
    print("\\begin{document}")
    print("\\thispagestyle{empty}")
    print("\\begin{center}\\texttt{" +
          (" ".join(sys.argv)).replace("_", "\\_") + "}\\\\[1cm]\\end{center}")


def print_footer():
    print("\\end{document}")


def print_table(funs, tbl):
    def shade(i, row, shade_style="\\cellcolor{{White!{}!SteelBlue}}"):
        row = np.round(row, 1)
        perc = 100 * np.where(np.sort(row) == row[i])[0][0] / len(row)
        best = row[i] == np.min(row)

        style_begin = shade_style.format(100 - int(perc)) + "{$"
        style_end = "$}"
        if best:
            style_begin = style_begin + "\\mathbf{"
            style_end = "}" + style_end

        return style_begin, style_end

    print("\\begin{table}")
    print("\\centering\\vspace{0.5cm}")
    print("\\small")
    print("\\sffamily")
    print("\\begin{tabular}{r r " + ("r " * n_neurons) + "}")
    print("\\toprule")
    print("\\textbf{Function} & \\textbf{Domain} & \\multicolumn{" + str(n_neurons) +
          "}{c}{\\textbf{Neuron}} \\\\")
    print("\\cmidrule(r){1-1}\\cmidrule(r){2-2}\\cmidrule{3-" + str(2 + n_neurons) + "}")
    print("&")
    for i in range(n_neurons):
        print("&\t \\multicolumn{1}{c}{" + TITLES[neurons[i]][0] + "}")
    print("\\\\")
    print("&")
    for i in range(n_neurons):
        print("&\t \\multicolumn{1}{c}{" + TITLES[neurons[i]][1] + "}")
    print("\\\\")
    print("%--------------------------------------------")
    last_i_param = None
    for i_row, (i_param, i_fun) in enumerate(funs):
        if i_param != last_i_param:
            print("\\midrule")
            print("\\multicolumn{" + str(n_neurons + 2) + "}{c}{" +
                  SUBTITLES[i_param] + "} \\\\")
            last_i_param = i_param
            print("\\midrule")

        fun_key = benchmark_functions[i_fun]
        print("$" + FORMULAS[fun_key] + "$")
        if fun_key in LIMITED_DOMAIN:
            print("& $[0, 1]^2$")
        else:
            print("& $[-1, 1]^2$")

        row = tbl[i_row, :]
        for i_col in range(n_neurons):
            if row.shape[1] == 1:
                statistic = "{:0.1f} \\%".format(row[i_col, 0] * 100.0)
            elif row.shape[1] == 2:
                statistic = "{:0.1f} \\pm {:0.1f} \\%".format(
                    row[i_col, 0] * 100.0, row[i_col, 1] * 100)
            style_begin, style_end = shade(i_col, 100 * row[:, 0])
            print("& " + style_begin + statistic + style_end)
        print("\\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("%--------------------------------------------")
    print("\\end{table}")
    print("")
    print("")


# Assemble the actual table
rows_err_net, rows_err_model, rows_err_net_min = [], [], []
row_funs = []
for i_param in range(n_params):
    for fun_key in FUNCTIONS_SORT:
        if not fun_key in benchmark_functions:
            continue
        else:
            i_fun = benchmark_functions.index(fun_key)
        if np.all(np.isnan(errs[:, i_param, i_fun])):
            continue
        column_err_net, column_err_model, column_err_net_min = [], [], []
        for i_neuron in range(n_neurons):
            errs_slice = errs[i_neuron, i_param, i_fun]
            column_err_net.append(
                np.array((np.nanmean(errs_slice[:, 1]),
                          np.nanstd(errs_slice[:, 1]))).T)
            column_err_model.append(
                np.array((np.nanmean(errs_slice[:, 0]),
                          np.nanstd(errs_slice[:, 0]))).T)
            column_err_net_min.append(
                np.array((np.nanmin(errs_slice[:, 1]),)).T)
        row_funs.append((i_param, i_fun))
        rows_err_net.append(column_err_net)
        rows_err_model.append(column_err_model)
        rows_err_net_min.append(column_err_net_min)

rows_err_net = np.array(rows_err_net)
rows_err_model = np.array(rows_err_model)
rows_err_net_min = np.array(rows_err_net_min)

print_header()
print_table(row_funs, rows_err_net)
print_table(row_funs, rows_err_model)
print_table(row_funs, rows_err_net_min)
print_footer()

