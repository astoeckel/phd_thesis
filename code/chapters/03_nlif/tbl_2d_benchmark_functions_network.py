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

FORMULAS = {
    "addition": "x_1 + x_2",
    "multiplication": "x_1 \\times x_2",
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

TITLES = {
    "linear": ["%0\nLIF", "standard"],
    "linear_2d": ["%0\nLIF", "two layers"],
    "gc50_no_noise":
    ["%1\nTwo comp. LIF $c_{12} = \\SI{50}{\\nano\\siemens}$", "%0\nstandard"],
    "gc50_noisy": [
        "%1\nTwo comp. LIF $c_{12} = \\SI{50}{\\nano\\siemens}$",
        "%1\nnoise model"
    ],
    "gc100_no_noise":
    ["%2\nTwo comp. LIF $c_{12} = \\SI{100}{\\nano\\siemens}$", "%0\nstandard"],
    "gc100_noisy": [
        "%2\nTwo comp. LIF $c_{12} = \\SI{100}{\\nano\\siemens}$",
        "%1\nnoise model"
    ],
    "gc200_no_noise":
    ["%3\nTwo comp. LIF $c_{12} = \\SI{200}{\\nano\\siemens}$", "%0\nstandard"],
    "gc200_noisy": [
        "%3\nTwo comp. LIF $c_{12} = \\SI{200}{\\nano\\siemens}$",
        "%1\nnoise model"
    ],
}


def title(setup):
    title, subtitle = TITLES[params_keys[setup[0]]]
    if setup[2] == 0:
        subtitle += "\\textsuperscript{\\dag}"
#        subtitle = ", ".join(list(filter(len, [subtitle])) + ["relax."])
    return (title, subtitle)


if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " <H5 FILE>")
    sys.exit(1)

def decode_all(xs):
    return [str(x, "utf-8") for x in xs]

with h5py.File(sys.argv[1], 'r') as f:
    benchmark_functions = decode_all(f["function_keys"][()].split(b"\n"))
    params_keys = decode_all(f["params_keys"][()].split(b"\n"))
    errs = np.array(f['errs'])
    n_params, n_funs, n_filt, n_repeat, n_errs = errs.shape


def filter_data(flt_fun, errs):
    # Select the setups that pass the filter
    idcs = []
    for i in range(errs.shape[0]):
        for j in range(errs.shape[2]):
            for k in [1, 0]:
                if (flt_fun((i, j, k))):
                    idcs.append((i, j, k))

    # Sort the indices according to the order in the "TITLES" array
    idcs = list(sorted(idcs, key=lambda idx: title(idx)))

    return np.array([errs[i, :, j, :, k::2] for i, j, k in idcs]), idcs


def fixparens(s):
    n_open = s.count("{")
    n_close = s.count("}")
    return "{" * max(0, n_close - n_open) + s + "}" * max(0, n_open - n_close)

def print_tbl(caption, errs, setups, print_Emodel=False, print_Epop=False):
    print("\\begin{table}")
    print("\\caption{{{}}}".format(caption))
    print("\\centering\\vspace{0.5cm}")
    print("%--------------------------------------------")

    print("\\small")
    print("\\sffamily")
    print("\\begin{tabular}{p{2.2cm} p{1.7cm} l " +
          ("r " * n_funs) + "}")
    print("\\toprule")
    print("\\multicolumn{3}{c}{\\textbf{Experiment setup}} & \\multicolumn{" +
          str(n_funs) + "}{c}{{\\textbf{Target Functions}}}\\\\")
    print("\\cmidrule(r){1-3}\\cmidrule(l){4-" + str(3 + n_funs) + "}")
    print("& & & $" + "$ & $".join([FORMULAS[x]
                                    for x in benchmark_functions]) + "$\\\\")
    print("\\midrule")

    stat_fn = np.nanmean

    vspan = 0
    i_setup = 0
    n_rows = 1 + (1 if print_Emodel else 0) + (1 if print_Epop else 0)
    for i, setup in enumerate(setups):
        if not setup:
            continue
        setup = tuple(setup)

        tCol1 = title(setup)[0]
        tCol2 = title(setup)[1]

        if vspan == 0:
            for j in range(i, len(setups)):
                if not setups[j]:
                    continue
                setup_next = tuple(setups[j])
                if tCol1 == title(setup_next)[0]:
                    vspan += 1
            print("\\multirow{" + str(vspan * n_rows) +
                  "}{2.2cm}{\\raggedleft " + tCol1 + "} &")
            vspan -= 1
        else:
            print("&")
            vspan -= 1
        print("\\multirow{" + str(n_rows) + "}{1.7cm}{\\raggedleft " + tCol2 +
              "} &")

        def dump_row(k_err, style="", best_style="\\mathbf{", shade_style=None):
            data = stat_fn(errs[i_setup, :, :, k_err], axis=1) * 100.0
            data_stddev = np.std(errs[i_setup, :, :, k_err], axis=1) * 100.0

            def is_best(i, x):
                min_ = np.min(stat_fn(errs[:, i, :, k_err], axis=1) * 100.0)
                if x == min_:
                    return best_style
                else:
                    return ""

            def shade(i, x):
                if shade_style is None:
                    return ""
                perc = np.where(
                    np.sort(stat_fn(errs[:, i, :, k_err], axis=1) *
                            100.0) == x)[0][0] / errs.shape[0] * 100
                return shade_style.format(100 - int(perc))

            print(" & ".join([
                style + shade(i, x) + "$" + fixparens(is_best(i, x) +
                "{:0.1f} \\pm {:0.1f}\\%".format(x, data_stddev[i])) + "$"
                for i, x in enumerate(data)
            ]))

        first = True

        if print_Emodel:
            if not first:
                print("& & ")
            first = False
            print("$E_\\mathrm{model}$ & ")
            dump_row(1, "\\color{Gray}")
            print("\\\\")


#        if print_Epop:
#            if not first:
#                print("& & ")
#            first = False
#            print("$E_\\mathrm{pop}$ &")
#            dump_row(1, "\\color{Gray}")
#            print("\\\\")

        if not first:
            print("& & ")
        first = False
        if n_rows == 1:
            print(" &")
        else:
            print("$E_\\mathrm{net}$ &")
        dump_row(0, "", "\\mathbf{", "\\cellcolor{{White!{}!SteelBlue}}")

        if i + 1 == len(setups):
            print("\\\\")
        else:
            if vspan == 0:
                print("\\\\\\midrule")
            else:
                if n_rows > 1:
                    print("\\\\\\cmidrule(l){2-" + str(3 + n_funs) + "}")
                else:
                    print("\\\\")

        i_setup += 1

    print("\\bottomrule")
    print("\\end{tabular}")
    print("%--------------------------------------------")
    print("\\end{table}")


def print_tbl_transposed(caption, errs, setups):
    print("\\begin{table}")
    print("\\caption{{{}}}".format(caption))
    print("\\centering\\vspace{0.5cm}")
    print("%--------------------------------------------")

    n_setups = len(setups)

    print("\\small")
    print("\\sffamily")
    print("\\begin{tabular}{r " + ("r " * n_setups) + "}")
    print("\\toprule")
    print("\\textbf{Target}& \\multicolumn{" + str(n_setups) +
          "}{c}{\\textbf{Experiment setup}} \\\\")
    print("\\cmidrule(r){1-1}\\cmidrule(l){2-" + str(n_setups + 1) + "}")
    i, j = 0, 0
    while i < len(setups):
        while j < len(setups) and (title(setups[j])[0] == title(setups[i])[0]):
            j += 1
        print("& \\multicolumn{" + str(j - i) + "}{c}{" + title(setups[i])[0] +
              "}")
        i = j
    print("\\\\")
    i, j = 0, 0
    while i < len(setups):
        while j < len(setups) and (title(setups[j])[0] == title(setups[i])[0]):
            j += 1
        print("\\cmidrule(" + ("l" if i + 1 < len(setups) else "r") + "){" +
              str(i + 2) + "-" + str(j + 1) + "}")
        i = j
    for i, _ in enumerate(setups):
        print("& " + title(setups[i])[1])
    print("\\\\")
    print("\\midrule")

    stat_fn = np.nanmean

    for i, benchmark_function in enumerate(benchmark_functions):
        print("$" + FORMULAS[benchmark_function] + "$ ")
        for j, setup in enumerate(setups):
            data = stat_fn(errs[j, i, :, 0]) * 100.0
            data_stddev = np.std(errs[j, i, :, 0]) * 100.0

            def is_best(x):
                return "\\mathbf{" if x == np.min(
                    stat_fn(errs[:, i, :, 0], axis=1) * 100.0) else ""

            def shade(x):
                perc = np.where(
                    np.sort(stat_fn(errs[:, i, :, 0], axis=1) *
                            100.0) == x)[0][0] / errs.shape[0] * 100
                return "\\cellcolor{{White!{}!SteelBlue}}".format(100 -
                                                                  int(perc))

            print("& " + shade(data) + "$" + fixparens(is_best(data) +
                  "{:0.1f} \\pm {:0.1f}\\%".format(data, data_stddev)) + "$")
        print("\\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("%--------------------------------------------")
    print("\\end{table}")
    print("")
    print("")


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


errs_no_flt, setups_no_flt = filter_data(lambda idx: idx[1] == 0, errs)
errs_flt, setups_flt = filter_data(lambda idx: idx[1] == 1, errs)

errs_no_flt_short, setups_no_flt_short = filter_data(
    lambda idx: (idx[1] == 0) and (not "gc200" in params_keys[idx[0]]) and
    (idx[2] == 0 or params_keys[idx[0]] == "linear"), errs)
errs_flt_short, setups_flt_short = filter_data(
    lambda idx: (idx[1] == 1) and (not "gc200" in params_keys[idx[0]]) and
    (idx[2] == 0 or params_keys[idx[0]] == "linear"), errs)

print_header()
print_tbl("Errors without filtering",
          errs_no_flt,
          setups_no_flt,
          print_Emodel=True)
print_tbl("Errors with filtering", errs_flt, setups_flt, print_Emodel=True)
print_tbl_transposed("Errors without filtering (short)", errs_no_flt_short,
                     setups_no_flt_short)
print_tbl_transposed("Errors with filtering (short)", errs_flt_short,
                     setups_flt_short)
print_footer()

