#   This file is part of NEF Synaptic Computation
#   (c) Andreas Stöckel 2017, 2018
#
#   NEF Synaptic Computation is free software: you can redistribute it and/or
#   modify it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   NEF Synaptic Computation is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License along with
#   NEF Synaptic Computation.  If not, see <http://www.gnu.org/licenses/>.

from . import magic

def generate_neuron_graph(compartments, connections, f):
    """
    Writes the neuron topology to a graphviz dot file which can be converted
    to an SVG, PDF or other vector graphics formats.

    f: is a file-like object to which the DOT description will be written.
    """

    def _fmt(x, scale, unit=None):
        if isinstance(x, magic.Number):
            s = "{:0.2f}".format(x * scale).rstrip("0").rstrip(".")
            if unit:
                s += " " + unit
            return s
        else:
            return repr(x)

    def _dot_elem(name, properties=None):
        first = True
        f.write("\t" + name + " [")
        if not properties is None:
            for key, value in properties.items():
                if not first:
                    f.write(",")
                first = False
                if isinstance(value, str):
                    if len(value) > 0 and value[0] == '<' and value[-1] == '>':
                        f.write(key + "=" + value + "")
                    else:
                        f.write(key + "=\"" + value + "\"")
                else:
                    f.write(key + "=" + str(value))
        f.write("];\n")

    # Fetch the soma compartment in order to detect excitatory and inhibitory
    # neurons
    soma = compartments[0]

    # Write the DOT file header and some default settings for nodes
    # and edges
    f.write("digraph neuron {\n")
    f.write("\toverlap=false;\n")
    f.write("\toutputorder=\"edgesfirst\";\n")
    _dot_elem("node", {
        "fontsize": 11.0,
        "fixedsize": "shape",
        "label": ""
    });
    _dot_elem("edge", {
        "fontsize": 11.0,
        "dir": "none"
    });

    # Draw compartments and associated channels
    cmp_map = dict()
    idx = 0
    for compartment in compartments:
        f.write("\n")
        cmp_map[compartment] = idx
        if compartment.is_soma:
            _dot_elem("n_" + str(idx), {
                "shape": "circle",
                "width": 0.25,
                "height": 0.25,
            })
        else:
            _dot_elem("n_" + str(idx), {
                "shape": "circle",
                "style": "filled",
                "fillcolor": "black",
                "width": 0.15,
                "height": 0.15,
            })

        info = []
        if compartment.is_soma:
            if not compartment.v_th is None:
                info.append("<I>v</I><SUB>th</SUB> = {}".format(_fmt(compartment.v_th, 1e3, "mV")))
            if not compartment.tau_ref is None:
                info.append("<I>τ</I><SUB>ref</SUB> = {}".format(_fmt(compartment.tau_ref, 1e3, "mS")))
        if not compartment.C_m is None:
            info.append("<I>C</I><SUB>m</SUB> = {} ".format(_fmt(compartment.C_m, 1e9, "nF")))

        label = "<<B>" + compartment.label + "</B><BR/><FONT POINT-SIZE=\"9\">" + "<BR/>".join(
            info) + "</FONT>>"

        _dot_elem("n_label_" + str(idx), {
            "shape": "box",
            "style": "filled",
            "fillcolor": "#f0f0f0",
            "penwidth": 0.0,
            "label": label,
        })
        _dot_elem("n_{} -> n_label_{}".format(idx, idx), {
            "penwidth": 0.0
        });

        for i, channel in enumerate(
                sorted(compartment.channels, key=lambda x: (x.is_static, x.label))):
            cidx = idx + i + 1
            shape = "square" if channel.type == "cur" else "circle"
            info = []
            if channel.type == "cond":
                info.append("{}".format(_fmt(channel.E_rev, 1e3, "mV")))
                if not channel.g is None:
                    info.append("{}".format(_fmt(channel.g, 1e9, "nS")))
            if channel.type == "cur":
                info.append("×{}".format(_fmt(channel.mul)))
                if not channel.J is None:
                    info.append("{}".format(_fmt(channel.J, 1e9, "nA")))

            label = "J" if channel.label == "j" else channel.label
            label = "<<I>" + label + "</I><BR/><FONT POINT-SIZE=\"9\">" + "<BR/>".join(
                info) + "</FONT>>"
            if channel.is_static:
                _dot_elem("n_" + str(cidx), {
                    "shape": shape,
                    "style": "filled",
                    "fillcolor": "black",
                    "width": 0.1,
                    "height": 0.1
                })
            else:
                _dot_elem("n_" + str(cidx),
                          {"shape": shape,
                           "width": 0.1,
                           "height": 0.1})
            _dot_elem("n_label_" + str(cidx),
                      {"shape": "none",
                       "label": label,
                       "margin": "0,0"})
            _dot_elem("n_{} -> n_label_{}".format(cidx, cidx), {
                "penwidth": 0.5,
                "len": 0.75,
                "dir": "back",
                "arrowsize": 0 if channel.is_static else 0.5
            });

        for i, channel in enumerate(
                sorted(compartment.channels, key=lambda x: x.label)):
            cidx = idx + i + 1
            if channel.is_static and channel.type == "cond":
                _dot_elem("n_{} -> n_{}".format(idx, cidx))
            else:
                inhibitory = False
                try:
                    inhibtory = channel.is_inhibitory(soma.v_th)
                except TypeError:
                    if not channel.is_symbolic:
                        raise
                _dot_elem("n_{} -> n_{}".format(idx, cidx), {
                    "dir": "back",
                    "arrowsize": 0.5,
                    "arrowtail": "dot" if inhibitory else "normal"
                });

        idx += len(compartment.channels) + 1

    # Draw the connections between compartments
    n = len(compartments)
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in connections:
                i0 = cmp_map[compartments[i]]
                i1 = cmp_map[compartments[j]]
                for k, con in enumerate(connections[(i, j)]):
                    _dot_elem(f"n_{i0}_{i1}_{k}", {
                        "shape": "square",
                        "style": "filled",
                        "fillcolor": "black",
                        "width": 0.1,
                        "height": 0.1
                    })
                    _dot_elem(f"n_{i0}_{i1}_{k}_label", {
                        "shape":
                        "none",
                        "label":
                        "<<FONT POINT-SIZE=\"9\">{}</FONT>>".format(_fmt(
                            con.g_c, 1e9, "nS")),
                        "margin":
                        "0,0"
                    })
                    _dot_elem(f"n_{i0} -> n_{i0}_{i1}_{k} -> n_{i1}", {
                        "penwidth": 1.5
                    })
                    _dot_elem(f"n_{i0}_{i1}_{k} -> n_{i0}_{i1}_{k}_label", {
                        "penwidth": 0.5,
                        "len": 0.75,
                        "dir": "back",
                        "arrowsize": 0.3
                    });

    f.write("}\n")
