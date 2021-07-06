#!/usr/bin/env python3

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
"""
This script wraps the Python scripts generating the figures found in this
thesis. This script provides some convenient imports and utility functions
that can be used directly from the figure generation scripts.
"""

import sys, os


def execute_figure_script(utils):
    # Import all the matplotlib shennanigans
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import h5py
    import json

    # Load the correct matplotlibrc
    mpl.rc_file(os.path.join(os.path.dirname(__file__), 'matplotlibrc'))
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["text.latex.preamble"] = r"""
        \usepackage{siunitx}
        \usepackage{libertine}
        \usepackage{libertinust1math}
        \usepackage{mathrsfs}
        \renewcommand*\familydefault{\sfdefault}
        \renewcommand{\vec}[1]{\mathbf{#1}}
        \newcommand{\mat}[1]{\mathbf{#1}}
    """

    # Assemble a bunch of stuff being passed as globals to the target script
    global_vars = {
        "FIG_WIDTH": 7.2,
        "PAGE_WIDTH": 6.2,
        "np": np,
        "mpl": mpl,
        "plt": plt,
        "cm": cm,
        "h5py": h5py,
        "json": json,
        "utils": utils,
    }

    # Append the script directory to the Python path
    sys.path.append(utils.scriptdir)

    # Append the library directory (contains shared code)
    sys.path.append(utils.libdir)

    # Execute the python file
    import runpy
    runpy.run_path(utils.script, global_vars)


def main():
    import argparse
    import traceback
    import subprocess
    import re

    # Parse the arguments
    parser = argparse.ArgumentParser(description="Figure generation runner")
    parser.add_argument('script', type=str, nargs='?')
    parser.add_argument(
        '--datafile_cache',
        default="",
        type=str,
        help="If set, reads/writes the datafile table to the given file.")
    parser.add_argument('--script',
                        dest="script2",
                        type=str,
                        help="Figure script")
    parser.add_argument('--target',
                        type=str,
                        help="File to save the resulting figure to")
    parser.add_argument('--view',
                        action='store_true',
                        default=False,
                        help="File to save the resulting figure to")
    args = parser.parse_args()

    # Make sure either script2 or script is set; the positional argument was
    # added later to this script as a convenience.
    if (args.script is None) == (args.script2 is None):
        print("Must either provide the positional or explicit script argument")
        sys.exit(1)
    args.script = args.script2 if (args.script is None) else args.script

    # Automatically generate the target file name if none is given
    if (not args.view) and (args.target is None):
        args.target = os.path.splitext(args.script)[0] + ".pdf"
    elif args.view:
        args.target = '/tmp/' + os.path.splitext(os.path.basename(
            args.script))[0] + ".pdf"

    # Make the target and script directories absolute
    args.script = os.path.abspath(args.script)
    if not args.target is None:
        args.target = os.path.abspath(args.target)

    # Construct the utilities class that is being passed to the target script
    class Utils:

        blues = ["#729fcfff", "#3465a4ff", "#204a87ff"][::-1]
        reds = ["#ef2929ff", "#cc0000ff", "#a40000ff"][::-1]
        greens = ["#8ae234ff", "#73d216ff", "#4e9a06ff"][::-1]
        oranges = ["#fcaf3eff", "#f57900ff", "#ce5c00ff"][::-1]
        purples = ["#ad7fa8ff", "#75507bff", "#5c3566ff"][::-1]
        yellows = ["#fce94fff", "#edd400ff", "#c4a000ff"][::-1]
        grays = [
            "#eeeeecff", "#d3d7cfff", "#babdb6ff", "#888a85ff", "#555753ff",
            "#2e3436ff"
        ][::-1]

        def run_with_cache(self, fun, *args, **kwargs):
            import numpy as np
            import os
            import types
            import pickle
            import hashlib
            import string
            import random

            if isinstance(fun, types.FunctionType):
                obj = None
            elif isinstance(fun, types.MethodType):
                obj = fun.__self__
            else:
                raise Exception("fun must be a function or a bound method!")

            m = hashlib.sha256()
            m.update(fun.__name__.encode("utf-8"))
            m.update(fun.__module__.encode("utf-8"))
            m.update(pickle.dumps(obj))
            m.update(pickle.dumps(args))
            m.update(pickle.dumps(kwargs))

            rnd_str = ''.join(
                random.choice(string.ascii_lowercase) for x in range(10))

            fn_base, _ = os.path.splitext(os.path.basename(self.script))
            fn = self.datafile(
                os.path.join("cache", fn_base,
                             fn_base + "_" + m.hexdigest()[:8] + ".npz"))
            fn_tmp = self.datafile(
                os.path.join("cache", fn_base,
                             fn_base + "_" + rnd_str + ".npz"))
            os.makedirs(os.path.dirname(fn), exist_ok=True)

            if os.path.isfile(fn):
                print(f"Loading intermediate result from cache file {fn}...")
                data = np.load(fn, allow_pickle=True)
                return data["res"]
            else:
                data = fun(*args, **kwargs)
                print(f"Storing intermediate result in cache file {fn}...")
                try:
                    np.savez(fn_tmp, res=data)
                    try:
                        os.rename(fn_tmp, fn)
                    except FileExistsError:
                        pass
                finally:
                    try:
                        os.unlink(fn_tmp)
                    except FileNotFoundError:
                        pass
                return data

        def save(self, fig, suffix=""):
            # Special treatment for PDFs. We need to run the resulting PDF
            # through Ghostscript to
            # a) trim the figures properly
            # b) subset fonts
            target_file, target_ext = os.path.splitext(args.target)
            target_file += suffix
            if target_ext == ".pdf":
                target = target_large = target_file + ".large" + target_ext
            else:
                target = target_file + target_ext
            print("Saving to", target)
            fig.savefig(target,
                        bbox_inches='tight',
                        pad_inches=0.05,
                        transparent=True)

            if target_ext == ".pdf":
                # Extract the bounding box
                print("Extracting bounding box of file", target_large)
                gs_out = subprocess.check_output(
                    ["gs", "-o", "-", "-sDEVICE=bbox", target_large],
                    stderr=subprocess.STDOUT)
                pattern = r"^%%HiResBoundingBox:\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$"
                x0, y0, x1, y1 = map(
                    float,
                    re.search(pattern, str(gs_out, "utf-8"),
                              re.MULTILINE).groups())

                # Add a small extension to the bounding box
                pad = 0.5
                x0 -= pad
                x1 += pad
                y0 -= pad
                y1 += pad

                # Run ghostscript again to crop the file
                # See https://stackoverflow.com/a/46058965
                target = target_file + target_ext
                print("Optimising PDF and saving to", target)
                subprocess.check_output([
                    "gs", "-o", target, "-dEmbedAllFonts=true",
                    "-dSubsetFonts=true", "-dCompressFonts=true",
                    "-dPDFSETTINGS=/prepress", '-dDoThumbnails=false',
                    "-sDEVICE=pdfwrite", f"-dDEVICEWIDTHPOINTS={x1 - x0}",
                    f"-dDEVICEHEIGHTPOINTS={y1 - y0}", "-dFIXEDMEDIA", "-c",
                    f"<</PageOffset [-{x0} -{y0}]>>setpagedevice", "-f",
                    target_large
                ])

                # Remove the large temporary file
                os.unlink(target_large)

                # Open the system viewer
                if args.view:
                    os.spawnv(os.P_NOWAIT, '/usr/bin/env',
                              ['/usr/bin/env', 'xdg-open', target])

        @property
        def script(self):
            return args.script

        @property
        def target(self):
            return args.target

        @property
        def logfile(self):
            return os.path.splitext(args.script)[0] + ".log"

        @property
        def rootdir(self):
            return os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', 'media'))

        @property
        def datadir(self):
            return os.path.abspath(os.path.join(self.rootdir, "..", "data"))

        @property
        def libdir(self):
            return os.path.abspath(
                os.path.join(self.rootdir, "..", "code", "lib"))

        def datafile(self, fn):
            import json

            cache_fn = None
            if args.datafile_cache:
                cache_fn = os.path.abspath(os.path.join(self.rootdir, '..', args.datafile_cache))

            if cache_fn and os.path.isfile(cache_fn):
                datafiles = json.load(open(cache_fn, "r"))
            else:
                import run_experiments
                manifest, _, root = run_experiments.load_manifest()
                datafiles = run_experiments.target_map(manifest, root)
                if args.datafile_cache:
                    json.dump(datafiles, open(cache_fn, "w"))

            if fn in datafiles:
                path = datafiles[fn]
            else:
                path = os.path.join(self.datadir, fn)
                path = os.path.relpath(os.path.abspath(path),
                                       os.path.abspath(os.path.curdir))

            print("Datafile", fn, "->", path)
            return path

        @property
        def scriptdir(self):
            return os.path.dirname(args.script)

        def outside_ticks(self, ax):
            ax.tick_params(direction="out", which="both")

        def remove_frame(self, ax):
            for spine in ["left", "right", "top", "bottom"]:
                ax.spines[spine].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticks([], minor=True)
            ax.set_yticks([], minor=True)

        def add_frame(self, ax):
            for spine in ["left", "right", "top", "bottom"]:
                ax.spines[spine].set_visible(True)

        def annotate(self,
                     ax,
                     x0,
                     y0,
                     x1,
                     y1,
                     s=None,
                     ha="center",
                     va="center",
                     fontdict=None,
                     zorder=None):
            ax.plot([x0, x1], [y0, y1],
                    color='k',
                    linewidth=0.5,
                    linestyle=(0, (1, 1)),
                    clip_on=False,
                    zorder=zorder)
            ax.plot([x0], [y0],
                    'ko',
                    markersize=1,
                    clip_on=False,
                    zorder=zorder)
            if not s is None:
                ax.text(x1,
                        y1,
                        s,
                        ha=ha,
                        va=va,
                        bbox={
                            "pad": 1.0,
                            "color": "w",
                            "linewidth": 0.0,
                        },
                        fontdict=fontdict,
                        zorder=zorder)

        def vslice(self, ax, x, y0, y1, **kwargs):
            ax.plot([x, x], [y0, y1],
                    'k-',
                    linewidth=0.75,
                    clip_on=False,
                    **kwargs)
            ax.plot(x, y0, 'k_', linewidth=0.75, clip_on=False, **kwargs)
            ax.plot(x, y1, 'k_', linewidth=0.75, clip_on=False, **kwargs)

        def hslice(self, ax, x0, x1, y, **kwargs):
            ax.plot([x0, x1], [y, y],
                    'k-',
                    linewidth=0.75,
                    clip_on=False,
                    **kwargs)
            ax.plot(x0, y, 'k|', linewidth=0.75, clip_on=False, **kwargs)
            ax.plot(x1, y, 'k|', linewidth=0.75, clip_on=False, **kwargs)

        def timeslice(self, ax, x0, x1, y, **kwargs):
            self.hslice(ax, x0, x1, y, **kwargs)

    # Create an instance of the utils class
    utils = Utils()

    # Set the working directory to wherever the target script is located
    os.chdir(utils.scriptdir)

    # Execute the script, if something goes wrong, print the error to the screen
    # and write it to a log file. Make sure that the output file will not exist.
    try:
        execute_figure_script(utils)

        # Delete the log file if there was no error
        if os.path.isfile(utils.logfile):
            os.unlink(utils.logfile)
    except Exception:
        traceback.print_exception(*sys.exc_info(), file=sys.stderr)
        with open(utils.logfile, 'w') as f:
            traceback.print_exception(*sys.exc_info(), file=f)

        # Delete the target file if there was an error
        if os.path.isfile(args.target):
            os.unlink(args.target)


if __name__ == "__main__":
    main()

