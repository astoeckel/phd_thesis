import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

FIG_WIDTH = 7.2


def mkfg_save(fig, filename):
    fig.savefig(
        filename,
        bbox_inches='tight',
        pad_inches=0.0,
        transparent=True,
    )


def mkfg_main(cback):
    import sys, os
    if len(sys.argv) != 2:
        print("Usage: {} [TARGET FILENAME]".format(sys.argv[0]))
        sys.exit(1)
    mpl.rc_file(os.path.join(os.path.dirname(__file__), 'matplotlibrc'))
    cback(sys.argv[1])

