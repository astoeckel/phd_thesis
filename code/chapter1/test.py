#!/usr/bin/env python3

# Import common figure drawing stuff
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common import *

def generate_figure(filename):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3))
    ts = np.linspace(-1, 1, 100)
    ax.plot(ts, ts ** 2)
    ax.text(0, 0.5, "Dies ist nur ein Test $x = 2x^2$", va='center', ha='center', clip_on=False)
    ax.set_xlabel("Time $t$ (s)")
    ax.set_ylabel("Function value $f(t)$")
    ax.set_title("A truely quadratic function")
    mkfg_save(fig, filename)

if __name__ == "__main__":
    mkfg_main(generate_figure)
