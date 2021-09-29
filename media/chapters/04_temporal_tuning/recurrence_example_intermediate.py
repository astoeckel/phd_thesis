import numpy as np
import nengo
import matplotlib.pyplot as plt

import sys

sys.path.append('../../lib')
from temporal_encoder_common import *

fig = plt.figure(figsize=(7.9, 2.0))


def run_and_plot(axs, tauIn, tauRec):
    res = solve_for_linear_dynamics(
        [Filters.lowpass_chained(t) for t in tauIn] +
        [Filters.lowpass_chained(t1, t2) for t1 in tauRec for t2 in tauIn] +
        [Filters.lowpass_chained(t1, t2) for t1 in tauRec for t2 in tauRec],
        [Filters.dirac()] * len(tauIn) + [Filters.dirac()] *
        (len(tauRec) * len(tauIn)) + [Filters.step()] *
        (len(tauRec) * len(tauRec)),
        [Filters.step()],
        T=10.0,
        N_smpls=100,
        sigma=None,
    )

    i0 = 0
    i1 = len(tauIn)
    i2 = i1 + len(tauRec) * len(tauIn)
    i3 = i2 + len(tauRec) * len(tauRec)

    M1 = res[i0:i1]
    M2 = res[i1:i2].reshape(len(tauRec), len(tauIn))
    M3 = res[i2:i3].reshape(len(tauRec), len(tauRec))

    def LP(*args):
        return nengo.LinearFilter(*Filters.lowpass_laplace_chained(*args),
                                  analog=True)

    T, dt = 10.0, 1e-3
    high = 5.0

    colors = [utils.blues[0], utils.oranges[1]]

    np.random.seed(39191)

    for idx_in, input_type in enumerate(["step", "noise"]):
        with nengo.Network() as model:
            if input_type == "step":
                u = nengo.Node(lambda t: 1.0 * (t >= 0.5) * (t < 1.5))
            else:
                u = nengo.Node(nengo.processes.WhiteSignal(period=T,
                                                           high=high))

            y = nengo.Ensemble(n_neurons=1,
                               dimensions=1,
                               neuron_type=nengo.Direct())

            for i, tau in enumerate(tauIn):
                nengo.Connection(u, y, synapse=tau, transform=M1[i])
            for i, tau1 in enumerate(tauRec):
                for j, tau2 in enumerate(tauIn):
                    nengo.Connection(u,
                                     y,
                                     synapse=LP(tau1, tau2),
                                     transform=M2[i, j])
            for i, tau1 in enumerate(tauRec):
                for j, tau2 in enumerate(tauRec):
                    nengo.Connection(y,
                                     y,
                                     synapse=LP(tau1, tau2),
                                     transform=M3[i, j])

            p_u = nengo.Probe(u, synapse=None)
            p_y = nengo.Probe(y, synapse=None)

        with nengo.Simulator(model, dt=dt) as sim:
            sim.run(T)

        ts = sim.trange()
        ys = sim.data[p_y]
        xs = sim.data[p_u]
        ys_tar = (np.cumsum(sim.data[p_u][:, 0]) * dt).reshape(-1, 1)

        for i in range(ys.shape[1]):
            axs[0, idx_in].plot(ts, ys[:, i], color=colors[i], lw=1.5)
            axs[0, idx_in].plot(ts,
                                ys_tar[:, i],
                                color=colors[i],
                                linestyle=(1, (1, 2)),
                                lw=0.7,
                                zorder=10)
            axs[0, idx_in].plot(ts,
                                ys_tar[:, i],
                                'white',
                                linestyle=(0, (1, 2)),
                                lw=0.7,
                                zorder=10)

        axs[1, idx_in].plot(ts, xs, lw=1.0, color='k', clip_on=False)

        # Setup the axes
        axs[0, idx_in].set_xticks([])
        axs[0, idx_in].set_yticks([])
        axs[0, idx_in].spines["bottom"].set_visible(False)
        for i in range(2):
            axs[i, idx_in].set_xlim(0, T)
            axs[i, idx_in].set_yticks([])
            axs[i, idx_in].spines["left"].set_visible(False)
        axs[1, idx_in].set_xticks(np.linspace(0, T, 3))
        axs[1, idx_in].set_xticks(np.linspace(0, T, 5), minor=True)
        if idx_in == 0:
            axs[1, idx_in].set_ylim(0, 1)

        rmse = np.sqrt(np.mean(np.square(ys - ys_tar)))
        rms = np.sqrt(np.mean(np.square(ys_tar)))
        axs[0, idx_in].text(1.0,
                            0.6 + (idx_in - 1.0) * 0.15,
                            "$E = {:0.1f}\\%$".format(
                                (rmse / rms) * 100).format(),
                            ha="right",
                            va="baseline",
                            transform=axs[0, idx_in].transAxes)


#
# Panel A: Heterogeneous with a single filter
#

gs = fig.add_gridspec(2,
                      2,
                      wspace=0.1,
                      height_ratios=[3, 1],
                      top=0.9,
                      bottom=0.6)
axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(2)]
                for i in range(2)])

axs[0, 0].set_title("\\textit{Unit pulse}", y=1.5)
axs[1, 0].set_xticklabels([])

axs[0, 1].set_title("\\textit{Band-limited noise ($B = 5\\,\\mathrm{Hz}$)}",
                    y=1.5)
axs[1, 1].set_xticklabels([])

tauIn = [0.01]
tauRec = [0.1]
run_and_plot(axs, tauIn, tauRec)

fig.text(0.0,
         1.25,
         "\\textbf{A}",
         size=12,
         ha="left",
         va="baseline",
         transform=axs[0, 0].transAxes)
fig.text(1.05,
         1.25,
         "\\textbf{Single filter per path}",
         ha="center",
         va="baseline",
         transform=axs[0, 0].transAxes)

#
# Panel B: Heterogeneous with three filters
#

gs = fig.add_gridspec(2,
                      2,
                      wspace=0.1,
                      height_ratios=[3, 1],
                      top=0.4,
                      bottom=0.1)
axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(2)]
                for i in range(2)])

tauIn = [0.01, 0.02, 0.03]
tauRec = [0.1, 0.11, 0.12]
run_and_plot(axs, tauIn, tauRec)

fig.text(0.0,
         1.25,
         "\\textbf{B}",
         size=12,
         ha="left",
         va="baseline",
         transform=axs[0, 0].transAxes)
fig.text(1.05,
         1.25,
         "\\textbf{Three filters per path}",
         ha="center",
         va="baseline",
         transform=axs[0, 0].transAxes)

axs[1, 0].set_xlabel("Time $t$ (s)")
axs[1, 1].set_xlabel("Time $t$ (s)")

utils.save(fig)

