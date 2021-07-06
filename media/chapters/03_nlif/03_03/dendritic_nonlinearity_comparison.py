#!/usr/bin/env python3

from nef_synaptic_computation.multi_compartment_lif import *
from nef_synaptic_computation import lif_utils

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as font_manager


# Helper-function used to plot a contour plot with colour bar
def plot_contour(fig,
                 ax,
                 cax,
                 xs,
                 ys,
                 zs,
                 levels,
                 xlabel="",
                 ylabel="",
                 zlabel="",
                 subth=1.0):
    utils.outside_ticks(ax)

    mpl.rcParams['hatch.linewidth'] = 0.5
    hatches = [('///' if l < subth else None) for l in levels[:-1]]

    contour = ax.contourf(xs,
                          ys,
                          zs,
                          hatches=hatches,
                          levels=levels,
                          cmap='viridis')
    for i, collection in enumerate(contour.collections):
        collection.set_edgecolor((1.0, 1.0, 1.0, 0.5))
        collection.set_linewidth(0.)
    ax.contour(xs,
               ys,
               zs,
               levels=contour.levels,
               colors=['white'],
               linestyles=[(1, (3, 2))])
    ax.set_xlim(np.min(xs), np.max(xs))
    ax.set_ylim(np.min(ys), np.max(ys))
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))

    if not cax is None:
        utils.outside_ticks(cax)
        cb = fig.colorbar(contour, ax=ax, cax=cax, orientation='horizontal')
        for i, patch in enumerate(cb.solids_patches):
            patch.set_edgecolor((1.0, 1.0, 1.0, 0.5))
            patch.set_linewidth(0.)
        cb.outline.set_visible(False)
        cax.xaxis.tick_top()
        cax.set_xlabel(zlabel, labelpad=0.0)
        cax.xaxis.labelpad = -37.5


def plot_analysis(ax, cax, model, xs, ys, xlabel="", ylabel="", letter=None, lx=-0.375, title=None):
    xss, yss = np.meshgrid(xs, ys)
    smpls = np.array((xss.flatten(), yss.flatten()))
    #zss = model.iSom_empirical(smpls)
    zss = model.iSom(smpls)
    zss_isom = zss.reshape(xss.shape)
    iTh = np.round(model.iTh() * 1e9, 2)

    levels = np.linspace(-0.5, 2.0, 11)

    plot_contour(fig,
                 ax,
                 cax,
                 xs * 1e9,
                 ys * 1e9,
                 zss_isom * 1e9,
                 levels,
                 xlabel,
                 ylabel,
                 'Somatic current ($\mathrm{nA}$)',
                 subth=iTh)
    ax.set_xlim(np.min(xs) * 1e9, np.max(xs) * 1e9)
    ax.set_ylim(np.min(ys) * 1e9, np.max(ys) * 1e9)

    if not letter is None:
        ax.text(lx, 1.175, "\\textbf{{{}}}".format(letter), va="top", ha="left", size=12, transform=ax.transAxes)

    if not title is None:
        ax.set_title(title)

    return fig, ax


fig = plt.figure(figsize=(7.35, 1.45))
gs1 = fig.add_gridspec(1, 4, wspace=0.5)
gs2 = fig.add_gridspec(1, 1, wspace=0.5, top=1.175, bottom=1.1, left=0.3, right=0.7)
axs = [fig.add_subplot(gs1[0, i]) for i in range(4)]
cax = fig.add_subplot(gs2[0, 0])

nsamples = 40

CurLIF = Neuron()
CurLIF.add_compartment(
    Compartment(soma=True).add_channel(CurChan(
        mul=1.0, name="exc")).add_channel(CurChan(mul=-1.0,
                                                  name="inh")).add_channel(
                                                      CondChan(Erev=-65e-3,
                                                               g=50e-9,
                                                               name="leak")))
CurLIF = CurLIF.assemble()

CondLIF = Neuron()
CondLIF.add_compartment(
    Compartment(soma=True).add_channel(
        CondChan(Erev=20e-3,
                 name="exc")).add_channel(CondChan(Erev=-75e-3,
                                                   name="inh")).add_channel(
                                                       CondChan(Erev=-65e-3,
                                                                g=50e-9,
                                                                name="leak")))
CondLIF = CondLIF.assemble()

TwoCompCondIF = Neuron()
TwoCompCondIF.add_compartment(
    Compartment(soma=True).add_channel(
        CondChan(Erev=-65e-3, g=53.333333334e-9, name="leak")))
TwoCompCondIF.add_compartment(
    Compartment(name="dendrites").add_channel(
        CondChan(Erev=20e-3,
                 name="exc")).add_channel(CondChan(Erev=-75e-3,
                                                   name="inh")).add_channel(
                                                       CondChan(Erev=-65e-3,
                                                                g=50e-9,
                                                                name="leak")))
TwoCompCondIF.connect("soma", "dendrites", 30e-9)
TwoCompCondIF = TwoCompCondIF.assemble()

ThreeCompCondIF = Neuron()
ThreeCompCondIF.add_compartment(
    Compartment(soma=True).add_channel(
        CondChan(Erev=-65e-3, g=53.333333334e-9, name="leak")))
ThreeCompCondIF.add_compartment(
    Compartment(name="distal")
        .add_channel(CondChan(Erev=20e-3, name="exc1"))
        .add_channel(CondChan(Erev=-75e-3, g=100e-9, name="inh1"))
        .add_channel(CondChan(Erev=-65e-3, g=50e-9, name="leak")))
ThreeCompCondIF.add_compartment(
    Compartment(name="proximal")
        .add_channel(CondChan(Erev=20e-3, g=95e-9, name="exc2"))
        .add_channel(CondChan(Erev=-75e-3, name="inh2"))
        .add_channel(CondChan(Erev=-65e-3, g=50e-9, name="leak")))
ThreeCompCondIF.connect("soma", "proximal", 40e-9)
ThreeCompCondIF.connect("proximal", "distal", 100e-9)
ThreeCompCondIF = ThreeCompCondIF.assemble()


plot_analysis(axs[0],
              cax,
              CurLIF,
              np.linspace(0.75e-9, 2.0e-9, nsamples),
              np.linspace(0, 1.25e-9, nsamples),
              xlabel="$J_\mathrm{E}$ ($\mathrm{nA}$)",
              ylabel="$J_\mathrm{I}$ ($\mathrm{nA}$)",
              letter="A", lx=-0.4, title="Current-based LIF")

plot_analysis(axs[1],
              None,
              CondLIF,
              np.linspace(9.75e-9, 25.75e-9, nsamples),
              np.linspace(0, 71e-9, nsamples),
              xlabel="$g_\mathrm{E}$ ($\mathrm{nS}$)",
              ylabel="$g_\mathrm{I}$ ($\mathrm{nS}$)",
              letter="B", title="Cond.-based LIF")

plot_analysis(axs[2],
              None,
              TwoCompCondIF,
              np.linspace(60e-9, 400e-9, nsamples),
              np.linspace(0, 290e-9, nsamples),
              xlabel="$g_\mathrm{E}$ ($\mathrm{nS}$)",
              ylabel="$g_\mathrm{I}$ ($\mathrm{nS}$)",
              letter="C", lx=-0.425, title="Two-comp. LIF")

plot_analysis(axs[3],
              None,
              ThreeCompCondIF,
              np.linspace(0e-9, 700e-9, nsamples),
              np.linspace(0, 132.5e-9, nsamples),
              xlabel="$g_\mathrm{E}^1$ ($\mathrm{nS}$)",
              ylabel="$g_\mathrm{I}^2$ ($\mathrm{nS}$)",
              letter="D", lx=-0.425, title="Three-comp. LIF")

utils.save(fig)

