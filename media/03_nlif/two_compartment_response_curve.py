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
    utils.outside_ticks(cax)

    mpl.rcParams['hatch.linewidth'] = 0.1
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

    if not cax is None:
        cb = fig.colorbar(contour, ax=ax, cax=cax, orientation='horizontal')
        for i, patch in enumerate(cb.solids_patches):
            patch.set_edgecolor((1.0, 1.0, 1.0, 0.5))
            patch.set_linewidth(0.)
        cb.outline.set_visible(False)
        cax.xaxis.tick_top()
        cax.set_xlabel(zlabel)
        cax.xaxis.labelpad = -37.5


def plot_analysis(model, xs, ys, xlabel="", ylabel=""):
    fig = plt.figure(figsize=(7.35, 2.25))
    gs1 = fig.add_gridspec(2, 3, height_ratios=[1, 20], wspace=0.3)
    gs2 = fig.add_gridspec(2,
                           1,
                           left=0.46,
                           right=0.56,
                           top=0.95,
                           bottom=0.1125,
                           hspace=0.8)

    xss, yss = np.meshgrid(xs, ys)
    smpls = np.array((xss.flatten(), yss.flatten()))
    zss = model.iSom_empirical(smpls)
    zss_rate = zss['rate'].reshape(xss.shape)
    zss_isom = zss['isom'].reshape(xss.shape)

    Js = np.linspace(0.5e-9, np.max(zss_isom))
    rates = model.estimate_lif_rate_from_current(Js)
    iTh = np.round(model.iTh() * 1e9, 2)

    ax, cax = fig.add_subplot(gs1[1, 0]), fig.add_subplot(gs1[0, 0])
    plot_contour(fig, ax, cax, xs * 1e9, ys * 1e9, zss_rate,
                 [0, 10, 20, 30, 40, 50, 60, 70, 80], xlabel, ylabel,
                 'Firing rate ($\mathrm{s}^{-1}$)')
    ax.set_xticks([100, 200, 300, 400, 500])
    ax.set_xticks(np.arange(50, 501, 50), minor=True)
    ax.set_yticks([0, 100, 200, 300])
    ax.set_yticks(np.arange(0, 401, 50), minor=True)
    ax.set_xlim(65, 500)
    ax.set_ylim(0, 365)
    ellipse = mpl.patches.Ellipse([0.075, 0.925],
                                  0.095 * 1.1,
                                  0.1 * 1.1,
                                  transform=ax.transAxes,
                                  color='white',
                                  linewidth=0.0)
    ax.add_artist(ellipse)
    ax.text(0.075 - 0.005,
            0.915,
            '$\\mathscr{G}$',
            va='center',
            ha='center',
            transform=ax.transAxes)
    ax.text(-0.27,
            1.3625,
            '\\textbf{A}',
            va='bottom',
            ha='left',
            transform=ax.transAxes,
            size=12)

    iTh = model.iTh() * 1e9

    ax, cax = fig.add_subplot(gs1[1, 2]), fig.add_subplot(gs1[0, 2])
    plot_contour(fig,
                 ax,
                 cax,
                 xs * 1e9,
                 ys * 1e9,
                 zss_isom * 1e9,
                 [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
                 xlabel,
                 ylabel,
                 'Somatic current ($\mathrm{nA}$)',
                 subth=iTh)
    ax.set_xticks([100, 200, 300, 400, 500])
    ax.set_xticks(np.arange(50, 501, 50), minor=True)
    ax.set_yticks([0, 100, 200, 300])
    ax.set_yticks(np.arange(0, 401, 50), minor=True)
    ax.set_xlim(65, 500)
    ax.set_ylim(0, 365)
    ellipse = mpl.patches.Ellipse([0.075, 0.925],
                                  0.095 * 1.1,
                                  0.1 * 1.1,
                                  transform=ax.transAxes,
                                  color='white',
                                  linewidth=0.0)
    ax.add_artist(ellipse)
    ax.text(0.075 - 0.005,
            0.92,
            '$H$',
            va='center',
            ha='center',
            transform=ax.transAxes)
    ax.text(-0.27,
            1.3625,
            '\\textbf{D}',
            va='bottom',
            ha='left',
            transform=ax.transAxes,
            size=12)

    ax = fig.add_subplot(gs2[0, 0])
    ax.arrow(-1.05 + 0.35,
             0.45,
             -0.35,
             -0.3,
             transform=ax.transAxes,
             clip_on=False,
             linewidth=0.5,
             head_width=0.03,
             color='k')
    ax.arrow(1.15 + 0.35,
             0.15,
             -0.35,
             0.3,
             transform=ax.transAxes,
             clip_on=False,
             linewidth=0.5,
             head_width=0.03,
             color='k')
    ax.plot(Js * 1e9, rates, color='k', linewidth=1, clip_on=False)

    mpl.rcParams['hatch.linewidth'] = 0.7
    ax.fill_betweenx([0, 100], [0.5, 0.5], [iTh, iTh],
                     color='none',
                     hatch='///',
                     edgecolor='k',
                     linewidth=0.0)

    ax.set_xlabel('Current ($\mathrm{nA}$)')
    ax.set_ylabel('Rate ($\mathrm{s}^{-1}$)')
    ax.set_xticks([0.5, 1, 1.5, 2])
    ax.set_xlim(0.5, 2)
    ax.set_ylim(0, 100)
    utils.outside_ticks(ax)
    ellipse = mpl.patches.Ellipse([0.85, 0.175],
                                  0.225,
                                  0.225,
                                  transform=ax.transAxes,
                                  facecolor='white',
                                  linewidth=0.5,
                                  edgecolor='k')
    ax.add_artist(ellipse)
    ax.text(0.85 - 0.01,
            0.1375 + 0.025,
            '$G$',
            va='center',
            ha='center',
            transform=ax.transAxes)
    ax.text(-0.6125,
            1.23,
            '\\textbf{B}',
            va='bottom',
            ha='left',
            transform=ax.transAxes,
            size=12)

    ax = fig.add_subplot(gs2[1, 0])
    ax.arrow(-1.05,
             0.75,
             0.35,
             -0.3,
             transform=ax.transAxes,
             clip_on=False,
             linewidth=0.5,
             head_width=0.03,
             color='k')
    ax.arrow(1.15,
             0.45,
             0.35,
             0.3,
             transform=ax.transAxes,
             clip_on=False,
             linewidth=0.5,
             head_width=0.03,
             color='k')
    ax.plot(rates, Js * 1e9, color='k', linewidth=1, clip_on=False)

    mpl.rcParams['hatch.linewidth'] = 0.7
    ax.fill_between([0, 100], [0.5, 0.5], [iTh, iTh],
                    color='none',
                    hatch='///',
                    edgecolor='k',
                    linewidth=0.0)

    ax.set_xlabel('Rate ($\mathrm{s}^{-1}$)')
    ax.set_ylabel('Current ($\mathrm{nA}$)', labelpad=6.45)
    ax.set_yticks([0.5, 1, 1.5, 2])
    ax.set_xlim(0, 100)
    ax.set_ylim(0.5, 2)
    utils.outside_ticks(ax)

    ellipse = mpl.patches.FancyBboxPatch([0.1, 0.75],
                                         0.4,
                                         0.2,
                                         boxstyle=mpl.patches.BoxStyle(
                                             'round',
                                             pad=0.02,
                                             rounding_size=0.125),
                                         transform=ax.transAxes,
                                         facecolor='white',
                                         linewidth=0.5,
                                         edgecolor='k')
    ax.add_artist(ellipse)
    ax.text(0.3,
            0.825,
            '$G^{-1}$',
            va='center',
            ha='center',
            transform=ax.transAxes)
    ax.text(-0.6125,
            1.1,
            '\\textbf{C}',
            va='bottom',
            ha='left',
            transform=ax.transAxes,
            size=12)

    return fig, ax


TwoCompCondIF = (Neuron().add_compartment(
    Compartment(soma=True).add_channel(
        CondChan(Erev=-65e-3, g=53.333333334e-9,
                 name="leak"))).add_compartment(
                     Compartment(name="dendrites").add_channel(
                         CondChan(Erev=20e-3, name="exc")).add_channel(
                             CondChan(Erev=-75e-3, name="inh")).add_channel(
                                 CondChan(Erev=-65e-3, g=50e-9,
                                          name="leak"))).connect(
                                              "soma", "dendrites",
                                              30e-9).assemble())

nsamples = 20
fig, _ = plot_analysis(
    TwoCompCondIF,
    np.linspace(65e-9, 500e-9, nsamples),
    np.linspace(0, 365e-9, nsamples),
    xlabel="Excitatory cond. $g_\mathrm{E}$ ($\mathrm{nS}$)",
    ylabel="Inhibitory cond. $g_\mathrm{I}$ ($\mathrm{nS}$)")

utils.save(fig)

