#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from nef_synaptic_computation.multi_compartment_lif import *
from nef_synaptic_computation import lif_utils


# Helper-function used to plot a contour plot with colour bar
def plot_contour(ax, cax, xs, ys, zs, zs2, levels, ylabel="", zlabel=""):
    contour = ax.contourf(xs, ys, zs, levels)
    ax.set_xlim(np.min(xs), np.max(xs))
    ax.set_ylim(np.min(ys), np.max(ys))
    ax.set_ylabel(ylabel)

    ax.contour(xs,
               ys,
               zs2,
               levels=levels,
               colors=['w'],
               linewidths=[1.0],
               linestyles=['--'])

    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))

    if not cax is None:
        cb = ax.get_figure().colorbar(contour,
                                      ax=ax,
                                      cax=cax,
                                      orientation='vertical')
        cb.outline.set_visible(False)
        cax.set_ylabel(zlabel)
        utils.outside_ticks(cax)


def plot_cross_section(ax,
                       axcross,
                       y0,
                       xs,
                       ys,
                       zs,
                       zs2,
                       xlabel="",
                       levels=None):
    ys_median = ys[len(ys) // 2]
    ax.plot(xs, np.ones_like(xs) * ys_median, 'k', linewidth=1)
    ax.plot(xs,
            np.ones_like(xs) * ys_median,
            'w',
            linewidth=1,
            linestyle=(0, (1, 1)))
    axcross.plot(xs, zs[len(ys) // 2, :], linewidth=1, color='k')
    axcross.plot(xs, zs2[len(ys) // 2, :], linewidth=1, color='k')
    axcross.plot(xs,
                 zs2[len(ys) // 2, :],
                 linewidth=1,
                 color='w',
                 linestyle=(1, (1, 2)))
    axcross.set_xlim(np.min(xs), np.max(xs))
    axcross.set_ylim(y0, None if levels is None else levels[-1])
    axcross.set_xlabel(xlabel, labelpad=2.5)
    axcross.set_ylabel('Rate ($\\mathrm{s}^{-1}$)')

    axcross.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    axcross.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))


def write_error(ax, Zmodel, Ztruth, mul, unit, clip=False):
    if clip:
        valid = np.logical_or(Ztruth > 12.5, Zmodel > 12.5)
        E = np.sqrt(np.mean((Zmodel[valid] - Ztruth[valid])**2))
    else:
        E = np.sqrt(np.mean((Zmodel - Ztruth)**2))
    if mul is None:
        scale = np.log10(E + 1e-12)
        if scale <= -9:
            mul = 1e12
            unit = 'p' + unit
        elif scale <= -6:
            mul = 1e9
            unit = 'n' + unit
        else:
            mul = 1e6
            unit = '\\mu ' + unit
    ax.text(0.03, 0.9, "$E = {:.1f}\,\\mathrm{{{}}}$".format(E * mul, unit), transform=ax.transAxes, va="baseline", ha="left", bbox={\
        'pad': 1.5,
        'facecolor': 'w',
        'linewidth': 0
    }, size=8)

def write_gc(ax, gc):
    if gc is None:
        return

    idx = {
        50: 0,
        100: 1,
        200: 2
    }[gc]

    ax.text(0.97, 0.9, "${:d}\,\\mathrm{{{}}}$".format(int(gc), "nS"), transform=ax.transAxes, va="baseline", ha="right", bbox={\
        'pad': 1.5,
        'facecolor': 'w',
        'linewidth': 0
    }, size=8, zorder=100)

    ox = 0.02 if idx == 0 else 0.0

    colors =  [mpl.cm.get_cmap('viridis')(0.3), mpl.cm.get_cmap('viridis')(0.6), mpl.cm.get_cmap('viridis')(0.9)]
    circle = mpl.patches.Ellipse((0.69 + ox, 0.93), 0.06, 0.065, transform=ax.transAxes, clip_on=False, zorder=101, color=colors[idx])
    ax.add_artist(circle)

    rect = mpl.patches.Rectangle((0.63 + ox, 0.858), 0.14 - ox, 0.13, transform=ax.transAxes, clip_on=False, zorder=99, color='white', linewidth=0.0)
    ax.add_artist(rect)

def plot_analysis(
        model,
        fig,
        gs,
        col,
        xs,
        ys,
        xlabel="",
        ylabel="",
        zlabel="",
        gc=None,
        cax=None,
        include_cbars=None,
        do_fit=True,  # Set to "false" when tinkering with the layout
        do_use_noise=False,  # Set to "true" to perform the experiment with spike noise #'data/network_noise_profile.h5'
        do_relu=False,  # Set to "true" to assume a ReLU instead of a LIF response curve
        do_lif=False,  # Set to "true" to fit the LIF response curve
):
    import sympy as sp

    if isinstance(do_use_noise, str):
        do_use_noise = utils.datafile(do_use_noise)

    xss, yss = np.meshgrid(xs, ys)
    if model.n_inputs == 2:
        din = (xss, yss)
    else:
        zos = np.zeros_like(xss)
        din = (zos, yss, xss, zos)

    T = 1.0 if not do_use_noise else 100.0
    tau = (5e-3, 10e-3)  # Time-constants for the individual synapses
    rate = (1800, 4500)  # Firing rates
    n_samples_per_dim = 5000
    clip = True

    # Print the original parameters
    def print_rational_fun(expr, α=None, β=None):
        expr = sp.cancel(sp.together(expr))
        gE, gI = model.input_syms()
        num, den = sp.fraction(expr)
        p_num, p_den = sp.Poly(num, gE, gI), sp.Poly(den, gE, gI)
        b1, b2, b0 = p_num.coeffs()
        a1, a2, a0 = p_den.coeffs()

        # Select a scaling factor for a0
        n_b0 = float(1e9 * b0 / b1)
        n_b1 = 1000.0
        n_b2 = float(1000.0 * b2 / b1)

        n_a0 = float(a0 / b1)
        n_a1 = float(1e-6 * a1 / b1)
        n_a2 = float(1e-6 * a2 / b1)

        #        print("\tb0 = {:.1f} µS".format((n_b0)))
        #        print("\tb1 = {:.1f}".format((n_b1)))
        #        print("\tb2 = {:.1f}".format((n_b2)))

        #        print("\ta0 = {:.1f} 1/mV".format((n_a0)))
        #        print("\ta1 = {:.1f} 1/nA".format((n_a1)))
        #        print("\ta2 = {:.1f} 1/nA".format((n_a2)))

        #        print("J_max = {:.1f} nA".format(float(1e9 * b1 / a1)))
        #        print("J_min = {:.1f} nA".format(float(1e9 * b2 / a2)))

        print('''{{
    "b0": {:0.1f}e-6,
    "b1": {:0.1f},
    "b2": {:0.1f},
    "a0": {:0.1f}e3,
    "a1": {:0.1f}e9,
    "a2": {:0.1f}e9,'''.format(n_b0, n_b1, n_b2, n_a0, n_a1, n_a2))
        if not (α is None or β is None):
            print('''    "α": {:0.1f}e9,
    "β": {:0.1f},'''.format(α * 1e-9, β))
        print("}")

    soma = model.soma()
    vSom = 0.5 * (soma.v_reset + soma.v_th)
    print('Nonlinearity without optimization')
    print_rational_fun(model.synaptic_nonlinearity(vSom=vSom))

    # Fit weights to each model
    if do_fit:
        xMin = np.zeros(2)
        xMax = np.ones(2) * np.max((xs, ys))
        if do_relu:
            ws, err_0, err_opt, relu_slope, relu_bias = utils.run_with_cache(
                model.fit_model_weights,
                xMin,
                xMax,
                n_samples_per_dim=n_samples_per_dim,
                fit_relu=True,
                dt=1e-4,
                T=T,
                noise=do_use_noise,
                tau=tau,
                rate=rate,
                random=np.random.RandomState(48912))
            print("ReLU parameters: slope={:.3g}, bias={:.3g}".format(
                relu_slope, relu_bias))
        elif do_lif:
            ws, err_0, err_opt, lif_w, G, _ = utils.run_with_cache(
                model.fit_model_weights,
                xMin,
                xMax,
                n_samples_per_dim=n_samples_per_dim,
                fit_lif=True,
                dt=1e-4,
                T=T,
                noise=do_use_noise,
                tau=tau,
                rate=rate,
                random=np.random.RandomState(48912))
            print("LIF parameters: w0={:.3g}, w1={:.3g}, w2={:.3g}".format(
                *lif_w))
        else:
            ws, err_0, err_opt = utils.run_with_cache(
                model.fit_model_weights,
                xMin,
                xMax,
                n_samples_per_dim=n_samples_per_dim,
                dt=1e-4,
                T=T,
                noise=do_use_noise,
                tau=tau,
                rate=rate,
                random=np.random.RandomState(48912))
        print('RMSE without fitted weights: ', err_0)
        print('RMSE with fitted weights:', err_opt)

        # Print the optimized parameters
        print('Nonlinearity with optimization')
        nlin2 = model.synaptic_nonlinearity(ws=ws, vSom=vSom)
        if do_relu:
            print_rational_fun(nlin2, relu_slope, relu_bias)
        else:
            print_rational_fun(nlin2)

    soma = model.soma()
    data_model = model.iSom(din)
    if do_relu:
        data_model_rate = relu_bias + relu_slope * data_model
    elif do_lif:
        data_model_rate = G(data_model)
    else:
        data_model_rate = utils.run_with_cache(
            model.estimate_lif_rate_from_current, data_model)

    if do_fit:
        data_model_fit = utils.run_with_cache(model.iSom, din, ws=ws)
        if do_relu:
            data_model_fit_rate = relu_bias + relu_slope * data_model_fit
        elif do_lif:
            data_model_fit_rate = G(data_model_fit)
        else:
            data_model_fit_rate = utils.run_with_cache(
                model.estimate_lif_rate_from_current, data_model_fit)

    data_empirical = utils.run_with_cache(model.iSom_empirical,
                                          din,
                                          dt=1e-4,
                                          T=T,
                                          noise=do_use_noise,
                                          tau=tau,
                                          rate=rate)

    iTh = model.iTh() * 1e9

    levels_rate = np.linspace(0, 100, 11)

    ax3 = fig.add_subplot(gs[0, col])
    ax3cross = fig.add_subplot(gs[1, col])
    utils.outside_ticks(ax3)
    utils.outside_ticks(ax3cross)
    plot_contour(ax3,
                 cax,
                 xs * 1e9,
                 ys * 1e9,
                 data_empirical['rate'],
                 data_model_rate,
                 levels_rate,
                 ylabel=ylabel,
                 zlabel="Average spike rate ($\mathrm{s}^{-1}$)")
    ax3.set_xticklabels([])
    plot_cross_section(ax3,
                       ax3cross,
                       0.0,
                       xs * 1e9,
                       ys * 1e9,
                       data_empirical['rate'],
                       data_model_rate,
                       xlabel=xlabel,
                       levels=levels_rate)
    write_error(ax3,
                data_model_rate,
                data_empirical['rate'],
                1,
                's^{-1}',
                clip=clip)
    write_gc(ax3, gc)

    ax4 = None
    ax4cross = None
    if do_fit:
        ax4 = fig.add_subplot(gs[3, col], sharey=ax3)
        ax4cross = fig.add_subplot(gs[4, col])
        utils.outside_ticks(ax4)
        utils.outside_ticks(ax4cross)
        plot_contour(ax4,
                     None,
                     xs * 1e9,
                     ys * 1e9,
                     data_empirical['rate'],
                     data_model_fit_rate,
                     levels_rate,
                     ylabel=ylabel,
                     zlabel="Average spike rate ($\mathrm{s}^{-1}$)")
        plot_cross_section(ax4,
                           ax4cross,
                           0.0,
                           xs * 1e9,
                           ys * 1e9,
                           data_empirical['rate'],
                           data_model_fit_rate,
                           xlabel=xlabel,
                           levels=levels_rate)
        write_error(ax4,
                    data_model_fit_rate,
                    data_empirical['rate'],
                    1,
                    's^{-1}',
                    clip=clip)
        write_gc(ax4, gc)
        ax4.set_xticklabels([])

    return list(filter(bool, [ax3, ax3cross, ax4, ax4cross]))


# Two-compartment, conductance-based LIF
TwoCompCondIF_50 = (Neuron().add_compartment(
    Compartment(v_th=-50e-3, v_reset=-65e-3, v_spike=20e-3, soma=True).add_channel(
        CondChan(Erev=-65e-3, g=50e-9, name="leak"))).add_compartment(
            Compartment(name="dendrites").add_channel(
                CondChan(Erev=20e-3, name="exc")).add_channel(
                    CondChan(Erev=-75e-3, name="inh")).add_channel(
                        CondChan(Erev=-65e-3, g=50e-9, name="leak"))).connect(
                            "soma", "dendrites", 0.05e-6).assemble())
TwoCompCondIF_100 = (Neuron().add_compartment(
    Compartment(soma=True).add_channel(
        CondChan(Erev=-65e-3, g=50e-9, name="leak"))).add_compartment(
            Compartment(name="dendrites").add_channel(
                CondChan(Erev=20e-3, name="exc")).add_channel(
                    CondChan(Erev=-75e-3, name="inh")).add_channel(
                        CondChan(Erev=-65e-3, g=50e-9, name="leak"))).connect(
                            "soma", "dendrites", 0.1e-6).assemble())
TwoCompCondIF_200 = (Neuron().add_compartment(
    Compartment(soma=True).add_channel(
        CondChan(Erev=-65e-3, g=50e-9, name="leak"))).add_compartment(
            Compartment(name="dendrites").add_channel(
                CondChan(Erev=20e-3, name="exc")).add_channel(
                    CondChan(Erev=-75e-3, name="inh")).add_channel(
                        CondChan(Erev=-65e-3, g=50e-9, name="leak"))).connect(
                            "soma", "dendrites", 0.2e-6).assemble())


def do_plot(suffix, **kwargs):
    fig = plt.figure(figsize=(6.5, 4.8))
    gs = mpl.gridspec.GridSpec(5,
                                3,
                                height_ratios=[5, 1, 1.75, 5, 1],
                                width_ratios=[6, 6, 6],
                                wspace=0.5,
                                hspace=0.2)
    cgs = fig.add_gridspec(1, 1, left=0.925, right=0.94, top=0.883, bottom=0.11)
    cax = fig.add_subplot(cgs[0, 0])

    nsamples = 100

    axs = []
    axs.append(
        plot_analysis(TwoCompCondIF_50,
                      fig,
                      gs,
                      0,
                      np.linspace(25e-9, 200e-9, nsamples),
                      np.linspace(0, 225e-9, nsamples),
                      xlabel="$g_\\mathrm{E}$ ($\\mathrm{nS}$)",
                      ylabel="$g_\\mathrm{I}$ ($\\mathrm{nS}$)",
                      gc=50,
                      cax=cax,
                      **kwargs))

    axs.append(
        plot_analysis(TwoCompCondIF_100,
                      fig,
                      gs,
                      1,
                      np.linspace(25e-9, 75e-9, nsamples),
                      np.linspace(0, 90e-9, nsamples),
                      xlabel="$g_\\mathrm{E}$ ($\\mathrm{nS}$)",
                      ylabel="$g_\\mathrm{I}$ ($\\mathrm{nS}$)",
                      gc=100,
                      **kwargs))

    axs.append(
        plot_analysis(TwoCompCondIF_200,
                      fig,
                      gs,
                      2,
                      np.linspace(25e-9, 50e-9, nsamples),
                      np.linspace(0, 60e-9, nsamples),
                      xlabel="$g_\\mathrm{E}$ ($\\mathrm{nS}$)",
                      ylabel="$g_\\mathrm{I}$ ($\\mathrm{nS}$)",
                      gc=200,
                      **kwargs))

    axs = np.array(axs).T
    axs[0, 1].text(0.5,
                   1.075,
                   "\\textbf{Theoretical model parameters " + suffix + "}",
                   va="bottom",
                   ha="center",
                   transform=axs[0, 1].transAxes)
    axs[0, 0].text(-0.36,
                   1.075,
                   "\\textbf{A}",
                   va="bottom",
                   ha="left",
                   size=12,
                   transform=axs[0, 0].transAxes)

    axs[2, 1].text(0.5,
                   1.075,
                   "\\textbf{Fitted model parameters " + suffix + "}",
                   va="bottom",
                   ha="center",
                   transform=axs[2, 1].transAxes)
    axs[2, 0].text(-0.36,
                   1.075,
                   "\\textbf{B}",
                   va="bottom",
                   ha="left",
                   size=12,
                   transform=axs[2, 0].transAxes)

    axs = np.array(axs).T

    fig.align_labels(axs)

    utils.save(fig)

