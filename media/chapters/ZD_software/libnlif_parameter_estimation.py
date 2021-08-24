import nlif

with nlif.Neuron() as two_comp_lif:
    with nlif.Soma(v_th=-50e-3, tau_ref=2e-3, tau_spike=1e-3,
                   C_m=1e-9) as soma:
        gL = nlif.CondChan(g=50e-9, E_rev=-65e-3)  # Static leak channel
    with nlif.Compartment(C_m=1e-9) as dendrites:
        gL = nlif.CondChan(g=50e-9, E_rev=-65e-3)  # Static leak channel
        gE = nlif.CondChan(E_rev=0e-3)  # Excitatory input channel
        gI = nlif.CondChan(E_rev=-75e-3)  # Inhibitory input channel
    nlif.Connection(soma, dendrites, g_c=50e-9)
two_comp_lif_assm = two_comp_lif.assemble()

res = 20
fig, axs = plt.subplots(1,
                        3,
                        figsize=(6.6, 1.5),
                        gridspec_kw={
                            "wspace": 0.5,
                            "top": 0.9,
                            "bottom": 0.1,
                        })

gs = fig.add_gridspec(1,
                      1,
                      wspace=0.5,
                      left=0.93,
                      right=0.945,
                      top=0.9,
                      bottom=0.1)
cax = fig.add_subplot(gs[0, 0])

gEs = np.linspace(0, 1e-6, res)
gIs = np.linspace(0, 1e-6, res)
gEss, gIss = np.meshgrid(gEs, gIs)

gs = two_comp_lif_assm.canonicalise_input({gE: gEss, gI: gIss})
sys = two_comp_lif_assm.reduced_system(v_som=None).condition()
i_som_pred = two_comp_lif_assm.i_som(gs, reduced_system=sys)

rates = two_comp_lif_assm.rate_empirical(gs,
                                         T=100.0,
                                         noise=True,
                                         rate=10000,
                                         tau=5e-3)
i_som_ref = two_comp_lif_assm.lif_rate_inv(rates)
valid = rates > 12.5
sys_opt, errs_train = nlif.parameter_optimisation.optimise_trust_region(
    sys, gs_train=gs[valid], Js_train=i_som_ref[valid], N_epochs=10)
i_som_pred_opt = two_comp_lif_assm.i_som(gs, reduced_system=sys_opt)


def plot_contour(ax, cax, i_pred):
    C = ax.contourf(gEs * 1e9, gIs * 1e9, i_som_ref * 1e9, cmap='inferno')
    ax.contour(gEs * 1e9,
               gIs * 1e9,
               i_pred * 1e9,
               levels=C.levels,
               colors=['white'],
               linewidths=[1.0],
               linestyles=['--'])

    if not cax is None:
        cb = plt.colorbar(C, ax=ax, cax=cax, orientation='vertical')
        cb.outline.set_visible(False)
        cb.set_label("Current $J$ ($\\si{\\nano\\ampere}$)")

    ax.set_xlabel("Excitatory cond. $g_\\mathrm{E}$ (\\si{\\nano\\siemens})")
    ax.set_ylabel("Inhibitory cond. $g_\\mathrm{I}$ (\\si{\\nano\\siemens})")


plot_contour(axs[0], cax, i_som_pred)

axs[1].plot(errs_train, 'k+-')
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("NRMSE")
axs[1].set_ylim(0.0, None)

axs[0].set_title("\\textbf{Uncalibrated prediction}")
axs[0].text(-0.35, 1.07, "\\textbf{A}", size=12, va="baseline", ha="left", transform=axs[0].transAxes)

axs[1].set_title("\\textbf{Training errors}")
axs[1].text(-0.35, 1.07, "\\textbf{B}", size=12, va="baseline", ha="left", transform=axs[1].transAxes)

axs[2].set_title("\\textbf{Calibrated prediction}")
axs[2].text(-0.35, 1.07, "\\textbf{C}", size=12, va="baseline", ha="left", transform=axs[2].transAxes)

plot_contour(axs[2], None, i_som_pred_opt)

utils.save(fig)

