import nlif

with nlif.Neuron() as two_comp_lif:
	with nlif.Soma(v_th=-50e-3, tau_ref=2e-3, tau_spike=1e-3, C_m=1e-9) as soma:
		gL = nlif.CondChan(g=50e-9, E_rev=-65e-3)  # Static leak channel
	with nlif.Compartment(C_m=1e-9) as dendrites:
		gL = nlif.CondChan(g=50e-9, E_rev=-65e-3)  # Static leak channel
		gE = nlif.CondChan(E_rev=0e-3)             # Excitatory input channel
		gI = nlif.CondChan(E_rev=-75e-3)           # Inhibitory input channel
	nlif.Connection(soma, dendrites, g_c=50e-9)
two_comp_lif_assm = two_comp_lif.assemble()

res = 100
fig, axs = plt.subplots(1, 3, figsize=(7.3, 1.75), gridspec_kw={
    "wspace": 0.5
})

gs = fig.add_gridspec(1, 3, wspace=0.6, left=0.13, right=0.895, top=-0.12, bottom=-0.15)
caxs = [fig.add_subplot(gs[0, i]) for i in range(3)]

def run_and_plot(ax, cax, **kwargs):
    gEs = np.linspace(0, 1e-6, res)
    gIs = np.linspace(0, 1e-6, res)
    gEss, gIss = np.meshgrid(gEs, gIs)

    def do_run(*args, **kwargs):
        return two_comp_lif_assm.rate_empirical(*args, **kwargs)

    rates = utils.run_with_cache(do_run, {
            gE: gEss,
            gI: gIss
        }, **kwargs)

    levels = np.linspace(0, 100, 11)

    C = ax.contourf(gEs * 1e9, gIs * 1e9, rates, levels=levels)
    ax.contour(gEs * 1e9, gIs * 1e9, rates, levels=C.levels, linestyles=[':'], colors=['white'], linewidths=[0.7])
    ax.set_aspect(1)
    ax.set_xlabel("Excitatory cond. $g_\\mathrm{E}$ (\\si{\\nano\\siemens})")
    ax.set_ylabel("Inhibitory cond. $g_\\mathrm{I}$ (\\si{\\nano\\siemens})")

    cb = plt.colorbar(C, ax=ax, cax=cax, orientation='horizontal')
    cb.outline.set_visible(False)
    cb.set_label("Rate $\\mathscr{G}(g_\\mathrm{E}, g_\\mathrm{I})$ (\\si{\per\second})")


run_and_plot(axs[0], caxs[0])
axs[0].set_title("\\textbf{No noise}")

run_and_plot(axs[1], caxs[1], T=100.0, noise=True, rate=1000, tau=5e-3)
axs[1].set_title("\\textbf{With noise} ($\\lambda = 1000$)")

run_and_plot(axs[2], caxs[2], T=100.0, noise=True, rate=100, tau=5e-3)
axs[2].set_title("\\textbf{With noise} ($\\lambda = 100$)")

utils.save(fig)
