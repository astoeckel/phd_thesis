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

dt, ss, T = 1e-4, 10, 1.0  # Simulation time-step, sub-sampling and end-time
ts = np.arange(0, T, ss * dt)   # Sample points
gEs = np.linspace(0.0, 200e-9, len(ts)) # Sampled input conductances
with nlif.Simulator(two_comp_lif_assm, dt=dt, ss=ss,
                    record_voltages=True, record_spike_times=True) as sim:
	res = sim.simulate({   # Also: simulate_poisson, simulate_filtered
		gE: gEs,           # Sampled input
		gI: 10e-9,       # Constant input
	})

fig, axs = plt.subplots(4, 1, figsize=(7.4, 2.5), gridspec_kw={
	"height_ratios": [1, 3, 3, 2],
	"hspace": 0.5,
}, sharex=False)

for t in res.times:
	axs[0].plot([t, t], [0, 1], 'k-', solid_capstyle='round')
axs[0].spines["left"].set_visible(False)
axs[0].spines["bottom"].set_visible(False)
axs[0].set_xlim(0, T)
axs[0].set_yticks([])
axs[0].set_xticks([])
utils.annotate(axs[0], 0.38, 0.5, 0.275, 0.5, "Output spike times", ha="right")

axs[1].plot(ts, res.v[:, 0] * 1e3, 'k')
axs[1].set_xlim(0, T)
axs[1].set_xticklabels([])
axs[1].set_ylabel("$v_1$ (mV)")
axs[1].set_ylim(-80, 20)
utils.annotate(axs[1], 0.38, 0.5, 0.275, 0.5, "Somatic membrane potential", ha="right")

axs[2].plot(ts, res.v[:, 1] * 1e3, 'k')
axs[2].set_xlim(0, T)
axs[2].set_xticklabels([])
axs[2].set_ylabel("$v_2$ (mV)")
axs[2].set_ylim(-80, 20)
utils.annotate(axs[2], 0.38, -26, 0.275, -10, "Dendritic membrane potential", ha="right")

axs[3].plot(ts, gEs * 1e9, 'k')
axs[3].set_xlim(0, T)
axs[3].set_ylabel("$g_\\mathrm{E}$ (nS)")
axs[3].set_xlabel("Time $t$ (s)")
utils.annotate(axs[3], 0.38, 100, 0.275, 150, "Excitatory conductance", ha="right")

utils.save(fig)

