import nlif

np.random.seed(528992)


class LIF:
    slope = 2.0 / 3.0

    @staticmethod
    def inverse(a):
        valid = a > 0
        return 1.0 / (1.0 - np.exp(LIF.slope - (1.0 / (valid * a + 1e-6))))

    @staticmethod
    def activity(x):
        valid = x > (1.0 + 1e-6)
        return valid / (LIF.slope - np.log(1.0 - valid * (1.0 / x)))


class Ensemble:
    def __init__(self, n_neurons, n_dimensions, neuron_type=LIF):
        self.neuron_type = neuron_type

        # Randomly select the intercepts and the maximum rates
        self.intercepts = np.random.uniform(-0.95, 0.95, n_neurons)
        self.max_rates = np.random.uniform(0.5, 1.0, n_neurons)

        # Randomly select the encoders
        self.encoders = np.random.normal(0, 1, (n_neurons, n_dimensions))
        self.encoders /= np.linalg.norm(self.encoders, axis=1)[:, None]

        # Compute the current causing the maximum rate/the intercept
        J_0 = self.neuron_type.inverse(0)
        J_max_rates = self.neuron_type.inverse(self.max_rates)

        # Compute the gain and bias
        self.gain = (J_0 - J_max_rates) / (self.intercepts - 1.0)
        self.bias = J_max_rates - self.gain

    def __call__(self, x):
        return self.neuron_type.activity(self.J(x))

    def J(self, x):
        return self.gain[:, None] * self.encoders @ x + self.bias[:, None]

with nlif.Neuron() as two_comp_lif:
	with nlif.Soma(v_th=-50e-3, tau_ref=2e-3, tau_spike=1e-3, C_m=1e-9) as soma:
		gL = nlif.CondChan(g=50e-9, E_rev=-65e-3)  # Static leak channel
	with nlif.Compartment(C_m=1e-9) as dendrites:
		gL = nlif.CondChan(g=50e-9, E_rev=-65e-3)  # Static leak channel
		gE = nlif.CondChan(E_rev=0e-3)             # Excitatory input channel
		gI = nlif.CondChan(E_rev=-75e-3)           # Inhibitory input channel
	nlif.Connection(soma, dendrites, g_c=50e-9)
two_comp_lif_assm = two_comp_lif.assemble()

gEs, gIs = np.linspace(0, 1e-6, 10), np.linspace(0, 1e-6, 10) 
gEss, gIss = np.meshgrid(gEs, gIs)        # Generate a dense sample grid
rates = two_comp_lif_assm.rate_empirical( # Simulate the neuron at each sample
	{gE: gEss, gI: gIss}, T=100.0, noise=True, rate=10000, tau=5e-3)

sys = two_comp_lif_assm.reduced_system(v_som=None)

i_som_pred = two_comp_lif_assm.i_som({gE: gEss, gI: gIss}, reduced_system=sys)

gs = two_comp_lif_assm.canonicalise_input(
	{gE: gEss, gI: gIss})   # Arrange the inputs in a matrix
sys = sys.condition()       # Condition the reduced system
i_som_ref = two_comp_lif_assm.lif_rate_inv(rates) # Apply G^-1
valid = rates > 12.5        # Discard subthreshold samples
sys_opt, errs_train = nlif.parameter_optimisation.optimise_trust_region(
	sys, gs_train=gs[valid], Js_train=i_som_ref[valid], N_epochs=10)

# Sample the represented space
xs, ys = np.linspace(-1, 1, 101), np.linspace(-1, 1, 101)
xss, yss = np.meshgrid(xs, ys)
Xs = np.array((xss.flatten(), yss.flatten())).T

# Obtain the target current function and pre-activities
Js = 0.5 * (1.0 + Xs[:, 0]) * (1.0 + Xs[:, 1]) * 1e-9

ens1, ens2 = Ensemble(101, 1), Ensemble(102, 1)
As = np.concatenate((ens1(Xs[:, 0:1].T), ens2(Xs[:, 1:2].T))).T

W_mask = np.ones((2, As.shape[1]), dtype=bool) # All-to-all

# Randomly select some training samples
idcs_train = np.random.randint(0, Xs.shape[0], 256)

W, errs_train = nlif.weight_optimisation.optimise_trust_region(sys_opt,
    As_train=As[idcs_train], Js_train=Js[idcs_train], W_mask=W_mask,
    N_epochs=10)

# Compute the decoded currents
Js_dec = two_comp_lif_assm.i_som(As @ W.T, reduced_system=sys_opt)

fig, axs = plt.subplots(1, 3, figsize=(7.45, 1.8), gridspec_kw={
	"wspace": 0.4
})

axs[0].plot(xs, As.reshape(101, 101, -1)[:, 0, 101::2], 'k-', linewidth=0.7)
axs[0].set_xlim(-1, 1)
axs[0].set_ylim(0, 1)
axs[0].set_xlabel("Encoded $\\xi$")
axs[0].set_ylabel("Normalised activity $a_i$")

axs[1].plot(errs_train, 'k+-')
axs[1].set_ylim(0, None)
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("NRMSE")

C = axs[2].contourf(xs, ys, Js_dec.reshape(101, 101))
axs[2].contour(xs, ys, Js_dec.reshape(101, 101),
               levels=C.levels,
               colors=['white'],
               linewidths=[0.7],
               linestyles=[':'])
axs[2].contour(xs, ys, Js.reshape(101, 101),
               levels=C.levels,
               colors=['white'],
               linewidths=[1.0],
               linestyles=['--'])
axs[2].set_xlabel("Represented $x_1$")
axs[2].set_ylabel("Represented $x_2$")

axs[0].set_title("\\textbf{Tuning curves}")
#axs[0].text(-0.28, 1.065, "\\textbf{A}", size=12, va="baseline", ha="left", transform=axs[0].transAxes)

axs[1].set_title("\\textbf{Training errors}")
#axs[1].text(-0.25, 1.065, "\\textbf{B}", size=12, va="baseline", ha="left", transform=axs[1].transAxes)

axs[2].set_title("\\textbf{Decoded currents}")
#axs[2].text(-0.28, 1.065, "\\textbf{C}", size=12, va="baseline", ha="left", transform=axs[2].transAxes)


utils.save(fig)
