import lif_utils

N = 10
min_max_rate = 10
max_max_rate = 50

np.random.seed(123486)

# Randomly select encoders, max_rates, x-intercept
E = np.random.choice([-1, 1], N)
max_rates = np.random.uniform(min_max_rate, max_max_rate, N)
xi = np.random.uniform(-0.95, 0.95, N)

# Sort the curves by x-intercept for a nicer plotting
order = np.argsort(xi)
E = E[order]
xi = xi[order]

# Convert the maximum rate and the x-intercept to a gain and bias
J0 = lif_utils.lif_rate_inv(1e-3, tau_rc=100e-3)
J1 = lif_utils.lif_rate_inv(max_rates, tau_rc=100e-3)
gain = (J0 - J1) / (E * xi - 1.0)
bias = J1 - gain

# Compute decoders for a linear function
xs = np.linspace(-1, 1, 1000)
f = lambda x: 2 * x**2 - 1
A = lif_utils.lif_rate(np.outer(xs, gain * E) + bias, tau_rc=100e-3)
w = np.linalg.lstsq(A.T @ A + np.eye(N) * np.max(A) * 1e-1,
                    A.T @ xs,
                    rcond=None)[0]
w_f = np.linalg.lstsq(A.T @ A + np.eye(N) * np.max(A) * 1e-1,
                      A.T @ f(xs),
                      rcond=None)[0]

# Plot all important quantities
fig, axs = plt.subplots(1, 3, figsize=(7.2, 1.8), gridspec_kw={"wspace": 0.35})

cmap = cm.get_cmap('viridis')
colours = np.array(list(map(cmap, np.linspace(0, 1,
                                              N)))) * [0.75, 0.75, 0.75, 1.0]
#cmap = cm.get_cmap('tab10')
#colours = [cmap(x) for x in np.linspace(0, 1, N, endpoint=False)]

for i, fJ in enumerate((np.outer(xs, gain * E) + bias).T):
    axs[0].plot(xs, fJ, color=colours[i])
axs[0].plot([-1, 1], [J0, J0], 'k', linestyle=(0, (3, 3)), linewidth=0.5)
axs[0].plot([-1, 1], [J0, J0], 'white', linestyle=(3, (3, 3)), linewidth=0.5)
axs[0].set_xlim(-1, 1)
axs[0].set_ylim(0, np.max(J1))
axs[0].set_title('Current translation')
axs[0].set_xlabel('Input $\\langle \\vec x, \\vec e_i \\rangle$')
axs[0].set_ylabel('Current $J_i$')
axs[0].text(-0.18,
            1.0325,
            "\\textbf{B}",
            fontdict={"size": 12},
            transform=axs[0].transAxes,
            va='bottom',
            ha='left')

for i, fA in enumerate(A.T):
    axs[1].plot(xs, fA, color=colours[i])
axs[1].set_xlim(-1, 1)
axs[1].set_ylim(0, max_max_rate)
axs[1].set_title('Tuning curves')
axs[1].set_xlabel('Input $\\langle \\vec x, \\vec e_i \\rangle$')
axs[1].set_ylabel('Activity $a_i$ ($\mathrm{s}^{-1}$)')
axs[1].text(-0.225,
            1.0325,
            "\\textbf{C}",
            fontdict={"size": 12},
            transform=axs[1].transAxes,
            va='bottom',
            ha='left')

axs[2].plot(xs, A @ w, color=colours[0])
axs[2].plot(xs, xs, 'k', linestyle=(0, (3, 3)), linewidth=0.5)
axs[2].plot(xs, xs, 'white', linestyle=(3, (3, 3)), linewidth=0.5)
axs[2].set_xlim(-1, 1)
axs[2].set_ylim(-1, 1)
axs[2].set_title('Decoding')
axs[2].set_xlabel('Represented $\\vec x$')
axs[2].set_ylabel('Decoded $\\vec{\\hat x}$')
axs[2].text(-0.28,
            1.0325,
            "\\textbf{D}",
            fontdict={"size": 12},
            transform=axs[2].transAxes,
            va='bottom',
            ha='left')

#axs[3].plot(xs, A @ w_f, color=colours[0])
#axs[3].plot(xs, f(xs), 'k', linestyle=(0, (3, 3)), linewidth=0.5)
#axs[3].plot(xs, f(xs), 'white', linestyle=(3, (3, 3)), linewidth=0.5)
#axs[3].set_xlim(-1, 1)
#axs[3].set_ylim(-1, 1)
#axs[3].set_title('Transformation')
#axs[3].set_xlabel('Represented $x$')
#axs[3].set_ylabel('Decoded $f(x)$')

utils.save(fig)

