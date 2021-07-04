import lif_utils

N = 20
min_max_rate = 10
max_max_rate = 50

#np.random.seed(123486)
np.random.seed(7897456)

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
A = lif_utils.lif_rate(np.outer(xs, gain * E) + bias, tau_rc=100e-3)

# Plot all important quantities
fig = plt.figure(figsize=(6.5, 2.2))

gs1 = fig.add_gridspec(1, 1, top=0.975, bottom=0.675, left=0.4125, right=0.5875)
gs2 = fig.add_gridspec(1,
                       4,
                       top=0.475,
                       bottom=0.15,
                       left=0.05,
                       right=0.95,
                       wspace=0.5)

ax0 = fig.add_subplot(gs1[0, 0])
ax0.text(-0.3, 1.26, '\\textbf{A}', va="top", ha="left", transform=ax0.transAxes, size=12)
axs = [fig.add_subplot(gs2[0, i]) for i in range(4)]

cmap = cm.get_cmap('viridis')
colours = np.array(list(map(cmap, np.linspace(0, 1,
                                              N)))) * [0.75, 0.75, 0.75, 1.0]

for i in range(A.shape[1]):
    ax0.plot(xs, A[:, i], color=colours[i])
#ax0.set_xlabel('Represented $x$', labelpad=2.0)
ax0.set_ylabel('$a_i(x)$ ($\\mathrm{s}^{-1}$)')
ax0.set_xlim(-1.125, 1.125)
ax0.set_ylim(0, 50)
ax0.set_yticks([0, 25, 50])
ax0.set_title('Tuning curves $\\vec a(x)$')
#ax0.set_xticks([])
#ax0.set_xticklabels([])
ax0.spines["bottom"].set_visible(False)

fs = [
    lambda x: x, lambda x: 2 * np.square(x) - 1,
    lambda x: np.sin(x * 1.5 * np.pi), lambda x: 2.0 * (x >= 0.0) - 1.0
]
fs_name = [
    "$f(x) = x$",
    "$f(x) = 2 x^2 - 1$",
    "$f(x) = \\sin(1.5x)$",
    "$f(x) = \\sigma(x)$",
]
for j in range(4):
    f = fs[j]
    D = np.linalg.lstsq(A.T @ A + np.eye(N) * np.max(A) * 1e-1,
                        A.T @ f(xs),
                        rcond=1e-3)[0]
    E = np.sqrt(np.mean(np.square(f(xs) - A @ D.T)))
    axs[j].plot(xs, f(xs), 'k-', linewidth=0.6)
    axs[j].plot(xs, A @ D.T, 'k-', clip_on=True)
    axs[j].plot(xs, f(xs), linestyle=':', color='white', linewidth=0.5)
    axs[j].set_xlim(-1.125, 1.125)
    axs[j].set_ylim(-1.25, 1.25)
    #axs[j].set_xticks([])
    #axs[j].set_xticklabels([])
    #axs[j].spines["bottom"].set_visible(False)
    axs[j].set_xlabel("Represented $x$", labelpad=2.0)
    axs[j].set_ylabel("Decoded $y$", labelpad=2.0)
    axs[j].text(0.05,
                0.95,
                f"$E = {E:0.2f}$",
                va='top',
                ha="left",
                transform=axs[j].transAxes,
                bbox={
                    "pad": 1.0,
                    "color": "w",
                    "linewidth": 0.0,
                })
    axs[j].set_title(fs_name[j])
    axs[j].text(-0.3, 1.26, '\\textbf{{{}}}'.format(chr(ord('B') + j)), va="top", ha="left", transform=axs[j].transAxes, size=12)

utils.save(fig)

