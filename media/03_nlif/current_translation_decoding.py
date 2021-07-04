import lif_utils

np.random.seed(489129)


def mk_ensemble(N, d=1, rng=np.random):
    max_rates = rng.uniform(50, 100, N)
    x_intercepts = rng.uniform(-0.99, 0.99, N)

    J0s = lif_utils.lif_rate_inv(1e-3)
    J1s = lif_utils.lif_rate_inv(max_rates)

    gains = (J1s - J0s) / (1.0 - x_intercepts)
    biases = (J0s - x_intercepts * J1s) / (1.0 - x_intercepts)

    encoders = rng.normal(0, 1, (N, d))
    encoders /= np.linalg.norm(encoders, axis=1)[:, None]

    if d == 1:
        idcs = np.argsort(x_intercepts * encoders[:, 0])
    else:
        idcs = np.arange(N)

    return gains[idcs], biases[idcs], encoders[idcs]


N_pre = 100
N_post = 21
N_smpls = 101

xs = np.linspace(-1, 1, N_smpls).reshape(-1, 1)
gains_pre, biases_pre, encoders_pre = mk_ensemble(N_pre)
Js_pre = gains_pre[None, :] * (xs @ encoders_pre.T) + biases_pre[None, :]
As_pre = lif_utils.lif_rate(Js_pre)

gains_post, biases_post, encoders_post = mk_ensemble(N_post)
Js_post = gains_post[None, :] * (xs @ encoders_post.T) + biases_post[None, :]

W1 = np.zeros((N_post, N_pre))
sigma = 0.1 * np.max(As_pre)
for i in range(N_post):
    A = As_pre
    ATA = (A.T @ A + np.square(sigma) * N_smpls * np.eye(N_pre))
    J = A.T @ Js_post[:, i]
    W1[i] = np.linalg.solve(ATA, J)

centres_post = np.sort(np.random.uniform(-0.75, 0.75, N_post))
sigmas_post = np.power(10, np.random.uniform(-0.5, -0.2, N_post))
maxs_post = np.random.uniform(2, 3, N_post)
Js_post2 = maxs_post[None, :] * np.exp(
    -np.square(xs - centres_post[None, :]) / np.square(sigmas_post[None, :]))

W2 = np.zeros((N_post, N_pre))
sigma = 0.1 * np.max(As_pre)
for i in range(N_post):
    A = As_pre
    ATA = (A.T @ A + np.square(sigma) * N_smpls * np.eye(N_pre))
    J = A.T @ Js_post2[:, i]
    W2[i] = np.linalg.solve(ATA, J)

fig, axs = plt.subplots(1,
                        4,
                        figsize=(7.5, 1.5),
                        gridspec_kw={
                            "wspace": 0.4,
                        })

axs[0].plot(xs, As_pre[:, ::2], 'k', linewidth=0.75)
axs[0].set_xlim(-1, 1)
axs[0].set_ylim(0, 100)
axs[0].set_title("\\textbf{Pre tuning-curves}")
axs[0].set_xlabel("Represented $x$")
axs[0].set_ylabel("Rate $a_i$ ($\\mathrm{s}^{-1}$)")
axs[0].text(-0.26,
            1.039,
            "\\textbf{A}",
            size=12,
            va="bottom",
            ha="right",
            transform=axs[0].transAxes)

axs[1].plot(xs, Js_post, 'k:', linewidth=0.5)
for i in range(N_post):
    c = mpl.cm.get_cmap('viridis')(i / (N_post - 1))
    axs[1].plot(xs, As_pre @ W1[i].T, color=c)
axs[1].plot(xs, np.ones_like(xs), 'k--', linewidth=1)
axs[1].set_ylim(0, 3)
axs[1].set_title("\\textbf{Affine $J_i$}")
axs[1].set_xlabel("Encoded $\\xi = \\langle \\mathbf{e}_i, x \\rangle$")
axs[1].set_ylabel("Current $J_i(\\xi)$ (nA)")
axs[1].text(-0.15,
            1.039,
            "\\textbf{B}",
            size=12,
            va="bottom",
            ha="right",
            transform=axs[1].transAxes)

axs[2].plot(xs, Js_post2, 'k:', linewidth=0.5)
for i in range(N_post):
    c = mpl.cm.get_cmap('viridis')(i / (N_post - 1))
    axs[2].plot(xs, As_pre @ W2[i].T, color=c)
axs[2].plot(xs, np.ones_like(xs), 'k--', linewidth=1)
axs[2].set_ylim(0, 3)
axs[2].set_title("\\textbf{Gaussian $J_i$}")
axs[2].set_xlabel("Encoded $\\xi = \\langle \\mathbf{e}_i, x \\rangle$")
axs[2].set_ylabel("Current $J_i(\\xi)$ (nA)")
axs[2].text(-0.15,
            1.039,
            "\\textbf{C}",
            size=12,
            va="bottom",
            ha="right",
            transform=axs[2].transAxes)

U, S, V = np.linalg.svd(W1)
axs[3].bar(np.arange(len(S)) + 1.0 - 0.2,
           S / np.sum(S),
           width=0.4,
           edgecolor='k',
           linewidth=0.5,
           label="Affine",
           color=utils.blues[0])

U, S, V = np.linalg.svd(W2)
axs[3].bar(np.arange(len(S)) + 1.0 + 0.2,
           S / np.sum(S),
           width=0.4,
           edgecolor='k',
           linewidth=0.5,
           label="Gaussian",
           color=utils.oranges[1])

axs[3].set_xlim(0.25, 6.75)
axs[3].set_xticks(np.arange(1, 7))
axs[3].set_xlabel("Singular value index $i$")
axs[3].set_ylabel("Singular value $\\Sigma_i$")
axs[3].set_title("\\textbf{$\\mathbf{W}$ singular values}")
axs[3].legend(loc='upper right',
              fontsize=8,
              handlelength=1.0,
              handletextpad=0.5,
              borderaxespad=0.0,
              labelspacing=0.25)
axs[3].text(-0.225,
            1.039,
            "\\textbf{D}",
            size=12,
            va="bottom",
            ha="right",
            transform=axs[3].transAxes)

utils.save(fig)

