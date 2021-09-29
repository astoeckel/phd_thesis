import dlop_ldn_function_bases as bases
from construct_lti_common import *


def mk_mod_fourier_basis(q, N):
    N_tot = int(1.1 * N)
    return bases.mk_fourier_basis(q, int(N_tot))[:, (N_tot - N):]

fig = plt.figure(figsize=(8.0, 1.75))

gs1 = fig.add_gridspec(2,
                       3,
                       top=0.9,
                       bottom=0.1,
                       left=0.15,
                       right=0.45,
                       wspace=0.7,
                       hspace=0.4)
axs1 = np.array([[fig.add_subplot(gs1[i, j]) for j in range(3)]
                 for i in range(2)])

gs2 = fig.add_gridspec(2,
                       3,
                       top=0.9,
                       bottom=0.1,
                       left=0.55,
                       right=0.85,
                       wspace=0.7,
                       hspace=0.4)
axs2 = np.array([[fig.add_subplot(gs2[i, j]) for j in range(3)]
                 for i in range(2)])

q = 11
for axs, dampen in [(axs1, False), (axs2, "erasure")]:
    for j, mk_basis_fun in enumerate(
        [mk_mod_fourier_basis, bases.mk_cosine_basis, bases.mk_dlop_basis]):
        H = mk_basis_fun(q, 1000)
        A, _ = reconstruct_lti(H, dampen=dampen)

        s = np.max(np.abs(A))
        axs[0, j].imshow(A, vmin=-s, vmax=s, cmap='RdBu', interpolation='none', extent=[0.5, q + 0.5, q + 0.5, 0.5])
        axs[0, j].set_aspect('auto')

        axs[0, j].spines["bottom"].set_position(("outward", 10.0))
        axs[0, j].spines["left"].set_position(("outward", 10.0))

        axs[0, j].set_xticks(np.arange(0, q, 2) + 1)
        axs[0, j].set_yticks(np.arange(0, q, 2) + 1)
        axs[0, j].set_xticklabels([])

        axs[0, j].set_title(["Mod.~Fourier", "Cosine", "Legendre"][j])

        L = np.linalg.eigvals(A)

        utils.outside_ticks(axs[1, j])

        axs[1, j].axhline(0.0, lw=0.5, linestyle=':', color='grey')
        axs[1, j].axvline(0.0, lw=0.5, linestyle=':', color='grey')

        for i in range(q):
            marker = 'o'
            markersize = 3
            fillstyle = 'full'
            if np.real(L[i]) > 0:
                fillstyle = 'none'
            if np.abs(np.imag(L[i])) < 1e-2:
                marker = 's'

            axs[1, j].plot(np.real(L[i]),
                           np.imag(L[i]),
                           'k',
                           marker=marker,
                           markersize=markersize,
                           fillstyle=fillstyle,
                           clip_on=False)

        axs[1, j].set_xlim(
            min(np.min(np.real(L)), -0.05),
            max(np.max(np.real(L)), 0.05),
        )

        axs[1, j].spines["bottom"].set_position(("outward", 10.0))
        axs[1, j].spines["left"].set_position(("outward", 10.0))

        axs[1, j].set_xlabel("$\\mathrm{Re}(\lambda_i)$")
        if j == 0:
            axs[1, j].set_ylabel("$\\mathrm{Im}(\lambda_i)$")

fig.text(0.075, 1.06, "\\textbf{A}", size=12, va="baseline", ha="left")
fig.text(0.29, 1.06, "\\textbf{Without information erasure}", va="baseline", ha="center")

fig.text(0.475, 1.06, "\\textbf{B}", size=12, va="baseline", ha="left")
fig.text(0.685, 1.06, "\\textbf{With information erasure}", va="baseline", ha="center")


utils.save(fig)

