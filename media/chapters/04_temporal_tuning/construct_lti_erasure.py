import scipy.linalg
import dlop_ldn_function_bases as bases

from construct_lti_common import *

fig = plt.figure(figsize=(7.4, 3.5))
gs = fig.add_gridspec(9, 3, wspace=0.3, hspace=1.0)
axs = np.array([[
    fig.add_subplot(gs[0:3, j]),
    fig.add_subplot(gs[3:6, j]),
    fig.add_subplot(gs[7:9, j])
] for j in range(3)]).T


def mk_mod_fourier_basis(q, N):
    N_tot = int(1.1 * N)
    return bases.mk_fourier_basis(q, int(N_tot))[:, (N_tot - N):]


plot_errors_and_impulse_response(
    axs, [mk_mod_fourier_basis, bases.mk_cosine_basis, bases.mk_dlop_basis],
    ["Modified Fourier basis", "Cosine basis", "Legendre basis"],
    T=2.0,
    err_min=1e-2,
    y_letter=1.095,
    dampen="erasure",
    plot_zero_errs=True,
    utils=utils)

utils.save(fig)

