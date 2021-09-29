import scipy.linalg
import dlop_ldn_function_bases as bases

from construct_lti_common import *

fig, axs = plt.subplots(2,
                        3,
                        figsize=(7.5, 2.74),
                        gridspec_kw={
                            "wspace": 0.3,
                            "hspace": 0.6,
                            "height_ratios": [2.5, 2],
                        })


def windowed(fun):
    def mk_windowed_basis_fun(q, N):
        return fun(q, N) * np.linspace(0, 1, N)[None, :]

    return mk_windowed_basis_fun


plot_errors_and_impulse_response(axs, [
    windowed(bases.mk_fourier_basis),
    windowed(bases.mk_cosine_basis),
    windowed(bases.mk_dlop_basis)
], ["Fourier basis", "Cosine basis", "Legendre basis"],
                                 T=2.0,
                                 err_min=0.5e-2,
                                 y_letter=1.11,
                                 utils=utils)

utils.save(fig)


