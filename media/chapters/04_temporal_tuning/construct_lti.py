import scipy.linalg
import dlop_ldn_function_bases as bases

from construct_lti_common import *

fig, axs = plt.subplots(2,
                        3,
                        figsize=(7.5, 3.5),
                        gridspec_kw={
                            "wspace": 0.3,
                            "hspace": 0.5,
                            "height_ratios": [3, 2],
                        })

plot_errors_and_impulse_response(
    axs, [bases.mk_fourier_basis, bases.mk_cosine_basis, bases.mk_dlop_basis],
    ["Fourier basis", "Cosine basis", "Legendre basis"], utils=utils)

utils.save(fig)

