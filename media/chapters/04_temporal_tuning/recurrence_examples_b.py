import scipy.signal
import nengo
from temporal_encoder_common import *

import runpy

module = runpy.run_path("recurrence_examples_common.py", globals())
plot_result = module["plot_result"]

#
# Panel B: Oscillator
#

fig, axs = plt.subplots(2,
                        2,
                        figsize=(7.9, 1.0),
                        gridspec_kw={
                            "wspace": 0.1,
                            "height_ratios": [3, 1]
                        })

A_tar = np.array((
    (0.0, 1.0 * np.pi),
    (-1.0 * np.pi, 0.0),
))
B_tar = np.array((1.0, 0.0))

tau = 100e-3
W = solve_for_linear_dynamics(
    [Filters.lowpass(tau),
     Filters.lowpass(tau),
     Filters.lowpass(tau)],  # Synaptic filters h
    [
        Filters.dirac(),
        Filters.lti(A_tar, B_tar, 0),
        Filters.lti(A_tar, B_tar, 1)
    ],  # Pre-population tuning a
    [Filters.lti(A_tar, B_tar, 0),
     Filters.lti(A_tar, B_tar, 1)],  # Target tuning e
)

print(W)

plot_result(fig, [(tau, )], [(tau, )], [[
    [W[1, 0], W[1, 1]],
    [W[2, 0], W[2, 1]],
]], [[W[0, 0], W[0, 1]]],
            [Filters.lti(A_tar, B_tar, 0),
             Filters.lti(A_tar, B_tar, 1)],
            E_yoffs=[0.65, 0.65],
            axs=axs)

utils.save(fig)

