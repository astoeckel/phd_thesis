import scipy.signal
import nengo
from temporal_encoder_common import *

import runpy

module = runpy.run_path("recurrence_examples_common.py", globals())
plot_result = module["plot_result"]

#
# Panel A: Integrator
#

fig, axs = plt.subplots(2,
                        2,
                        figsize=(7.9, 1.0),
                        gridspec_kw={
                            "wspace": 0.1,
                            "height_ratios": [3, 1]
                        })

tau0B = 10e-3
tau1B = 20e-3
tau2B = 30e-3

tau0A = 100e-3
tau1A = 110e-3
tau2A = 120e-3

W = solve_for_linear_dynamics(
    [
        Filters.lowpass(tau0B, 1),
        Filters.lowpass(tau1B, 1),
        Filters.lowpass(tau2B, 1),
        Filters.lowpass(tau0A, 1),
        Filters.lowpass(tau1A, 1),
        Filters.lowpass(tau2A, 1)
    ],
    [
        Filters.dirac(),
        Filters.dirac(),
        Filters.dirac(),
        Filters.step(),
        Filters.step(),
        Filters.step()
    ],
    [Filters.step()],
)

flt_rec = [(tau0A, 1), (tau1A, 1), (tau2A, 1)]
flt_pre = [(tau0B, 1), (tau1B, 1), (tau2B, 1)]

A = np.array([
    [[W[3, 0]]],
    [[W[4, 0]]],
    [[W[5, 0]]],
])
B = np.array([
    [W[0, 0]],
    [W[1, 0]],
    [W[2, 0]],
])

plot_result(fig, flt_rec, flt_pre, A, B, [Filters.step()], axs=axs)

# Reference run

tauB, tauA = 20e-3, 110e-3

W = solve_for_linear_dynamics(
    [
        Filters.lowpass(tauB, 1),
        Filters.lowpass(tauA, 1),
    ],
    [
        Filters.dirac(),
        Filters.step(),
    ],
    [Filters.step()],
)

flt_rec, flt_pre = [
    (tauA, 1),
], [
    (tauB, 1),
]

A = np.array([[[W[1, 0]]]])
B = np.array([[W[0, 0]]])

plot_result(fig,
            flt_rec,
            flt_pre,
            A,
            B, [Filters.step()],
            axs=axs,
            ref_plot=True)

utils.annotate(axs[0, 0],
               2.5,
               0.65,
               3.5,
               0.3,
               "Single filter response",
               ha="left")

utils.save(fig)

