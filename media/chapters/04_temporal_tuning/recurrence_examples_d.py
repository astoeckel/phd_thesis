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

tau0B = 100e-3
tau1B = 110e-3
tau2B = 120e-3

tau0A = 200e-3
tau1A = 210e-3
tau2A = 220e-3

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
        Filters.lowpass(1e-3),
        Filters.lowpass(1e-3),
        Filters.lowpass(1e-3)
    ],
    [Filters.lowpass(1e-3)],
    T=10.0,
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

plot_result(fig,
            flt_rec,
            flt_pre,
            A,
            B, [Filters.lowpass(1e-3)],
            E_yoffs=[0.0, 0.3],
            axs=axs)

# Reference NEF solution

tauB, tauA = 110e-3, 210e-3

W = solve_for_linear_dynamics(
    [Filters.lowpass(tauB, 1),
     Filters.lowpass(tauA, 1)],
    [
        Filters.dirac(),
        Filters.lowpass(10e-3),
    ],
    [Filters.lowpass(10e-3)],
    T=10.0,
)

flt_rec, flt_pre = [(tauA, 1)], [(tauB, 1)]

A, B = np.array([[[W[1, 0]]]]), np.array([[W[0, 0]]])

plot_result(fig,
            flt_rec,
            flt_pre,
            A,
            B, [Filters.lowpass(1e-3)],
            ref_plot=True,
            axs=axs)

utils.save(fig)


