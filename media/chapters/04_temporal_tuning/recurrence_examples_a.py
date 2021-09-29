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

tau = 100e-3
W = solve_for_linear_dynamics(
    [Filters.lowpass(tau), Filters.lowpass(tau)],  # Synaptic filters h
    [Filters.dirac(), Filters.step()],  # Pre-population tuning a
    [Filters.step()]  # Target tuning e
)

plot_result(fig, [(tau, )], [(tau, )],
            W[1, 0],
            W[0, 0], [Filters.step()],
            axs=axs)

utils.annotate(axs[1, 0], 1.65, 1.05, 2.0, 1.4, "Input $u(t)$", ha="left")
utils.annotate(axs[0, 0],
               1.2,
               0.5,
               2.0,
               0.5,
               "System response $x(t)$",
               ha="left")

utils.save(fig)

