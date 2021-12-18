#    Code for the "Nonlinear Synaptic Interaction" Paper
#    Copyright (C) 2017-2020   Andreas Stöckel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

from nef_synaptic_computation.multi_compartment_lif import *

#
# Neuron models used in the actual simulator
#

NEURON_MODELS = {
    "linear": (Neuron().add_compartment(
        Compartment(soma=True).add_channel(
            CondChan(Erev=-65e-3, g=50e-9, name="leak")).add_channel(
                CurChan(mul=1, name="exc")).add_channel(
                    CurChan(mul=-1, name="inh"))).assemble()),
    "gc50": (Neuron().add_compartment(
        Compartment(soma=True).add_channel(
            CondChan(Erev=-65e-3, g=50e-9, name="leak"))).add_compartment(
                Compartment(name="dendrites").add_channel(
                    CondChan(Erev=20e-3, name="exc")).add_channel(
                        CondChan(Erev=-75e-3, name="inh")).add_channel(
                            CondChan(Erev=-65e-3, g=50e-9,
                                     name="leak"))).connect(
                                         "soma", "dendrites",
                                         0.05e-6).assemble()),
    "gc100": (Neuron().add_compartment(
        Compartment(soma=True).add_channel(
            CondChan(Erev=-65e-3, g=50e-9, name="leak"))).add_compartment(
                Compartment(name="dendrites").add_channel(
                    CondChan(Erev=20e-3, name="exc")).add_channel(
                        CondChan(Erev=-75e-3, name="inh")).add_channel(
                            CondChan(Erev=-65e-3, g=50e-9,
                                     name="leak"))).connect(
                                         "soma", "dendrites",
                                         0.1e-6).assemble()),
    "gc200": (Neuron().add_compartment(
        Compartment(soma=True).add_channel(
            CondChan(Erev=-65e-3, g=50e-9, name="leak"))).add_compartment(
                Compartment(name="dendrites").add_channel(
                    CondChan(Erev=20e-3, name="exc")).add_channel(
                        CondChan(Erev=-75e-3, name="inh")).add_channel(
                            CondChan(Erev=-65e-3, g=50e-9,
                                     name="leak"))).connect(
                                         "soma", "dendrites",
                                         0.2e-6).assemble())
}

NEURON_MODELS_KEYS = list(NEURON_MODELS.keys())

#
# Corresponding solver parameters
#

SOLVER_PARAMS = {
    "linear": {
        "b0": 0.0,
        "b1": 1.0,
        "b2": -1.0,
        "a0": 1.0,
        "a1": 0.0,
        "a2": 0.0,
    },
    "linear_2d": {
        "b0": 0.0,
        "b1": 1.0,
        "b2": -1.0,
        "a0": 1.0,
        "a1": 0.0,
        "a2": 0.0,
    },
    "gc50_no_noise": {
        "b0": -19.5e-6,
        "b1": 1000.0,
        "b2": -425.5,
        "a0": 15.7e3,
        "a1": 296.4e9,
        "a2": 132.2e9,
    },
    "gc100_no_noise": {
        "b0": -18.8e-6,
        "b1": 1000.0,
        "b2": -376.0,
        "a0": 9.0e3,
        "a1": 193.0e9,
        "a2": 41.6e9,
    },
    "gc200_no_noise": {
        "b0": -17.1e-6,
        "b1": 1000.0,
        "b2": -352.3,
        "a0": 8.3e3,
        "a1": 113.3e9,
        "a2": 18.6e9
    },
    "gc50_noisy": {
        "b0": -26.3e-6,
        "b1": 1000.0,
        "b2": -487.5,
        "a0": 5.9e3,
        "a1": 350.7e9,
        "a2": 26.8e9,
        "α": 51.3e9,
        "β": -22.8,
    },
    "gc100_noisy": {
        "b0": -20.7e-6,
        "b1": 1000.0,
        "b2": -368.3,
        "a0": 4.2e3,
        "a1": 260.6e9,
        "a2": 7.0e9,
        "α": 51.5e9,
        "β": -25.1,
    },
    "gc200_noisy": {
        "b0": -17.1e-6,
        "b1": 1000.0,
        "b2": -307.6,
        "a0": 5.3e3,
        "a1": 185.1e9,
        "a2": 17.1e9,
        "α": 51.3e9,
        "β": -26.5,
    },
}

def mkws(p_key):
    p = SOLVER_PARAMS[p_key]
    ws = np.array((p["b0"], p["b1"], p["b2"], p["a0"], p["a1"], p["a2"]))
    return ws / ws[1] # Make sure that b1 is one


SOLVER_PARAMS_KEYS = list(SOLVER_PARAMS.keys())

N_SOLVER_PARAMS = len(SOLVER_PARAMS)

SOLVER_PARAMS_SWEEP_KEYS = ["linear", "linear_2d", "gc50_no_noise", "gc50_noisy", "gc100_no_noise", "gc100_noisy", "gc200_no_noise", "gc200_noisy"]

N_SOLVER_PARAMS_SWEEP = len(SOLVER_PARAMS_SWEEP_KEYS)

SOLVER_REG_MAP = {
    ('gc100_no_noise', False): 0.001,
    ('gc100_no_noise', True): 0.0775825016856679,
    ('gc100_noisy', False): 0.0029678431503900376,
    ('gc100_noisy', True): 0.0964388379154446,
    ('gc200_no_noise', False): 0.0011149209970080915,
    ('gc200_no_noise', True): 0.04503428864545838,
    ('gc200_noisy', False): 0.001,
    ('gc200_noisy', True): 0.062413076493974616,
    ('gc50_no_noise', False): 0.050209673996144656,
    ('gc50_no_noise', True): 0.11987818459583773,
    ('gc50_noisy', False): 0.029145192568009907,
    ('gc50_noisy', True): 0.11987818459583773,
    ('linear', False): 81.91922344953684,
    ('linear', True): 22.205302450155354,
    ('linear_2d', False): 4.842162612301507,
    ('linear_2d', True): 4.842162612301507
}

NETWORK_PARAMS_SWEEP_KEYS = ["linear", "linear_2d", "gc50_no_noise", "gc50_noisy", "gc100_no_noise", "gc100_noisy", "gc200_no_noise", "gc200_noisy"]

N_NETWORK_PARAMS_SWEEP = len(NETWORK_PARAMS_SWEEP_KEYS)

NETWORK_REG_MAP = {
    ('linear', False): 93.71019332979176,
    ('linear', True): 68.21121719582916,
    ('linear_2d', False): 15.644344352347058,
    ('linear_2d', True): 11.721022975334806,
    ('gc50_no_noise', False): 0.011553030629911948,
    ('gc50_no_noise', True): 0.10748502514263064,
    ('gc50_noisy', False): 0.001241777787729667,
    ('gc50_noisy', True): 0.001,
    ('gc100_no_noise', False): 0.05493029257717619,
    ('gc100_no_noise', True): 0.16937141873226041,
    ('gc100_noisy', False): 0.001,
    ('gc100_noisy', True): 8.531678524172806,
    ('gc200_no_noise', False): 0.035622478902624426,
    ('gc200_no_noise', True): 0.04423520304514771,
    ('gc200_noisy', False): 1.9567531884454445,
    ('gc200_noisy', True): 8.531678524172806
}

NETWORK_FILTER_REG_MAP = {
    ('linear', False): 38.28883040491414,
    ('linear', True): 38.28883040491414,
    ('linear_2d', False): 8.78160033390695,
    ('linear_2d', True): 8.531678524172806,
    ('gc50_no_noise', False): 0.001,
    ('gc50_no_noise', True): 0.0015420120740987876,
    ('gc50_noisy', False): 0.001241777787729667,
    ('gc50_noisy', True): 0.001,
    ('gc100_no_noise', False): 0.001,
    ('gc100_no_noise', True): 0.004754621911497074,
    ('gc100_noisy', False): 0.001,
    ('gc100_noisy', True): 8.531678524172806,
    ('gc200_no_noise', False): 0.001,
    ('gc200_no_noise', True): 0.32431854788917563,
    ('gc200_noisy', False): 1.9567531884454445,
    ('gc200_noisy', True): 8.78160033390695
}

NETWORK_FILTER_TAU_MAP = {
    ('linear', False): 0.03650144272877066,
    ('linear', True): 0.02056221561374837,
    ('linear_2d', False): 0.011583232862547104,
    ('linear_2d', True): 0.011583232862547104,
    ('gc50_no_noise', False): 0.013511360701886646,
    ('gc50_no_noise', True): 0.011583232862547104,
    ('gc50_noisy', False): 0.001,
    ('gc50_noisy', True): 0.001,
    ('gc100_no_noise', False): 0.015325347622834329,
    ('gc100_no_noise', True): 0.013323553212817248,
    ('gc100_noisy', False): 0.001,
    ('gc100_noisy', True): 0.0017751707021477282,
    ('gc200_no_noise', False): 0.023651581311899404,
    ('gc200_no_noise', True): 0.017876382359836518,
    ('gc200_noisy', False): 0.001,
    ('gc200_noisy', True): 0.017627901206922238
}

#
# Fixed 2D Functions
#

BENCHMARK_FUNCTIONS = {
    "addition": lambda x, y: 0.5 * (x + y),
    "multiplication": lambda x, y: x * y,
    "sqrt-multiplication": lambda x, y: np.sqrt(x) * np.sqrt(y),
    "sqr-multiplication": lambda x, y: (x ** 2) * (y ** 2),
    "shunting": lambda x, y: (1 + x) / (2 + 2 * y),
    "norm": lambda x, y: np.sqrt(x * x + y * y) / np.sqrt(2.0),
    "arctan": lambda x, y: 2.0 * np.arctan2(y, x) / np.pi,
    "max": lambda x, y: np.max((x, y), axis=0),
}

N_BENCHMARK_FUNCTIONS = len(BENCHMARK_FUNCTIONS)

BENCHMARK_FUNCTIONS_KEYS = list(BENCHMARK_FUNCTIONS.keys())

#
# 2D Function Parameters
#

# Image resolution (test set is res * res)
RES = 63

# Make sure the image resolution is odd
RES = (2 * (RES // 2) + 1)

# Number of training samples (training set)
N_SAMPLES = 256  # should be significantly smaller than RES * RES

#
# Neuron population parameters
#

# Number of neurons per dimensions
N_NEURONS = 100

# Maximum rate (lower and upper bound)
MAX_RATES = (50, 100)

# Probability of a neuron to be inhibitory
PINH = 0.3

TAU_SYN_I = 10e-3

TAU_SYN_E = 5e-3

# Scaling factor for the random current functions
J_SCALE = 0.5e-9

#
# Sweep parameters
#

# Noise added to Apre when evaluating
NOISE_A_PRE = 1e-2  # 1% of the maximum rate (relative to the maximum rate upper bound)

# Noise trials, i.e., how many times the function should be sampled for
# different noisy pre-activities
N_NOISE_TRIALS = 1

# Number of steps on the frequency axis
N_SIGMAS = 60

# Number of repetitions
N_REPEAT = 100

# Number of regularisation factors to try
N_REGS = 32

# Sigmas to use
SIGMAS = np.logspace(np.log10(0.075), 1, N_SIGMAS)[::-1]

# Sigma used when computing the regularisation factor
SIGMA_REG_EST = 2.0

# Regularisation factors
REGS = np.logspace(-3, 3, N_REGS)

# Regularisation factors to use in the filter/regularisation network sweep
REGS_FLT_SWEEP1 = np.logspace(-1, 3, N_REGS)
REGS_FLT_SWEEP2 = np.logspace(-3, 0, N_REGS)

REGS_FLT_SWEEP_MAP = {
    ("linear", True): REGS_FLT_SWEEP1,
    ("linear_2d", True): REGS_FLT_SWEEP1,
    ("linear", False): REGS_FLT_SWEEP1,
    ("linear_2d", False): REGS_FLT_SWEEP1,

    ("gc50_no_noise", True): REGS_FLT_SWEEP2,
    ("gc100_no_noise", True): REGS_FLT_SWEEP2,
    ("gc200_no_noise", True): REGS_FLT_SWEEP2,
    ("gc50_no_noise", False): REGS_FLT_SWEEP2,
    ("gc100_no_noise", False): REGS_FLT_SWEEP2,
    ("gc200_no_noise", False): REGS_FLT_SWEEP2,

    ("gc50_noisy", True): REGS_FLT_SWEEP2,
    ("gc100_noisy", True): REGS_FLT_SWEEP1,
    ("gc200_noisy", True): REGS_FLT_SWEEP1,
    ("gc50_noisy", False): REGS_FLT_SWEEP2,
    ("gc100_noisy", False): REGS_FLT_SWEEP2,
    ("gc200_noisy", False): REGS_FLT_SWEEP1,
}

# Number of pre-filters to try in the pre-filter sweep experiment
N_TAU_PRE_FILTS = 33

# Pre-filters to sweep over
TAU_PRE_FILTS = np.logspace(-3, -1, N_TAU_PRE_FILTS)

#
# Network simulation parameters
#

# Simulation time
SIM_T = 10.0

# Simulation timestep
SIM_DT = 1e-4

# Simulation subsampling
SIM_SS = 10

#
# Solver parameters
#

# Number of iterations in the weight solver
MAX_ITER = 1000

# Default solver tolerance
TOL = 1e-6

# Default iTh
ITH = 0.75

# Regularisation factor used to compute the identity decoders
DECODER_REG = 1e-2

#
# Misc Parameters
#

import multiprocessing, os

hostname = os.uname()[1]
if ("ctngpu" in hostname):
    # Full throttle when running this on a compute server
    N_CPUS = multiprocessing.cpu_count()
else:
    # Don't overwhelm my workstation
    N_CPUS = multiprocessing.cpu_count() // 2

# When running a single experiment from Jupyter, use all threads for solving
def in_notebook():
    # See https://stackoverflow.com/a/22424821
    try:
        from IPython import get_ipython
        ipy = get_ipython()
        if (not ipy) or ('IPKernelApp' not in ipy.config):
            return False
    except ImportError:
        return False
    return True

if in_notebook():
    N_SOLVER_THREADS = multiprocessing.cpu_count() // 2
else:
    if hasattr(multiprocessing, "current_process") and (multiprocessing.current_process().name == "MainProcess"):
        N_SOLVER_THREADS = multiprocessing.cpu_count() // 2
    else:
        N_SOLVER_THREADS = 1 # We're already parallelising over experiments
