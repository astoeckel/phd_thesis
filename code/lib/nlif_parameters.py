import numpy as np
import nlif
import nlif.solver
import nlif.parameter_optimisation as param_opt

#
# Instantiate the solver
#

SOLVER = [nlif.solver.Solver(debug=False, parallel_compile=False)]

#
# Neuron models used in the actual simulator
#

lif_neuron = nlif.LIF(mul_E=5e-3, mul_I=-5e-3) # make the uS and nA range compatible
one_comp_lif_neuron = nlif.LIFCond()
two_comp_lif_neuron = nlif.TwoCompLIFCond()
three_comp_lif_neuron = nlif.ThreeCompLIFCond(g_c1=100e-9, g_c2=200e-9)
with nlif.Neuron() as four_comp_lif_neuron:
    with nlif.Soma() as soma:
        nlif.CondChan(E_rev=-65e-3, g=50e-9)
    with nlif.Compartment() as comp1:
        nlif.CondChan(E_rev=-65e-3, g=50e-9)
        four_comp_lif_neuron.g_E1 = nlif.CondChan(E_rev=0e-3)
        four_comp_lif_neuron.g_I1 = nlif.CondChan(E_rev=-75e-3)
    with nlif.Compartment() as comp2:
        nlif.CondChan(E_rev=-65e-3, g=50e-9)
        four_comp_lif_neuron.g_E2 = nlif.CondChan(E_rev=0e-3)
        four_comp_lif_neuron.g_I2 = nlif.CondChan(E_rev=-75e-3)
    with nlif.Compartment() as comp3:
        nlif.CondChan(E_rev=-65e-3, g=50e-9)
        four_comp_lif_neuron.g_E3 = nlif.CondChan(E_rev=0e-3)
        four_comp_lif_neuron.g_I3 = nlif.CondChan(E_rev=-75e-3)
    nlif.Connection(soma, comp1, g_c=100e-9)
    nlif.Connection(comp1, comp2, g_c=200e-9)
    nlif.Connection(comp2, comp3, g_c=500e-9)

NEURON_MODELS = {
    "lif": lif_neuron,
    "one_comp": one_comp_lif_neuron,
    "two_comp": two_comp_lif_neuron,
    "three_comp": three_comp_lif_neuron,
    "four_comp": four_comp_lif_neuron,
}

NEURON_MODELS_KEYS = list(NEURON_MODELS.keys())

NEURON_SYS_CACHE = {}

def get_neuron_sys(neuron):
    if not neuron in NEURON_SYS_CACHE:
        assm = neuron.assemble()
        sim = nlif.Simulator(nlif)
        rng = np.random.RandomState(578281)

        gs_train = rng.uniform(0, 0.5e-6, (1000, assm.n_inputs))
        As_train = assm.rate_empirical(gs_train, progress=True)
        valid_train = As_train > 12.5
        Js_train = assm.lif_rate_inv(As_train)

        sys = assm.reduced_system().condition()
        sys, _ = param_opt.optimise_trust_region(sys,
                                                 gs_train[valid_train],
                                                 Js_train[valid_train],
                                                 alpha3=1e-5,
                                                 gamma=0.99,
                                                 N_epochs=100,
                                                 progress=False,
                                                 parallel_compile=False)

        NEURON_SYS_CACHE[neuron] = (assm, sys)

    return NEURON_SYS_CACHE[neuron]


#
# Fixed 2D Functions
#

def scale(f):
    return lambda x, y: 2.0 * f(0.5 * (x + 1.0), 0.5 * (y + 1.0)) - 1.0

BENCHMARK_FUNCTIONS = {
    "addition": lambda x, y: 0.5 * (x + y),
    "multiplication_limited": scale(lambda x, y: x * y),
    "multiplication": lambda x, y: x * y,
    "sqrt-multiplication": scale(lambda x, y: np.sqrt(x) * np.sqrt(y)),
    "sqr-multiplication": lambda x, y: (x ** 2) * (y ** 2),
    "shunting": scale(lambda x, y: (1 + x) / (2 + 2 * y)),
    "norm": lambda x, y: np.sqrt(x * x + y * y) / np.sqrt(2.0),
    "arctan": lambda x, y: np.arctan2(y, x) / np.pi,
    "max": lambda x, y: np.max((x, y), axis=0),
}

#
# Neuron population parameters
#

# Use 256 samples when sampling functions
N_SAMPLES = 256

RES = 33
RES = (2 * RES + 1) // 2

# Number of neurons per dimensions
N_NEURONS = 100

# Maximum rate (lower and upper bound)
MAX_RATES = (50, 100)

# x-intercepts to use
INTERCEPTS = (-0.95, 0.95)

# Maximum rate (lower and upper bound)
MAX_RATES_TAR = (50, 100)

# x-intercepts to use
INTERCEPTS_TAR = (-0.95, 0.95)

# Probability of a neuron to be inhibitory
PINH = 0.3

TAU_SYN_I = 10e-3

TAU_SYN_E = 5e-3

# Scaling factor for the random current functions
J_SCALE = 0.5e-9

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

# Number of epochs to use
N_EPOCHS = 30

# Default solver tolerance
TOL = 1e-6

# Default iTh
ITH = 0.75

# Regularisation factor used to compute the identity decoders
DECODER_REG = 1e-2

# Regularisation factor to use in the weight solver
SOLVER_REG = 1e-3

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
