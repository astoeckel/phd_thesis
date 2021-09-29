#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.join('..', '..', 'lib'))

import nlif.solver
import nlif_sim_2d_fun_network as netsim
import nlif_parameters as params

params.SOLVER[0] = nlif.solver.Solver(debug=True)

print(params.N_SOLVER_THREADS)

rng = np.random.RandomState(45891)
res = netsim.run_single_spiking_trial('three_comp',
                                      lambda x, y: 0.5 * (x + y),
                                      intermediate=True,
                                      pinh=None,
                                      n_epochs=30,
                                      rng=rng)

print(res["errors"])
