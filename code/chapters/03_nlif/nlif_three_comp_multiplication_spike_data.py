#!/usr/bin/env python3

import numpy as np
import pickle
import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..', 'lib'))

import nlif_sim_2d_fun_network as netsim

rng = np.random.RandomState(41268)

res = netsim.run_single_spiking_trial('three_comp',
                                      lambda x, y: x * y,
                                      intercepts_tar=(-0.95, 0.0),
                                      intermediate=False,
                                      pinh=None,
                                      N_epochs=30,
                                      rng=rng,
                                      reg=1e-6,
                                      compute_pre_spike_times=True,
                                      compute_post_spike_times=True)

def make_primitive(o):
    if isinstance(o, dict):
        res = {}
        for key, value in o.items():
            res[key] = make_primitive(value)
        return res
    elif isinstance(o, (list, tuple)):
        return [make_primitive(x) for x in o]
    elif isinstance(o, (np.ndarray, str, int, float)):
        return o
    else:
        return None

res = make_primitive(res)

fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
                  'nlif_three_comp_multiplication_spike_data.pkl')
with open(fn, "wb") as f:
    pickle.dump(res, f)

