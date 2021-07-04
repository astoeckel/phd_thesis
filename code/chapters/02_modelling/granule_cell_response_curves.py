#!/usr/bin/env python3

import argparse
import nengo_bio as bio
import numpy as np
import multiprocessing
import scipy.stats
import scipy.optimize
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument('--tar',
                    type=str,
                    help='Target filename in the data directory',
                    default='cerebellum_granule_tuning_curves.npy')

args = parser.parse_args()

def compute_response_curve(constructor, J_min=-20e-12, J_max=100e-12, max_rate=100, dJ=1e-12, dt=1e-3, T=10.0):
    Js, Gs = [], []
    J, rate = J_min, 0

    # Repeat until either the maximum current or the maximum rate is reached
    while (J < J_max) and (rate < max_rate):
        # Compile the neuron simulator
        sim = constructor().compile(dt, 1)
        spike_out = np.zeros(1)
        n_spikes = 0
        for t in np.arange(0, T, dt):
            J_cur = J + np.random.normal(0, 1e-11)
            J_in_exc = np.ones(1) * np.clip(J_cur, 0, None)
            J_in_inh = np.ones(1) * np.clip(-J_cur, 0, None)
            sim(spike_out, J_in_exc, J_in_inh)
            if spike_out > 0:
                n_spikes = n_spikes + 1

        # Record the J, G[J] tuple
        rate = n_spikes / T
        Js.append(J)
        Gs.append(rate)

        # Increase the current
        J += dJ

    return np.array(Js), np.array(Gs)

def _compute_response_curve_single(p):
    i, E_rev_leak = p
    return i, compute_response_curve(lambda: bio.neurons.LIF(
        C_som=5e-12,
        g_leak_som=3.5e-10,
        v_th=-35e-3,
        tau_ref=1e-3,
        tau_spike=0e-3,
        E_rev_leak=E_rev_leak,
        v_reset=E_rev_leak,
    ))

# Chadderton, 2004, Figure 1b
orig_bins = np.linspace(-90,-30, 16)
orig_qty = np.array([0, 2, 4, 4, 6, 15, 12, 12, 8, 3, 5, 1, 3, 0, 0])
orig_qty_density = orig_qty / (np.sum(orig_qty) * 4)

# Fit a Gaussian to the resting potential histogram
p_mu, p_sigma = scipy.optimize.curve_fit(scipy.stats.norm.pdf, orig_bins[:-1] + 2, orig_qty_density, p0=(-64, 10))[0]
ps = np.linspace(-90, -30, 100)
ps_density = scipy.stats.norm.pdf(ps, p_mu, p_sigma)


# Number of neurons to simulate
n_E_rev_leak = 128
Js_lst = [None] * n_E_rev_leak
Gs_lst = [None] * n_E_rev_leak
E_rev_leaks = np.concatenate(([-64e-3], np.random.normal(p_mu * 1e-3, p_sigma * 1e-3, n_E_rev_leak - 1)))
with multiprocessing.Pool(16) as pool:
    for i, (Js, Gs) in tqdm(pool.imap_unordered(
            _compute_response_curve_single,
            enumerate(E_rev_leaks)), total=n_E_rev_leak):
        Js_lst[i] = Js
        Gs_lst[i] = Gs

fn = os.path.join(os.path.dirname(__file__), '..', '..', 'data', args.tar)
np.save(fn, {
    "Js_lst": Js_lst,
    "Gs_lst": Gs_lst,
     "E_rev_leaks": E_rev_leaks,
})

