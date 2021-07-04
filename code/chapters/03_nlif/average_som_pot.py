#!/usr/bin/env python3

import sys, os

sys.path.append(os.path.join('..', 'lib'))

import numpy as np
from nef_synaptic_computation.multi_compartment_lif import *
import multiprocessing
import tqdm
import h5py
import env_guard
import random

# One-compartment, current-based LIF
LIF_cur = (Neuron().add_compartment(
    Compartment(soma=True).add_channel(CurChan(mul=1, name="j")).add_channel(
        CondChan(Erev=-65e-3, g=50e-9, name="leak"))).assemble())


def run_single_experiment(params):
    idx_i, idx_j, J, n_pre_spikes, repeat, dt, T = params
    sim = LIF_cur.simulator(dt=dt, record_voltages=True, record_in_refrac=True)

    np.random.seed(3891 + idx_i * 4)
    if n_pre_spikes is None:
        res = sim.simulate(np.ones(int(T / dt + 1e-9)) * J)
    else:
        res = sim.simulate(
            AssembledNeuron.make_noise(5e-3, n_pre_spikes, T, dt) * J)

    return idx_i, idx_j, np.mean(res.v), np.mean(
        res.v[~res.in_refrac]), np.sum(res.out * dt) / T


def run_experiment(n_pre_spikes=None, repeat=128, dt=1e-5, T=1.0, n_Js=1000):
    # Simulate the neuron in high resolution over a range of currents, record
    # the average membrane potentials
    n_pre_spikes_list = [50, 100, 500, None]
    Js = np.logspace(-10, -7, n_Js)

    params = [(idx_i, idx_j, J, n_pre_spikes, repeat, dt, T)
              for idx_j, J in enumerate(Js) for idx_i in range(repeat)]
    random.shuffle(params)
    V_som, V_som_no_ref, Rates = np.zeros((3, repeat, n_Js))
    with env_guard.SingleThreadEnvGuard():
        with multiprocessing.get_context("spawn").Pool(16) as pool:
            for idx_i, idx_j, v_som, v_som_no_ref, rates in tqdm.tqdm(
                    pool.imap_unordered(run_single_experiment, params),
                    total=n_Js*repeat):
                V_som[idx_i, idx_j] = v_som
                V_som_no_ref[idx_i, idx_j] = v_som_no_ref
                Rates[idx_i, idx_j] = rates

    return Js, V_som, V_som_no_ref, Rates



def main():
    n_pre_spikes_list = [50, 100, 500, None]
    N_PRE_SPIKES = len(n_pre_spikes_list)
    N_JS = 1000
    N_REPEAT = 128
    N_BINS = 100

    fn = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                      "average_som_pot_large.h5")
    with h5py.File(fn, "w") as f:
        DS_JS = f.create_dataset("Js", (N_JS, ))
        DS_V_SOM = f.create_dataset("v_som", (N_PRE_SPIKES, N_REPEAT, N_JS))
        DS_V_SOM_NO_REF = f.create_dataset("v_som_no_ref",
                                           (N_PRE_SPIKES, N_REPEAT, N_JS))
        DS_RATES = f.create_dataset("rates", (N_PRE_SPIKES, N_REPEAT, N_JS))

        for idx_i, n_pre_spikes in enumerate(n_pre_spikes_list):
            Js, v_som, v_som_no_ref, rates = run_experiment(n_pre_spikes,
                                                            T=1.0,
                                                            repeat=N_REPEAT,
                                                            n_Js=N_JS)

            DS_JS[...] = Js
            DS_V_SOM[idx_i] = v_som
            DS_V_SOM_NO_REF[idx_i] = v_som_no_ref
            DS_RATES[idx_i] = rates

if __name__ == "__main__":
    main()

