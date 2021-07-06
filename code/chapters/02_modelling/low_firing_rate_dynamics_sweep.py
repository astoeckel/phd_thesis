#!/usr/bin/env python3

import h5py
import json
import nengo
import numpy as np
import random
import tqdm
import multiprocessing
import os

dt = 1e-3
T = 100.0
bandwidth = 100.0

neuron_types = [
    nengo.SpikingRectifiedLinear,
    nengo.LIF,
    nengo.AdaptiveLIF,
]


class EnvGuard:
    """
    Class used to temporarily set some environment variables to a certain value.
    In particular, this is used to set the variable OMP_NUM_THREADS to "1" for
    each of the subprocesses using cvxopt.
    """
    def __init__(self, env):
        self.env = env
        self.old_env_stack = []

    def __enter__(self):
        # Create a backup of the environment variables and write the desired
        # value.
        old_env = {}
        for key, value in self.env.items():
            if key in os.environ:
                old_env[key] = os.environ[key]
            else:
                old_env[key] = None
            os.environ[key] = str(value)

        # Push the old_env object onto the stack of old_env objects.
        self.old_env_stack.append(old_env)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Get the environment variable backup from the stack.
        old_env = self.old_env_stack.pop()

        # Either delete environment variables that were not originally present
        # or reset them to their original value.
        for key, value in old_env.items():
            if not value is None:
                os.environ[key] = value
            else:
                del os.environ[key]

        return False


def compute_transfer_function(us, xs, dt=dt, smoothing=100.0):
    Us = np.fft.fftshift(np.fft.fft(us))
    Xs = np.fft.fftshift(np.fft.fft(xs))
    fs = np.fft.fftshift(np.fft.fftfreq(len(us), dt))

    wnd = np.exp(-np.square(fs) / np.square(smoothing))
    wnd /= np.sum(wnd) * dt

    UXs = np.fft.fftshift(np.fft.fft(Us * np.conj(Xs)))
    XXs = np.fft.fftshift(np.fft.fft(Xs * np.conj(Xs)))

    As = np.fft.ifft(np.fft.ifftshift(UXs * wnd))
    Bs = np.fft.ifft(np.fft.ifftshift(XXs * wnd))

    return fs, As / Bs


def run_single_experiment(args):
    # Unpack the arguments
    i, j, k, rate = args
    seed = 58219 * i + 213
    neuron_type = neuron_types[j]

    n_neurons = int(100000 / rate)  # 10 Hz ==> 10,000; 100 Hz ==> 1000

    with nengo.Network(seed=seed) as model:
        ens_us = nengo.Node(
            nengo.processes.WhiteSignal(period=T,
                                        high=bandwidth,
                                        rms=0.5,
                                        seed=seed))

        ens_xs = nengo.Ensemble(n_neurons=n_neurons,
                                neuron_type=neuron_type(),
                                dimensions=1,
                                max_rates=nengo.dists.Uniform(
                                    0.5 * rate, rate),
                                seed=seed)

        nengo.Connection(ens_us, ens_xs, synapse=None)

        p_us = nengo.Probe(ens_us, synapse=None)
        p_xs = nengo.Probe(ens_xs, synapse=None)

    with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
        sim.run(T)

    ts = sim.trange()
    us = sim.data[p_us][:, 0]
    xs = sim.data[p_xs][:, 0]

    fs, Hs = compute_transfer_function(us, xs, dt)
    i0 = np.min(np.where(fs >= -bandwidth)[0])
    i1 = np.max(np.where(fs <= bandwidth)[0])

    return i, j, k, Hs[i0:i1]


def main():
    # Compute the number of samples to expect
    N_smpls = int((T + 1e-9) / dt)
    fs = np.fft.fftshift(np.fft.fftfreq(N_smpls, dt))
    i0 = np.min(np.where(fs >= -bandwidth)[0])
    i1 = np.max(np.where(fs <= bandwidth)[0])
    N_fs = i1 - i0

    # Compute the sweep to perform
    N_repeat = 50
    N_neuron_types = len(neuron_types)
    N_rates = 100
    rates = np.logspace(np.log10(10), np.log10(200), N_rates)
    params = [(i, j, k, rate) for i in range(N_repeat)
              for j in range(N_neuron_types) for k, rate in enumerate(rates)]
    random.shuffle(params)

    fn = os.path.join("data", "low_firing_rates_dynamics_sweep.h5")
    with h5py.File(fn, "w") as f:
        f.create_dataset("fs", data=fs[i0:i1])
        f.create_dataset("rates", data=rates)

        dset_res = f.create_dataset("results",
                                    (N_repeat, N_neuron_types, N_rates, N_fs),
                                    dtype=np.complex128)

        with EnvGuard({
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
        }) as guard:
            with multiprocessing.get_context("spawn").Pool() as pool:
                for i, j, k, res in tqdm.tqdm(pool.imap_unordered(
                        run_single_experiment, params),
                                              total=len(params)):
                    dset_res[i, j, k] = res


if __name__ == "__main__":
    main()

