#!/usr/bin/env python3

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))

import numpy as np
import pykinsim as pks
import nengo
import scipy.signal
import h5py
import tqdm

DT = 1e-3
T = 1000.0
NOISE_FREQ_HIGH = 0.2
NOISE_RMS = 1.5
ETA = 0.2e-4


def nts(T, dt=1e-3):
    return int(T / dt + 1e-9)


def mkrng(rng):
    return np.random.RandomState(rng.randint(1 << 31))


class FilteredGaussianSignal:
    """
    The FilteredGaussianSignal class generates a low-pass filtered white noise
    signal.
    """
    def __init__(self,
                 n_dim=1,
                 freq_low=None,
                 freq_high=1.0,
                 order=4,
                 dt=1e-3,
                 rng=None,
                 rms=1.0):
        assert (not freq_low is None) or (not freq_high is None)

        # Copy the given parameters
        self.n_dim = n_dim
        self.dt = dt
        self.rms = rms

        # Derive a new random number generator from the given rng. This ensures
        # that the signal will always be the same for a given random state,
        # independent of other
        self._rng = mkrng(rng)

        # Build the Butterworth filter
        if freq_low is None:
            btype = "lowpass"
            Wn = freq_high
        elif freq_high is None:
            btype = "highpass"
            Wn = freq_low
        else:
            btype = "bandpass"
            Wn = [freq_low, freq_high]
        self.b, self.a = scipy.signal.butter(N=order,
                                             Wn=Wn,
                                             btype=btype,
                                             analog=False,
                                             output='ba',
                                             fs=1.0 / dt)

        # Scale the output to reach the RMS
        self.b *= rms / np.sqrt(2.0 * dt * freq_high)

        # Initial state
        self.zi = np.zeros((max(len(self.a), len(self.b)) - 1, self.n_dim))

    def __call__(self, n_smpls):
        # Generate some random input
        xs = self._rng.randn(n_smpls, self.n_dim)

        # Filter each dimension independently, save the final state so multiple
        # calls to this function will create a seamless signal
        ys = np.empty((n_smpls, self.n_dim))
        for i in range(self.n_dim):
            ys[:, i], self.zi[:, i] = scipy.signal.lfilter(self.b,
                                                           self.a,
                                                           xs[:, i],
                                                           zi=self.zi[:, i])
        return ys


def generate_pendulum_dataset(phi0=np.pi / 2,
                              m=1.0,
                              L=1.0,
                              rms=NOISE_RMS,
                              freq_high=NOISE_FREQ_HIGH,
                              dt=DT,
                              T=T):
    # Setup the pykinsim model
    with pks.Chain() as chain:
        f1 = pks.Fixture()
        m1 = pks.Mass()
        j1 = pks.Joint(torque=pks.External, theta=np.pi / 2)
        pks.Link(f1, j1, l=0.0)
        pks.Link(j1, m1, l=1.0)

    # Run the simulation
    N = nts(T, dt)
    taus, phis = np.zeros((2, N))
    with pks.Simulator(chain, root=f1) as sim:
        state = sim.initial_state()
        signal = FilteredGaussianSignal(1,
                                        freq_high=freq_high,
                                        rng=np.random.RandomState(578181),
                                        rms=rms,
                                        dt=dt)
        for i in tqdm.tqdm(range(N)):
            taus[i] = tau = signal(1)[0]
            phis[i] = state[j1]
            state = sim.step(state, DT)
            state.torques[j1] = tau

            # Apply a hard limit to the transformation angles
            if state[j1] > 7 * np.pi / 8:
                state[j1] = 7 * np.pi / 8
            elif state[j1] < -3 * np.pi / 4:
                state[j1] = -3 * np.pi / 4

    ts = np.arange(N) * dt
    return ts, taus, phis


def execute_network(taus,
                    phis,
                    W_in,
                    W_rec,
                    gains,
                    biases,
                    gain_phis=1.0,
                    gain_taus=1.0,
                    delay=0.75,
                    dt=1e-3,
                    tau=100e-3,
                    tau_learn=20e-3,
                    eta=ETA):
    # Number of input samples
    N_tot = len(taus)

    # Number of samples by which to delay phis
    N_delay = int(delay / dt + 1e-9)

    # For how long the simulation should be executed
    N = N_tot - N_delay - 1

    # Number of neurons
    n_neurons = len(gains)

    # Time point at which learning is switched off
    t_learn_off = 0.9 * N * dt

    with nengo.Network() as model:
        # Two-dimensional input. Note that phis is delayed relative to taus
        nd_in = nengo.Node(lambda t: [
            gain_taus * taus[(int(t / dt) + N_delay) % N_tot],
            gain_phis * phis[(int(t / dt)) % N],
        ])

        # Target
        nd_tar = nengo.Node(lambda t: [
            phis[(int(t / dt) + N_delay) % N_tot],
        ])

        # Assemble the spatiotemporal network
        ens_x = nengo.Ensemble(n_neurons=n_neurons,
                               dimensions=1,
                               bias=biases,
                               gain=gains,
                               encoders=np.ones((n_neurons, 1)))
        nengo.Connection(nd_in,
                         ens_x.neurons,
                         transform=W_in[:, :, 0],
                         synapse=tau)
        nengo.Connection(ens_x.neurons,
                         ens_x.neurons,
                         transform=W_rec[:, :, 0],
                         synapse=tau)

        # Setup the learning connection
        nd_out = nengo.Node(size_in=1)
        conn = nengo.Connection(
            ens_x.neurons,
            nd_out,
            transform=np.zeros((1, n_neurons)),
            synapse=tau_learn,
            learning_rule_type=nengo.PES(learning_rate=eta))

        nd_err = nengo.Node(size_in=1)
        nd_err_valve = nengo.Node(lambda t, x: x * (t < t_learn_off),
                                  size_in=1)
        nengo.Connection(nd_out, nd_err, synapse=None)
        nengo.Connection(nd_tar, nd_err, transform=-1, synapse=tau_learn)
        nengo.Connection(nd_tar, nd_err_valve, synapse=None)
        nengo.Connection(nd_err_valve, conn.learning_rule, synapse=None)

        # Record everything
        p_in = nengo.Probe(nd_in, synapse=None)
        p_tar = nengo.Probe(nd_tar, synapse=None)
        p_out = nengo.Probe(nd_out, synapse=None)
        p_err = nengo.Probe(nd_err, synapse=None)

    with nengo.Simulator(model, dt=dt) as sim:
        sim.run(N * dt)

    return sim.trange(
    ), sim.data[p_in], sim.data[p_tar], sim.data[p_out], sim.data[p_err]


def main():
    fn_dataset = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                              'data', "pendulum_dataset.h5")
    fn_net = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'data', "manual",
        "chapters", "04_temporal_tuning",
        "ae3bf70a32be60b6_spatio_temporal_network_matrices.h5")
    fn_res = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
                          "pendulum_adaptive_filter.h5")

    # Fetch the pendulum data
    if not os.path.isfile(fn_dataset):
        # Run the experiment
        _, taus, phis = generate_pendulum_dataset()

        # Store the data
        with h5py.File(fn_dataset, 'w') as f:
            f.create_dataset("taus", data=taus, compression="gzip")
            f.create_dataset("phis", data=phis, compression="gzip")
    else:
        with h5py.File(fn_dataset, 'r') as f:
            taus = f["taus"][()]
            phis = f["phis"][()]

    # Normalise the data
    taus_norm = (taus - np.mean(taus)) / np.std(taus)
    phis_norm = (phis - np.mean(phis)) / np.std(phis)
    taus_norm /= np.percentile(np.abs(taus_norm), 95)
    phis_norm /= np.percentile(np.abs(phis_norm), 95)

    # Load the network
    with h5py.File(fn_net, "r") as f:
        W_rec = f["W_rec"][()]
        W_in = f["W_in"][()]
        gains = f["gains"][()]
        biases = f["biases"][()]

    # Execute the network with both taus and phis being available
    _, xs_in, xs_tar, xs_out_both, xs_err_both = execute_network(taus_norm,
                                                                 phis_norm,
                                                                 W_in=W_in,
                                                                 W_rec=W_rec,
                                                                 gains=gains,
                                                                 biases=biases)

    # Execute the network without taus being available
    _, _, _, xs_out_no_taus, xs_err_no_taus = execute_network(taus_norm,
                                                              phis_norm,
                                                              W_in=W_in,
                                                              W_rec=W_rec,
                                                              gains=gains,
                                                              biases=biases,
                                                              gain_taus=0.0)

    # Execute the network without phis being available
    _, _, _, xs_out_no_phis, xs_err_no_phis = execute_network(taus_norm,
                                                              phis_norm,
                                                              W_in=W_in,
                                                              W_rec=W_rec,
                                                              gains=gains,
                                                              biases=biases,
                                                              gain_phis=0.0)

    # Save the result
    with h5py.File(fn_res, 'w') as f:
        f.create_dataset("xs_in",
                         data=xs_in,
                         compression="gzip",
                         compression_opts=9)
        f.create_dataset("xs_tar",
                         data=xs_tar,
                         compression="gzip",
                         compression_opts=9)
        f.create_dataset("xs_out_both",
                         data=xs_out_both,
                         compression="gzip",
                         compression_opts=9)
        f.create_dataset("xs_err_both",
                         data=xs_err_both,
                         compression="gzip",
                         compression_opts=9)
        f.create_dataset("xs_out_no_taus",
                         data=xs_out_no_taus,
                         compression="gzip",
                         compression_opts=9)
        f.create_dataset("xs_err_no_taus",
                         data=xs_err_no_taus,
                         compression="gzip",
                         compression_opts=9)
        f.create_dataset("xs_out_no_phis",
                         data=xs_out_no_phis,
                         compression="gzip",
                         compression_opts=9)
        f.create_dataset("xs_err_no_phis",
                         data=xs_err_no_phis,
                         compression="gzip",
                         compression_opts=9)


if __name__ == "__main__":
    main()

