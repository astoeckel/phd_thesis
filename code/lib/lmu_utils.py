#  Adaptive Filter Benchmark
#  Copyright (C) 2020 Andreas Stöckel
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import copy
import dataclasses
import numpy as np
import scipy.signal
import scipy.linalg
import tqdm
import dlop_ldn_function_bases as bases

def mk_ldn_basis(q, N, Nmul=1):
    A, B = bases.mk_ldn_lti(q)
    return bases.mk_lti_basis(A, B, N, N * Nmul)


def mk_mod_fourier_basis(q, N, Nmul=1, fac=0.9, Ninternal=10000):
    def mk_fourier_oscillator(q, mul=1.0):
        B = (np.arange(0, q) + 1) % 2
        A = np.zeros((q, q))
        for k in range(1, q):
            ki = (k + 1) // 2
            fk = 2.0 * np.pi * mul * ki
            A[2 * ki - 1, 2 * ki - 1] = 0
            A[2 * ki - 1, 2 * ki + 0] =  fk
            A[2 * ki + 0, 2 * ki - 1] = -fk
            A[2 * ki + 0, 2 * ki + 0] = 0
        return A, B

    assert q % 2 == 1

    A, B = mk_fourier_oscillator(q, mul=0.9)
    Ad, Bd = np.zeros((q, q)), np.zeros((q,))

    Ad[1:, 1:], Bd[1:] = bases.discretize_lti(1.0 / Ninternal, A[1:, 1:], B[1:])
    Bd[0] = 1e-3
    Ad[0, 0] = 1.0

    H = bases.mk_lti_basis(Ad, Bd, Ninternal, from_discrete_lti=True, normalize=False)
    enc = H[:, 0]
    dec = np.linalg.pinv(H, rcond=1e-2)[0]

    Ad = Ad - np.outer(enc, dec) @ Ad
    Bd = Bd - np.outer(enc, dec) @ Bd

    A = np.real(scipy.linalg.logm(Ad)) * Ninternal

    return bases.mk_lti_basis(A, B, N, N * Nmul)


def mackey_glass(N_smpls,
                 tau=17.0,
                 dt=1e-3,
                 a=0.2,
                 b=0.1,
                 x0=1.2,
                 σ0=1.0,
                 rng=np.random):
    def df(x, xd):  # xd is the delayed x
        return a * xd / (1.0 + xd**10) - b * x

    # Compute the delay in samples
    N_delay = int(tau / dt)
    N_total = N_smpls + N_delay

    # Generate the initial state with some gaussian noise
    xs = np.zeros(N_total)
    if σ0 <= 0.0:
        xs[:N_delay] = x0
    else:
        xs[:N_delay] = rng.normal(x0, σ0, N_delay)

    # Integrate the differential equation using Runge-Kutta
    for i in range(N_delay, N_total):
        # We need to compute the delayed x at three time-points for
        # Runge-Kutta: t - tau, t - tau + 0.5 * dt, t - tau + dt. We estimate
        # the centre value linearly.
        xd0 = xs[i - N_delay]
        xd2 = xs[i - N_delay + 1]
        xd1 = 0.5 * (xd0 + xd2)

        # Evaluate the differential at the Runge-Kutta sample points
        k1 = df(xs[i - 1], xd0)
        k2 = df(xs[i - 1] + 0.5 * dt * k1, xd1)
        k3 = df(xs[i - 1] + 0.5 * dt * k2, xd1)
        k4 = df(xs[i - 1] + dt * k3, xd2)

        # Perform the Runge-Kutta step
        xs[i] = xs[i - 1] + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    return xs[N_delay:]


def mk_mackey_glass_dataset(N_wnds,
                            dt=1.0,
                            T=10000,
                            tau=30.0,
                            N_pred=15,
                            N_smpls=40000,
                            N_test=10000,
                            N_grp=100,
                            N_batch=100,
                            seed=4718,
                            σ0=1.0,
                            verbose=False):
    import tensorflow as tf
    import os
    import json
    import hashlib
    import random
    import string

    # Generate a hash describing the dataset
    dset_str = json.dumps({
        "N_wnds": N_wnds,
        "dt": dt,
        "T": T,
        "tau": tau,
        "N_pred": N_pred,
        "N_smpls": N_smpls,
        "N_test": N_test,
        "N_grp": N_grp,
        "N_batch": N_batch,
        "seed": seed,
        "σ0": σ0,
    }).encode('utf-8')
    hash = hashlib.sha224(dset_str).hexdigest()[0:16]
    fn = '/tmp/mcgl_' + hash + '.npz'
    tmp_suffix = ''.join(
        (random.choice(string.ascii_lowercase) for x in range(16)))
    fn_tmp = '/tmp/mcgl_' + hash + '_' + tmp_suffix + '.npz'
    if os.path.isfile(fn):
        data = np.load(fn)
        smpls_x_train, smpls_t_train = data["smpls_x_train"], data[
            "smpls_t_train"]
        smpls_x_val, smpls_t_val = data["smpls_x_val"], data["smpls_t_val"]
        smpls_x_test, smpls_t_test = data["smpls_x_test"], data["smpls_t_test"]
    else:
        # Generate the random number generator
        rng = np.random.RandomState(57503 + 15173 * seed)

        # Number of samples per MG dataset
        N_ts = int(T / dt)

        # Create the arrays holding the individual samples
        N_wnd = sum(N_wnds) - len(N_wnds) + 1
        smpls_x_train = np.zeros((N_smpls, N_wnd))
        smpls_t_train = np.zeros((N_smpls, N_pred))
        smpls_x_val, smpls_x_test = np.zeros((2, N_test, N_wnd))
        smpls_t_val, smpls_t_test = np.zeros((2, N_test, N_pred))

        # Generate the training data
        pbar = tqdm.tqdm if verbose else lambda x: x
        N_smpls -= N_smpls % N_grp
        for i in pbar(range(N_smpls // N_grp)):
            # Create three new Mackey-Glass datasets
            xs_train = mackey_glass(N_ts, tau=tau, dt=dt, rng=rng, σ0=σ0)

            # Extract random segments from the generated signal
            for j in range(N_grp):
                k = i * N_grp + j
                i0 = rng.randint(int(tau / dt) + 1, N_ts - N_wnd - N_pred)
                i1, i2 = i0 + N_wnd, i0 + N_wnd + N_pred
                smpls_x_train[k], smpls_t_train[k] = xs_train[i0:i1], xs_train[
                    i1:i2]

        # Independently generate the test data
        N_test -= N_test % N_grp
        for i in pbar(range(N_test // N_grp)):
            # Create two new Mackey-Glass datasets
            xs_val = mackey_glass(N_ts, tau=tau, dt=dt, rng=rng, σ0=σ0)
            xs_test = mackey_glass(N_ts, tau=tau, dt=dt, rng=rng, σ0=σ0)

            # Extract random segments from the generated signal
            for j in range(N_grp):
                k = i * N_grp + j
                i0 = rng.randint(int(tau / dt) + 1, N_ts - N_wnd - N_pred)
                i1, i2 = i0 + N_wnd, i0 + N_wnd + N_pred
                smpls_x_val[k], smpls_t_val[k] = xs_val[i0:i1], xs_val[i1:i2]
                smpls_x_test[k], smpls_t_test[k] = xs_test[i0:i1], xs_test[
                    i1:i2]

        # Save the generated data to a temporary file
        np.savez(
            fn_tmp, **{
                "smpls_x_train": smpls_x_train,
                "smpls_t_train": smpls_t_train,
                "smpls_x_val": smpls_x_val,
                "smpls_t_val": smpls_t_val,
                "smpls_x_test": smpls_x_test,
                "smpls_t_test": smpls_t_test,
            })

        # Publish that file under the global name
        try:
            os.rename(fn_tmp, fn)
        except OSError:
            os.unlink(fn_tmp)

    ds_train = tf.data.Dataset.from_tensor_slices(
        (smpls_x_train, smpls_t_train))
    ds_train = ds_train.shuffle(N_smpls)
    ds_train = ds_train.batch(N_batch)

    ds_val = tf.data.Dataset.from_tensor_slices((smpls_x_val, smpls_t_val))
    ds_val = ds_val.batch(N_batch)

    ds_test = tf.data.Dataset.from_tensor_slices((smpls_x_test, smpls_t_test))
    ds_test = ds_test.batch(N_batch)

    return ds_train, ds_val, ds_test


def read_idxgz(filename):
    import gzip
    with gzip.open(filename, mode="rb") as f:
        # Read the header
        z0, z1, dtype, ndim = f.read(4)
        assert z0 == 0 and z1 == 0 and dtype == 0x08 and ndim > 0

        dims = []
        for i in range(ndim):
            nit0, nit1, nit2, nit3 = f.read(4)
            dims.append(nit3 | (nit2 << 8) | (nit1 << 16) | (nit0 << 24))

        # Read the remaining data
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(*dims)


def nts(T, dt=1e-3):
    return int(T / dt + 1e-9)


def mkrng(rng=np.random):
    """
    Derives a new random number generator from the given random number
    generator.
    """
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
                 rng=np.random,
                 rms=0.5):
        assert (not freq_low is None) or (not freq_high is None)

        # Copy the given parameters
        self.n_dim = n_dim
        self.dt = dt
        self.rms = rms

        # Derive a new random number generator from the given rng. This ensures
        # that the signal will always be the same for a given random state,
        # independent of other
        self.rng = mkrng(rng)

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
        xs = self.rng.randn(n_smpls, self.n_dim)

        # Filter each dimension independently, save the final state so multiple
        # calls to this function will create a seamless signal
        ys = np.empty((n_smpls, self.n_dim))
        for i in range(self.n_dim):
            ys[:, i], self.zi[:, i] = scipy.signal.lfilter(self.b,
                                                           self.a,
                                                           xs[:, i],
                                                           zi=self.zi[:, i])
        return ys


@dataclasses.dataclass
class EnvironmentDescriptor:
    """
    Dataclass containing run-time information about an instanciated environment.
    """
    n_state_dim: int = 1
    n_observation_dim: int = 1
    n_control_dim: int = 0


class Environment:
    def do_init(self):
        raise NotImplemented("do_init not implemented")

    def do_step(self, n_smpls):
        raise NotImplemented("do_step not implemented")

    def __init__(self, dt=1e-3, rng=np.random, *args, **kwargs):
        """
        Initializes the environment.

        dt: is the timestep.
        rng: is the random number generator.
        """
        assert dt > 0.0

        # Copy the given arguments
        self._dt = dt
        self._rng = mkrng(rng)

        self._descr = self.do_init(*args, **kwargs)

    def step(self, n_smpls):
        """
        Executes the environment for the specified number of samples. Returns
        the n_smpls x n_state_dim matrix containing the state, a
        n_smpls x n_observation_dim matrix of observations, and a n_smpls x
        n_control_dim matrix of control dimensions.
        """

        # Make sure the number of samples is non-negative
        assert int(n_smpls) >= 0

        # Call the actual implementation of step and destructure the return
        # value
        xs, zs, us = self.do_step(int(n_smpls))

        # Make sure the resulting arrays have the right dimensionality
        xs, zs, us = np.asarray(xs), np.asarray(zs), np.asarray(us)
        if xs.ndim != 2:
            xs = xs.reshape(n_smpls, -1)
        if zs.ndim != 2:
            zs = zs.reshape(n_smpls, -1)
        if us.ndim != 2:
            us = us.reshape(n_smpls, -1)

        # Make sure the returned arrays have the right shape
        assert xs.shape == (n_smpls, self.n_state_dim)
        assert zs.shape == (n_smpls, self.n_observation_dim)
        assert us.shape == (n_smpls, self.n_control_dim)

        return xs, zs, us

    def clone(self):
        """
        Produces a copy of this Environment instance that will behave exactly
        as this one, but is decoupled from this instance.
        """
        return copy.deepcopy(self)

    @property
    def dt(self):
        return self._dt

    @property
    def rng(self):
        return self._rng

    @property
    def descr(self):
        return self._descr

    @property
    def n_state_dim(self):
        return self._descr.n_state_dim

    @property
    def n_observation_dim(self):
        return self._descr.n_observation_dim

    @property
    def n_control_dim(self):
        return self._descr.n_control_dim


class EnvironmentWithSignalBase(Environment):
    def do_init(self, signal_kwargs=None):
        # Assemble the parameters that are being passed to the 1D signal
        # generator
        if signal_kwargs is None:
            signal_kwargs = {}
        if not "freq_high" in signal_kwargs:
            signal_kwargs["freq_high"] = 0.1
        if not "rms" in signal_kwargs:
            signal_kwargs["rms"] = 1.0

        # Initialize the filtered signal instance
        self.signal = FilteredGaussianSignal(n_dim=1,
                                             dt=self.dt,
                                             rng=self.rng,
                                             **signal_kwargs)


class FrequencyModulatedSine(EnvironmentWithSignalBase):
    def do_init(self,
                f0=0.0,
                f1=2.0,
                signal_kwargs=None,
                use_control_dim=True):
        # Call the inherited constructor
        super().do_init(signal_kwargs=signal_kwargs)

        # Copy the given arguments
        self.f0 = f0
        self.f1 = f1
        self.use_control_dim = use_control_dim

        # Initialize the current state
        self.phi = 0.0

        return EnvironmentDescriptor(1, 1, 1 if use_control_dim else 0)

    def do_step(self, n_smpls):
        # Compute the frequencies
        us = 0.5 * self.signal(n_smpls)
        fs = (self.f1 - self.f0) * 0.5 * (us + 1.0) + self.f0

        # Integrate the frequencies to obtain the phases
        phis = self.phi + np.cumsum(fs) * (2.0 * np.pi * self.dt)
        xs = phis % (2.0 * np.pi)
        self.phi = xs[-1]

        # Compute the observation
        zs = np.sin(xs)

        # Do not return the control data if "use_control_dim" is set to false
        if not self.use_control_dim:
            us = np.zeros((n_smpls, 0))

        return xs, zs, us

