import datetime
import itertools
import tqdm
import numpy as np
import scipy.signal
import multiprocessing
import random


def nts(T, dt=1e-3):
    return int(T / dt + 1e-9)


def mkrng(rng):
    return np.random.RandomState(rng.randint(1 << 31))


def mk_low_pass_filter_basis(q, N, tau0=0.1, tau1=100.0):
    ts = np.linspace(1, 0, N)
    taus = np.geomspace(tau0, tau1, q)
    res = np.array([np.exp(-ts / tau) for tau in taus])
    res /= np.sqrt(
        np.array([np.inner(res[i], res[i]) for i in range(q)])[:, None])
    return res


def eval_lti(A, B, ts):
    import scipy.linalg
    return np.array([scipy.linalg.expm(A * t) @ B for t in ts])


def eval_lti_euler(A, B, N, dt):
    import scipy.linalg
    Ad = scipy.linalg.expm(A * dt)
    res = np.zeros((N, A.shape[0]))
    res[0, :] = B
    for i in range(1, N):
        res[i, :] = Ad @ res[i - 1, :]
    return res


def reconstruct_lti(H, T=1.0, dampen=False, return_discrete=False):
    # Fetch the number of state dimensions and the number of samples
    q, N = H.shape

    # Canonicalise "dampen"
    if dampen and isinstance(dampen, str):
        dampen = {dampen}
    elif isinstance(dampen, bool):
        dampen = {"erasure"} if dampen else set()
    if (not isinstance(dampen,
                       set)) or (len(dampen - {"lstsq", "erasure"}) > 0):
        raise RuntimeError("Invalid value for \"dampen\"")

    # Time-reverse H
    Hrev = H[:, ::-1]

    # Compute the samples
    X = Hrev[:, :-1].T
    Y = (Hrev[:, 1:].T - Hrev[:, :-1].T) * N / T

    # Estimate the discrete system At, Bt
    At = np.linalg.lstsq(X, Y, rcond=None)[0].T
    Bt = Hrev[:, 0]

    # Add the Euler update
    At = np.eye(q) + At * T / N

    if "erasure" in dampen:
        enc = H[:, 0]
        dec = np.linalg.pinv(H, rcond=1e-2)[0]
        At = At - np.outer(enc, dec) @ At
        Bt = Bt - np.outer(enc, dec) @ Bt

    if return_discrete:
        return At, Bt

    # Undo discretization (this is the inverse of discretize_lti)
    A = np.real(scipy.linalg.logm(At)) * N / T
    B = scipy.linalg.expm(-0.5 * A * T / N) @ Bt * np.sqrt(N)

    return A, B


def mk_impulse_response(basis,
                        window,
                        q=7,
                        theta=1.0,
                        T=10.0,
                        dt=1e-2,
                        N_solve=20000,
                        return_sys=False,
                        use_closed_form=True,
                        use_euler=False):
    import dlop_ldn_function_bases as bases

    def mk_mod_fourier_basis(q, N):
        N_tot = int(1.1 * N)
        return bases.mk_fourier_basis(q, int(N_tot))[:, (N_tot - N):]

    def bartlett_window(fun):
        def mk_bartlett_window_basis_fun(q, N):
            return fun(q, N) * np.linspace(0, 1, N)[None, :]

        return mk_bartlett_window_basis_fun

    # Select the underlying function
    fun = {
        "lowpass": mk_low_pass_filter_basis,
        "fourier": bases.mk_fourier_basis,
        "cosine": bases.mk_cosine_basis,
        "mod_fourier": mk_mod_fourier_basis,
        "legendre": bases.mk_dlop_basis,
    }[basis]

    # When using the Bartlett window, adapt the basis generation function itself
    if window == "bartlett":
        fun = bartlett_window(fun)

    # Sample the function
    if window == "optimal":
        N = int(theta / dt + 1e-9)
        H = fun(q, N)
    else:
        H = fun(q, N_solve)

    # Reconstruct the LTI system
    if window == "bartlett" or (not window):
        A, B = reconstruct_lti(H, theta)
    elif window == "erasure":
        if basis == "legendre" and use_closed_form:
            A, B = bases.mk_ldn_lti(q, rescale=True)
        else:
            A, B = reconstruct_lti(H, theta, dampen="erasure")

    if return_sys:
        return A, B

    # Either just copy the sampled function (optimal rectangle window) or
    # evaluate the LTI system
    ts = np.arange(0, T, dt)
    res = np.zeros((len(ts), q))
    if window == "optimal":
        res[:N] = (H / np.max(np.abs(H), axis=1)[:, None]).T
    else:
        if use_euler:
            res = eval_lti_euler(A, B, len(ts), dt)
        else:
            res = eval_lti(A, B, ts)

    return ts, res


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


# Function used to generate the input/target sample pairs
def generate_dataset(N_smpls,
                     N_sig,
                     N,
                     theta,
                     rng,
                     freq_high=5.0,
                     signal_type="lowpass"):
    N_delay = int(np.clip(np.floor(N * theta), 0, N))
    if signal_type == "lowpass":
        sig = FilteredGaussianSignal(N_smpls,
                                     dt=1.0 / N,
                                     freq_high=freq_high,
                                     rng=rng)
        xs = sig(N_sig).T
    elif signal_type == "bandlimit":
        import nengo
        xs = np.zeros((N_smpls, N_sig))
        for i in range(N_smpls):
            sig = nengo.processes.WhiteSignal(period=N_sig / N,
                                              high=freq_high,
                                              seed=rng.randint(2 << 31))
            xs[i] = sig.run(N_sig / N, dt=1.0 / N).flatten()

    ys = np.concatenate((np.zeros(
        (N_smpls, N_delay)), xs[:, :(N_sig - N_delay)]),
                        axis=1)
    return xs, ys


def generate_full_dataset(N_thetas,
                          N_test,
                          N_train,
                          N_sig,
                          N,
                          rng,
                          freq_high=5.0,
                          signal_type="lowpass"):
    # Backup the current RNG state
    state = rng.get_state()

    # Prepare the output arrays
    thetas = np.linspace(0.0, 1.0, N_thetas + 1)[:-1]
    xs_test, ys_test = np.zeros((2, N_thetas, N_test, N_sig))
    xs_train, ys_train = np.zeros((2, N_thetas, N_train, N_sig))
    for i, theta in enumerate(thetas):
        rng.set_state(state)
        xs_test[i], ys_test[i] = generate_dataset(N_test, N_sig, N, theta, rng,
                                                  freq_high, signal_type)
        xs_train[i], ys_train[i] = generate_dataset(N_train, N_sig, N, theta,
                                                    rng, freq_high,
                                                    signal_type)

    return thetas, xs_test, ys_test, xs_train, ys_train


# Function used to convolve a set of datapoints with a given temporal basis
def convolve(H, xs):
    N_smpls, N_sig = xs.shape
    q, N = H.shape
    ys_conv = np.zeros((N_smpls, N_sig, q))
    for i in range(N_smpls):
        for j in range(q):
            ys_conv[i, :, j] = scipy.signal.fftconvolve(xs[i],
                                                        H[j],
                                                        mode='full')[:N_sig]
    return ys_conv

