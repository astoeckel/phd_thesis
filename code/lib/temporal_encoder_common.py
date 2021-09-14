import numpy as np
import scipy.signal
import scipy.linalg
import tqdm

class Filters:
    @staticmethod
    def lti(A, B, idx=0):
        def mk_lti(ts, dt):
            xs = np.zeros_like(ts)
            for i, t in enumerate(ts):
                xs[i] = (scipy.linalg.expm(A * t) @ B)[idx]
            return xs

        return mk_lti

    @staticmethod
    def lowpass(tau, order=0):
        def mk_lowpass(ts, dt):
            xs = (ts >= 0.0) * (np.power(ts, order)) * (np.exp(-ts / tau))
            xs /= (np.sum(xs) * dt)
            return xs

        return mk_lowpass

    @staticmethod
    def lowpass_laplace(tau, order=0):
        denom = np.polynomial.Polynomial([tau, 1.0])**(order + 1)
        return ((1.0,), tuple(denom.coef))

    @staticmethod
    def dirac(t=0.0):
        def mk_dirac(ts, dt):
            xs = np.zeros_like(ts)
            xs[np.argmin(np.abs(ts - t))] = 1.0 / dt
            return xs

        return mk_dirac

    @staticmethod
    def step(t=0.0):
        def mk_step(ts, dt):
            xs = np.zeros_like(ts)
            xs[ts >= t] = 1.0
            return xs

        return mk_step


def mk_sig(N, dt, tau=0.1, order=1, rng=np.random):
    ts = np.arange(N) * dt
    xs = rng.normal(0, 1, N)
    xs = scipy.signal.fftconvolve(xs,
                                  Filters.lowpass(tau, order)(ts, dt))[:N] * dt
    xs /= np.max(np.abs(xs))
    return xs


def solve_for_linear_dynamics(synaptic_filters,
                              pre_tuning,
                              target_tuning,
                              N_smpls=1000,
                              xs_tau=0.1,
                              xs_order=1,
                              sigma=None,
                              dt=1e-3,
                              T=10.0,
                              rcond=None,
                              rng=np.random,
                              silent=False):
    assert len(synaptic_filters) == len(pre_tuning)
    q_pre = len(pre_tuning)
    q_post = len(target_tuning)

    # Fetch the discretised filters
    N = int(T / dt + 1e-9)
    ts = np.arange(N) * dt

    flt_syn = np.array(
        [synaptic_filters[i](ts, dt) for i in range(q_pre)])
    flt_pre = np.array([pre_tuning[i](ts, dt) for i in range(q_pre)])
    flt_tar = np.array(
        [target_tuning[i](ts, dt) for i in range(q_post)])

    A = np.zeros((N_smpls, q_pre))
    B = np.zeros((N_smpls, q_post))

    for i_smpl in tqdm.tqdm(range(N_smpls), disable=silent):
        # Generate some input signal
        xs = mk_sig(N, dt, tau=xs_tau, order=xs_order, rng=rng)
        for i_pre in range(q_pre):
            noise = np.zeros(N) if sigma is None else rng.normal(0, sigma, N)
            xs_syn = scipy.signal.fftconvolve(xs + noise, flt_syn[i_pre],
                                              'full')[:N] * dt
            xs_flt_pre = np.convolve(xs_syn, flt_pre[i_pre], 'valid') * dt
            A[i_smpl, i_pre] = xs_flt_pre

        for i_post in range(q_post):
            xs_flt_tar = np.convolve(xs, flt_tar[i_post], 'valid') * dt
            B[i_smpl, i_post] = xs_flt_tar

    W = np.linalg.lstsq(A, B, rcond=rcond)[0]

    return W


def simulate_dynamics(flt_rec,
                      flt_pre,
                      A,
                      B,
                      input_type="step",
                      T=2.0,
                      dt=1e-3,
                      high=10.0,
                      silent=False):
    import nengo

    def LP(*args):
        return nengo.LinearFilter(*Filters.lowpass_laplace(*args), analog=True)

    A, B = np.atleast_3d(A), np.atleast_2d(B)

    assert A.ndim == 3
    assert B.ndim == 2
    assert A.shape[0] == len(flt_rec)
    assert B.shape[0] == len(flt_pre)
    assert A.shape[1] == A.shape[2] == B.shape[1]

    q = A.shape[1]

    # Build the network!
    with nengo.Network() as model:
        # Create the input node
        if input_type == "step":
            x = nengo.Node(lambda t: 1.0 * (t >= 0.5) * (t < 1.5))
        elif input_type == "noise":
            x = nengo.Node(nengo.processes.WhiteSignal(period=T, high=high))
        else:
            raise RuntimeError(
                "Unknown input type, must be \"step\" or \"noise\".")

        # Create the "Neural Ensemble"
        y = nengo.Ensemble(n_neurons=1,
                           dimensions=A.shape[1],
                           neuron_type=nengo.Direct())

        # Manually wire up each connection
        for i in range(len(flt_pre)):
            for j in range(q):
                nengo.Connection(x,
                                 y[j],
                                 transform=B[i, j],
                                 synapse=LP(*flt_pre[i]))

        for i in range(len(flt_rec)):
            for j in range(q):
                for k in range(q):
                    nengo.Connection(y[j],
                                     y[k],
                                     transform=A[i, j, k],
                                     synapse=LP(*flt_rec[i]))

        # Probe the state of both
        p_x = nengo.Probe(x, synapse=None)
        p_y = nengo.Probe(y, synapse=None)

    # Simulate the network!
    with nengo.Simulator(model, dt=dt, progress_bar=not silent) as sim:
        sim.run(T)

    return sim.trange(), sim.data[p_x], sim.data[p_y]

