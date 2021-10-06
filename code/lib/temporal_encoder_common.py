import numpy as np
import scipy.signal
import scipy.linalg
import tqdm
import halton
from lstsq_cstr import lstsq_cstr
import dlop_ldn_function_bases as bases

_lit_impulse_response_cache = {}


def cached_lti_impulse_response(A, B, ts):
    # Compute the hash for the input
    from hashlib import sha1
    m = sha1()
    m.update(A)
    m.update(B)
    m.update(np.array(ts))
    digest = m.digest()

    # Compute the impulse
    if not digest in _lit_impulse_response_cache:
        _lit_impulse_response_cache[digest] = np.array(
            [scipy.linalg.expm(A * t) @ B for t in ts])
    return np.copy(_lit_impulse_response_cache[digest])


class Filters:
    @staticmethod
    def lti(A, B, idx=0):
        def mk_lti(ts, dt):
            return cached_lti_impulse_response(A, B, ts)[:, idx]

        return mk_lti

    @staticmethod
    def lti_enc(A, B, E):
        def mk_lti(ts, dt):
            return cached_lti_impulse_response(A, B, ts) @ E

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
        return ((1.0, ), tuple(denom.coef))

    @staticmethod
    def lowpass_laplace_chained(*taus):
        denom = 1.0
        for tau in taus:
            denom = denom * np.polynomial.Polynomial([tau, 1.0])
        return ((1.0, ), tuple(denom.coef))

    @staticmethod
    def lowpass_chained(*taus):
        def mk_lowpass_chained(ts, dt):
            _, xs = scipy.signal.impulse(
                Filters.lowpass_laplace_chained(*taus), T=ts)
            xs /= (np.sum(xs) * dt)
            return xs

        return mk_lowpass_chained

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


_fourier_basis_cache = {}


def mk_sig(N,
           dt,
           sigma=7.5,
           scale=1.0,
           rng=np.random,
           i_smpl=None,
           Ms=None,
           biased=False,
           bias_cstr_count=None):
    # If "biased" is set to true, bot i_smpl and M_F must be supplied
    assert (not biased) or (not ((i_smpl is None) or (Ms is None)))

    # Fetch a mapping from indices onto frequencies
    f_max = np.sqrt(-np.log(1e-6)) * sigma
    q = min(N, int(np.ceil(2.0 * f_max * N * dt)))
    if not (q, N) in _fourier_basis_cache:
        _fourier_basis_cache[(q, N)] = bases.mk_fourier_basis(q, N)
    H = _fourier_basis_cache[(q, N)]

    fs = np.zeros(q)
    fs[0] = 0.0
    for i in range(1, q):
        fs[i] = ((i + 1) // 2) / (N * dt)

    # Compute the hull for each scaling factor
    ps = np.exp(-np.square(fs) / np.square(sigma))

    # Generate the Fourier coefficients, scale the function to an RMS of 0.5
    X_F = rng.normal(0, 1, q) * ps
    xs = (H.T @ X_F)[::-1]  # This could be an FFT
    peak_min, peak_max = np.min(xs), np.max(xs)
    offs = -0.5 * (peak_max + peak_min)
    scale = scale * 2.0 / (peak_max - peak_min)
    if not biased:
        return (xs + offs) * scale
    else:
        X_F[0] += offs
        X_F *= scale

    # Select the dimensions we would like to use a constraints
    if bias_cstr_count is None:
        bias_cstr_count = min(
            50,
            Ms.shape[1])  # Our halton sequence generator only works till 50...
        dims = np.arange(0, Ms.shape[1], dtype=int)
    else:
        bias_cstr_count = min(bias_cstr_count, Ms.shape[1])
        dims = rng.choice(np.arange(0, Ms.shape[1], dtype=int),
                          bias_cstr_count,
                          replace=False)


#    Xi = halton.halton_ball(bias_cstr_count, i_smpl)
#    A = Ms[:, dims].T @ Ms[:, dims] * dt
#    alpha = np.linalg.lstsq(A, Xi, rcond=1e-2)[0]
#    return Ms[::-1, dims] @ alpha

    # Transform Ms into the target domain
    M_H = H @ Ms[:, dims]  # This could be an FFT

    # Generate a target activiation
    Xi = 1.0 * halton.halton_ball(bias_cstr_count, i_smpl)

    # Solve for weights that result in the desired tuning
    X_F = lstsq_cstr(np.diag(1.0 / ps), X_F / ps, M_H.T, Xi / dt)

    xs = (H.T @ X_F)[::-1]  # This could be an FFT

    return xs


def solve_for_linear_dynamics(synaptic_filters,
                              pre_tuning,
                              target_tuning,
                              N_smpls=1000,
                              xs_sigma=5.0,
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

    flt_syn = np.array([synaptic_filters[i](ts, dt) for i in range(q_pre)])
    flt_pre = np.array([pre_tuning[i](ts, dt) for i in range(q_pre)])
    flt_tar = np.array([target_tuning[i](ts, dt) for i in range(q_post)])

    A = np.zeros((N_smpls, q_pre))
    B = np.zeros((N_smpls, q_post))

    for i_smpl in tqdm.tqdm(range(N_smpls), disable=silent):
        # Generate some input signal
        xs = mk_sig(N, dt, sigma=xs_sigma, rng=rng)

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


def solve_for_recurrent_population_weights(G,
                                           gains,
                                           biases,
                                           A,
                                           B,
                                           TEs,
                                           flts_in,
                                           flts_rec,
                                           N_smpls=1000,
                                           xs_sigma=5.0,
                                           dt=1e-3,
                                           T=10.0,
                                           reg=0.01,
                                           rng=np.random,
                                           silent=False,
                                           Ms=None,
                                           biased=True,
                                           bias_cstr_count=None,
                                           xs_scale=1.0):
    """
    G:   Neural non-linearity
    Es:  Non-temporal encoders (determines the dimensionality of the neuron
         population.
    gains, biases: Neural gains and biases
    A: Target LTI system feedback matrix
    B: Target LTI system input matrix
    TEs: Temporal encoders
    flt_in, flt_rec: List of filters (returned by Filters.mk_*) for the
        feed-forward and recurrent connection. For the sake of simplicity,
        each neuron has access to the same filters.
    """

    assert (Ms is None) != ((A is None) or (B is None))

    # Boring bookkeeping
    TEs = np.atleast_2d(TEs)
    N_neurons = TEs.shape[0]
    gains, biases = np.atleast_1d(gains), np.atleast_1d(biases)
    assert (TEs.shape[0] == gains.shape[0] == biases.shape[0])
    if Ms is None:
        A = np.atleast_2d(A)
        B = np.atleast_1d(B).flatten()
        q = N_temporal_dimensions = A.shape[0]
        assert A.shape[0] == A.shape[1] == B.shape[0]
    else:
        Ms = np.atleast_2d(Ms)
        q = N_temporal_dimensions = Ms.shape[1]
    assert TEs.shape[1] == q

    # Evaluate all filters
    ts = np.arange(0, T, dt)
    N = len(ts)
    hs_in = np.array([flt_in(ts, dt) for flt_in in flts_in])
    hs_rec = np.array([flt_rec(ts, dt) for flt_rec in flts_rec])

    # Compute the impulse response of the desired LTI system
    if Ms is None:
        Ms = cached_lti_impulse_response(A, B, ts)
    else:
        assert Ms.shape[0] == N

    # Assemble the X and Y matrices that are being passed to the least-squares
    # algorithm
    N_pre = len(flts_in) + len(flts_rec) * N_neurons
    N_tar = N_neurons
    X = np.zeros((N_smpls, N_pre))
    Y = np.zeros((N_smpls, N_tar))

    x_in_scale = 1.0 * N_neurons  # Compensate for the input not being neural activitie
    for i_smpl in tqdm.tqdm(range(N_smpls), disable=silent):
        # Sample an input sequence; increase the magnitude to prevent regularisation problems
        xs = mk_sig(N,
                    dt,
                    sigma=xs_sigma,
                    scale=xs_scale,
                    rng=rng,
                    Ms=Ms,
                    i_smpl=i_smpl,
                    biased=biased,
                    bias_cstr_count=bias_cstr_count)

        # Filter the input sequence with the input filters and store the result
        # in the corresponding slot in X
        I = 0
        for i_flt in range(len(flts_in)):
            X[i_smpl,
              I] = np.convolve(xs, hs_in[i_flt], 'valid') * dt * x_in_scale
            I += 1

        # Filter the input sequences with the impulse response of the desired
        # LTI system
        Xs = np.zeros((N, q))
        for i in range(q):
            Xs[:, i] = scipy.signal.fftconvolve(xs, Ms[:, i], 'full')[:N] * dt

        # For each pre-neuron, compute the filtered activities
        for i_pre in range(N_neurons):
            A_pre = G(gains[i_pre] * (Xs @ TEs[i_pre]) + biases[i_pre])
            for i_flt in range(len(flts_rec)):
                X[i_smpl, I] = np.convolve(A_pre, hs_rec[i_flt], 'valid') * dt
                I += 1

        # Compute the target input current of each post-neuron
        for i_post in range(N_neurons):
            Y[i_smpl, i_post] = (Xs[-1] @ TEs[i_post])

    XTX = X.T @ X + N_smpls * np.square(reg * np.max(X)) * np.eye(N_pre)
    XTY = X.T @ Y

    W = np.linalg.solve(XTX, XTY)
    W_in = W[:len(flts_in)].T * x_in_scale
    W_rec = W[len(flts_in):].reshape(N_neurons, len(flts_rec),
                                     N_neurons).transpose(2, 0, 1)
    return W_in, W_rec


def solve_for_recurrent_population_weights_heterogeneous_filters(G,
                                           gains,
                                           biases,
                                           A,
                                           B,
                                           TEs,
                                           flts,
                                           flts_in_map,
                                           flts_rec_map,
                                           N_smpls=1000,
                                           xs_sigma=5.0,
                                           dt=1e-3,
                                           T=10.0,
                                           reg=0.01,
                                           rng=np.random,
                                           silent=False,
                                           Ms=None,
                                           biased=True,
                                           bias_cstr_count=None,
                                           xs_scale=1.0):
    """
    G:   Neural non-linearity
    Es:  Non-temporal encoders (determines the dimensionality of the neuron
         population.
    gains, biases: Neural gains and biases
    A: Target LTI system feedback matrix
    B: Target LTI system input matrix
    TEs: Temporal encoders
    flt_in, flt_rec: List of filters (returned by Filters.mk_*) for the
        feed-forward and recurrent connection. For the sake of simplicity,
        each neuron has access to the same filters.
    """

    assert (Ms is None) != ((A is None) or (B is None))

    # Boring bookkeeping
    TEs = np.atleast_2d(TEs)
    N_neurons = TEs.shape[0]
    N_flts = len(flts)
    gains, biases = np.atleast_1d(gains), np.atleast_1d(biases)
    flts_rec_map = np.atleast_2d(flts_rec_map)
    assert (TEs.shape[0] == gains.shape[0] == biases.shape[0])
    if Ms is None:
        A = np.atleast_2d(A)
        B = np.atleast_1d(B).flatten()
        q = N_temporal_dimensions = A.shape[0]
        assert A.shape[0] == A.shape[1] == B.shape[0]
    else:
        Ms = np.atleast_2d(Ms)
        q = N_temporal_dimensions = Ms.shape[1]
    assert TEs.shape[1] == q
    assert flts_rec_map.shape == (N_neurons, N_neurons)

    # Evaluate all filters
    ts = np.arange(0, T, dt)
    N = len(ts)
    hs = np.array([flt(ts, dt) for flt in flts])

    # Compute the impulse response of the desired LTI system
    if Ms is None:
        Ms = cached_lti_impulse_response(A, B, ts)
    else:
        assert Ms.shape[0] == N

    # Assemble the X and Y matrices that are being passed to the least-squares
    # algorithm
    N_flts_in = len(flts_in_map)
    N_pre = N_flts_in + N_neurons

    flts_in_map_u = np.unique(flts_in_map)
    flts_rec_map_u = np.unique(flts_rec_map)

    X = np.zeros((N_smpls, N_neurons, N_pre))
    Y = np.zeros((N_smpls, N_neurons))

    x_in_scale = 1.0 * N_neurons  # Compensate for the input not being neural activitie
    for i_smpl in tqdm.tqdm(range(N_smpls), disable=silent):
        # Sample an input sequence; increase the magnitude to prevent regularisation problems
        xs = mk_sig(N,
                    dt,
                    sigma=xs_sigma,
                    scale=xs_scale,
                    rng=rng,
                    Ms=Ms,
                    i_smpl=i_smpl,
                    biased=biased,
                    bias_cstr_count=bias_cstr_count)

        # Filter the input with all filters used in the input connections
        xs_flt = np.zeros(N_flts)
        for i_flt in flts_in_map_u:
            xs_flt[i_flt] = np.convolve(xs, hs[i_flt], 'valid') * dt

        # Pass the input through the desired impulse responses
        ms_flt = np.zeros((N, q))
        for i_q in range(q):
            ms_flt[:, i_q] = scipy.signal.fftconvolve(xs, Ms[:, i_q], 'full')[:N] * dt

        # For each filter, and each pre-neuron, compute the activity at the
        # post-neuron at time zero
        As_pre = np.zeros((N_neurons, N_flts))
        for i_pre in range(N_neurons):
            A_pre = G(gains[i_pre] * (ms_flt @ TEs[i_pre]) + biases[i_pre])
            for i_flt in flts_rec_map_u:
                As_pre[i_pre, i_flt] = np.convolve(A_pre, hs[i_flt], 'valid') * dt

        # Now re-arrange the matrices computed above into a series of
        # least-squares problems, one for each neuron
        for i_post in range(N_neurons):
            I = 0
            # Input
            for i_pre in range(N_flts_in):
                X[i_smpl, i_post, I] = xs_flt[flts_in_map[i_pre]] * x_in_scale
                I += 1

            # Recurrent connection
            for i_pre in range(N_neurons):
                X[i_smpl, i_post, I] = As_pre[i_pre, flts_rec_map[i_post, i_pre]]
                I += 1

        # Compute the target input current of each post-neuron
        for i_post in range(N_neurons):
            Y[i_smpl, i_post] = (ms_flt[-1] @ TEs[i_post])


    # Solve the least-squares problems one at a time
    W = np.zeros((N_neurons, N_pre))
    Es = np.zeros(N_neurons)
    for i in range(N_neurons):
        XTX = X[:, i].T @ X[:, i] + N_smpls * np.square(reg * np.max(X)) * np.eye(N_pre)
        XTY = X[:, i].T @ Y[:, i]
        W[i] = np.linalg.solve(XTX, XTY)
        Es[i] = np.sqrt(np.mean(np.square(X[:, i] @ W[i] - Y[:, i])))

    W_in = W[:, :N_flts_in] * x_in_scale
    W_rec = W[:, N_flts_in:]

    return W_in, W_rec, Es

