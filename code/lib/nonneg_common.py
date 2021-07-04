import lif_utils
import numpy as np
import scipy.optimize

def mk_ensemble(N,
                x_intercepts_cback=None,
                encoders_cback=None,
                d=1,
                rng=np.random,
                max_rates=(50, 100)):
    def default_x_intercepts(rng, N):
        return rng.uniform(-0.99, 0.99, N)

    def default_encoders_cback(rng, N, d):
        return rng.normal(0, 1, (N, d))

    x_intercepts_cback = (default_x_intercepts if x_intercepts_cback is None
                          else x_intercepts_cback)
    encoders_cback = (default_encoders_cback
                      if encoders_cback is None else encoders_cback)

    max_rates = rng.uniform(*max_rates, N)
    x_intercepts = x_intercepts_cback(rng, N)  #rng.uniform(-0.99, 0.99, N)

    J0s = lif_utils.lif_rate_inv(1e-3)
    J1s = lif_utils.lif_rate_inv(max_rates)

    gains = (J1s - J0s) / (1.0 - x_intercepts)
    biases = (J0s - x_intercepts * J1s) / (1.0 - x_intercepts)

    encoders = encoders_cback(rng, N, d)  #rng.normal(0, 1, (N, d))
    encoders /= np.linalg.norm(encoders, axis=1)[:, None]

    if d == 1:
        idcs = np.argsort(x_intercepts * encoders[:, 0])
    else:
        idcs = np.arange(N)

    return gains[idcs], biases[idcs], encoders[idcs]


def decode_currents(As,
                    Js_tar,
                    p_exc=0.5,
                    do_dale=True,
                    sigma=0.1,
                    split=False,
                    rng=np.random,
                    is_exc=None,
                    is_inh=None):
    assert (is_exc is None) == (is_inh is None) == (not p_exc is None)
    assert As.shape[0] == Js_tar.shape[0]
    N = As.shape[0]
    N_post = Js_tar.shape[1]
    N_pre = As.shape[1]

    if not do_dale:
        is_exc = np.ones(N_pre, dtype=bool)
        is_inh = np.ones(N_pre, dtype=bool)
    else:
        if is_exc is None:
            if p_exc == 1.0:
                is_exc = np.ones(N_pre, dtype=bool)
            elif p_exc == 0.0:
                is_exc = np.zeros(N_pre, dtype=bool)
            else:
                is_exc = rng.choice([False, True], N_pre, p=[1.0 - p_exc, p_exc])
            is_inh = ~is_exc
        else:
            assert is_exc.shape == (As.shape[1],)
            assert is_inh.shape == (As.shape[1],)

    N_exc = np.sum(is_exc)
    N_inh = np.sum(is_inh)
    i0, i1, i2 = 0, N_exc, N_exc + N_inh

    idcs_exc = np.where(is_exc)[0]
    idcs_inh = np.where(is_inh)[0]

    As_exc = As[:, is_exc]
    As_inh = As[:, is_inh]
    As_comb = np.concatenate((As_exc, -As_inh), axis=1)

    A = (As_comb.T @ As_comb +
         N * np.square(sigma * np.max(As)) * np.eye(N_exc + N_inh))
    Y = As_comb.T @ Js_tar

    W_exc = np.zeros((N_post, N_exc))
    W_inh = np.zeros((N_post, N_inh))
    for i in range(N_post):
        w = scipy.optimize.nnls(A, Y[:, i])[0]
        W_exc[i] = w[i0:i1]
        W_inh[i] = w[i1:i2]

    if split:
        return W_exc, W_inh, idcs_exc, idcs_inh

    W = np.zeros((N_post, N_pre))
    W[:, idcs_exc] += W_exc
    W[:, idcs_inh] -= W_inh
    return W
