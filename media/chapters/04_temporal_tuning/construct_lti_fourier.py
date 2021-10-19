import scipy.linalg
import numpy as np


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
        enc, dec = enc, dec = H[:, 0], np.linalg.pinv(H)[0]
        At = At - np.outer(enc, dec) @ At
        Bt = Bt - np.outer(enc, dec) @ Bt

    if return_discrete:
        return At, Bt

    # Undo discretization (this is the inverse of discretize_lti)
    A = np.real(scipy.linalg.logm(At)) * N / T
    B = scipy.linalg.expm(-0.5 * A * T / N) @ Bt * np.sqrt(N)

    return A, B


def eval_lti(A, B, ts):
    return np.array([scipy.linalg.expm(A * t) @ B for t in ts])


def plot_lti_basis(
    ax,
    mk_basis_fun,
    q=11,
    q_plot=6,
    N=1000,
    T=1.1,
    dt=1e-3,
    dampen=False,
):
    ts_H = np.linspace(0, 1, N)[::-1]
    H = mk_basis_fun(q, N)

    ts = np.arange(0, T, dt)
    A, B = reconstruct_lti(H, dampen=dampen)

    ys = eval_lti(A, B, ts)[:, :q_plot]
    for i in range(q_plot):
        ax.plot(ts, ys[:, i])
    ax.plot(ts_H, H.T[:, :q_plot] * np.sqrt(N), 'k:', lw=0.5)

    return A, B


import dlop_ldn_function_bases as bases


def mk_mod_fourier_basis(q, N):
    N_tot = int(1.1 * N)
    return bases.mk_fourier_basis(q, int(N_tot))[:, (N_tot - N):]

fig, ax = plt.subplots(1, 1, figsize=(6.0, 2.0))
plot_lti_basis(ax, bases.mk_fourier_basis, dampen='erasure', T=3.0)
ax.set_title("\\textbf{Fourier basis with information erasure} ($q = 11$)")
ax.set_xlabel("Time $t$ (s)")
ax.set_ylim(-1.5, 1.5)
ax.set_xlim(0, 3)
utils.save(fig, suffix="_a")

fig, ax = plt.subplots(1, 1, figsize=(6.0, 2.0))
plot_lti_basis(ax, mk_mod_fourier_basis, dampen='erasure', T=3.0)
ax.set_title("\\textbf{Modified Fourier basis with information erasure} ($q = 11$)")
ax.set_xlabel("Time $t$ (s)")
ax.set_ylim(-1.5, 1.5)
ax.set_xlim(0, 3)
utils.save(fig, suffix="_b")


