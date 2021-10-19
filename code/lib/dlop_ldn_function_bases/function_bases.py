# Code Generating Discrete Legendre Orthogonal Polynomials and the
# Legendre Delay Network Basis
#
# Andreas Stöckel, December 2020
#
# The code in this file is licensed under the Creative Commons Zero license.
# To the extent possible under law, Andreas Stöckel has waived all copyright and
# related or neighboring rights to this code. You should have received a
# complete copy of the license along with this code. If not, please visit
#
#    https://creativecommons.org/publicdomain/zero/1.0/legalcode-plain.
#
# This work is published from: Canada.

import numpy as np
import scipy.linalg

## Utility functions


def fading_factorial(K, m):
    # Fading factorial as defined in the Neuman and Schonbach paper
    res = 1
    for i in range(m):
        res *= K - i
    return res


def nCr(n, r):
    # Binomial coefficient (n choose r; nCr is what my trusty pocket
    # calculator calls it).
    return fading_factorial(n, r) // \
           fading_factorial(r, r)


## Polynomial generation


def _shift_polys(P, domain=(0, 1), window=(-1, 1)):
    """
    Shifts and scales a polynomial basis from the source onto the target window
    """
    from fractions import Fraction
    s0, s1 = Fraction(domain[0]), Fraction(domain[1])
    t0, t1 = Fraction(window[0]), Fraction(window[1])
    a = (t1 - t0) / (s1 - s0)
    b = t0 - a * s0
    print(a, b)
    Pout = np.zeros(P.shape)
    for i in range(P.shape[0]):  # Polynomial basis index
        for k in range(P.shape[1]):  # Coefficient index
            beta = Fraction(0)
            for n in range(k, P.shape[1]):
                beta += nCr(n, k) * Fraction(P[i, n]) * (a**k) * (b**(n - k))
            Pout[i, k] = beta
    return Pout


def _mk_poly_basis(q, fun):
    P = np.zeros((q, q))
    for i in range(q):
        p = fun([0] * i + [1])
        P[i, :len(p)] = p
    return P


def mk_power_poly_basis(q, domain=(0, 1), window=(-1, 1), offs=1.0, scale=0.5):
    P = np.eye(q)
    P[:, 0] = offs  # Constant offset
    return _shift_polys(scale * P, domain=domain, window=window)


def mk_leg_poly_basis(q, domain=(0, 1), window=(-1, 1)):
    return _shift_polys(_mk_poly_basis(q, np.polynomial.legendre.leg2poly),
                        domain=domain,
                        window=window)


def mk_lag_poly_basis(q, domain=(0, 1), window=(0, 1)):
    return _shift_polys(_mk_poly_basis(q, np.polynomial.laguerre.lag2poly),
                        domain=domain,
                        window=window)


def mk_cheb_poly_basis(q, domain=(0, 1), window=(-1, 1)):
    return _shift_polys(_mk_poly_basis(q, np.polynomial.chebyshev.cheb2poly),
                        domain=domain,
                        window=window)


## Generic LTI Code


def discretize_lti(dt, A, B):
    """
    Discretizes an LTI system described by matrices A, B under a
    zero-order-hold (ZOH) assumption. The new matrices Ad, Bd can be used in
    the following manner

       x[t + 1] = Ad x[t] + Bd u[t] ,

    i.e., the returned matrices implicitly contain the
    integration step.

    See https://en.wikipedia.org/wiki/Discretization for
    more information.
    """
    # See https://en.wikipedia.org/wiki/Discretization
    Ad = scipy.linalg.expm(A * dt)
    Bd = np.linalg.solve(A, (Ad - np.eye(A.shape[0])) @ B)
    return Ad, Bd


def reconstruct_lti(H,
                    T=1.0,
                    dampen=False,
                    return_discrete=False,
                    rcond=1e-2,
                    dampen_fac=1.0):
    """
    Given a discrete q x N basis transformation matrix H constructs a linear
    time-invariant dynamical system A, B that approximately has this basis
    as an impulse response over [0, T].

    This function can be thought of as the inverse of mk_lti_basis.

    H: Basis transformation matrix for which the LTI system should be
        reconstructed.

    dampen: Determines the dampening mode. If not False (default), dampen may
        be one of "lstsq" or "erasure". Furthermore, "dampen" may be a set
        containing both strings. Both methods will be used in this case.
        Lastly, dampen can be set to True, which is equivalent to "delay".

        In "lstsq" mode, an appropriately weighted dampening term is added to
        the system of equations. This term is meant to encourage the resulting
        system to converge to zero for t > T, but this is not guaranteed.

        In "erasure" mode the system is dampened by explicitly erasing
        information beyond the current time-window.

    return_discrete: If True, returns the discreteized LTI system; does not
        attempt to convert the system into a continuous system.

    rcond: Regularisation term passed to the least-squares solver.

    dampen_fac: Factor used to weight the dampening term in the least-squares
        solution. Only relevant if `dampen` is set to "lstsq".
    """
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

    # Construct the least squares problem. If dampen is True, prepend a row of
    # zeros to the system. This dampening factor is weightened in the linear
    # system of equations to maintain a ratio of 1 : (q - 1) between the
    # dampening term and the remaining weights.
    if "lstsq" in dampen:
        X = H.T
        Y = np.concatenate((np.zeros((q, 1)), H[:, :-1]), axis=1).T
        w0 = np.sqrt(dampen_fac * (N - 1) / (q - 1))
        W = np.array([w0] + [1] * (N - 1))
    else:
        X = H[:, 1:].T
        Y = H[:, :-1].T
        W = np.ones(N - 1)

    # Estimate the discrete system At, Bt
    At = np.linalg.lstsq(W[:, None] * X, W[:, None] * Y, rcond=rcond)[0].T
    Bt = H[:, -1]

    # In "delay mode" subtract the outer product of the longest delay
    # decoder/encoder pair from the state in each timestep
    if "erasure" in dampen:
        enc, dec = enc, dec = H[:, 0], np.linalg.pinv(H)[0]
        At = At - np.outer(enc, dec) @ At
        Bt = Bt - np.outer(enc, dec) @ Bt

    # If so desired, return the discrete system
    if return_discrete:
        return At, Bt

    # Undo discretization (this is the inverse of discretize_lti)
    A = np.real(scipy.linalg.logm(At)) * N / T
    B = np.linalg.solve(At - np.eye(q), A @ Bt) / T

    return A, B


def mk_lti_basis(A, B, N=None, N_smpl=None, normalize=True, from_discrete_lti=False):
    """
    Constructs a basis transformation matrix for the given LTI system. This
    function is used internally by mk_ldn_basis.

    A: q x q LTI system feedback matrix
    B: q LTI system input vectur
    N: Window width. Defaults to q
    N_smpl: Actual number of samples to return. Defaults to N
    """
    # Make sure A is a square matrix
    assert (A.ndim == 2) and (A.shape[0] == A.shape[1])
    q = A.shape[0]

    # Make sure B has the right shape
    B = B.flatten()
    assert (B.shape[0] == q)

    # Fetch the window width
    N = q if N is None else int(N)
    assert N > 0

    # Fetch the number of samples
    N_smpl = N if N_smpl is None else int(N_smpl)
    assert N_smpl > 0

    # Generate the impulse response matrix
    At, Bt = (A, B) if from_discrete_lti else discretize_lti(1.0 / N, A, B)
    res = np.zeros((q, N_smpl))
    Aexp = np.eye(q)
    for i in range(N_smpl):
        res[:, N_smpl - i - 1] = Aexp @ Bt
        Aexp = At @ Aexp
    return (res / np.linalg.norm(res, axis=1)[:, None]) if normalize else res


def mk_poly_basis_lti(P, rcond=None, dampen=False, reencoder_kw={}):
    """
    Converts a set of polynomials into the corresponding LTI system.

    P is a matrix of polynomial coefficients i.e.,
        p_n(x) = sum_{i = 0}^{q - 1} P_{n + 1, i + 1} x^i

    The polynomials should form a function basis over [0, 1].
    """
    assert P.shape[0] == P.shape[1]
    assert np.linalg.matrix_rank(P, tol=1e-6) == P.shape[0]
    q = P.shape[0]

    # Compute the differential of each polynomial
    PD = np.concatenate((P[:, 1:], np.zeros(
        (q, 1))), axis=1) * np.arange(1, q + 1)[None, :]

    # Compute a matrix constructing the differential from the other polynomials
    A = np.linalg.lstsq(P.T, PD.T, rcond=rcond)[0].T

    # The x-intercept is equal to B
    B = P[:, 0]

    # If dampening is requested, subtract the delay re-encoder from A
    return (A - mk_poly_basis_reencoder(P, **reencoder_kw) if dampen else A), B


def mk_poly_sys_lti(A, B, dampen=True, reencoder_kw={}):
    return (A -
            mk_poly_sys_reencoder(A, B, **reencoder_kw) if dampen else A), B


def mk_poly_basis_reencoder_hilbert(P):
    # Flip the polynomials
    N, q = P.shape
    P_flip = _shift_polys(P, window=(1, 0))

    # Evaluate the target polynomials at `theta - theta' = 0` and at `theta`
    y = np.array([np.polyval(P[i, ::-1], 0) for i in range(N)])
    e = np.array([np.polyval(P[i, ::-1], 1) for i in range(N)])

    # Compute the inverse Hilbert matrix
    QInv = scipy.linalg.invhilbert(P.shape[1])
    return np.outer(e, np.linalg.solve(P.T, QInv @ np.linalg.inv(P_flip) @ y))


def mk_poly_basis_inverse_hilbert(P, rcond=1e-6):
    """
    Computes a polynomial basis that is orthogonal to P over the interval
    [0, 1].
    """
    N, q = P.shape
    QInv = scipy.linalg.invhilbert(q)
    return np.linalg.lstsq(P.T, QInv, rcond=rcond)[0]


def mk_poly_basis_reencoder_hilbert_2(P, rcond=1e-6):
    (N, _), PI = P.shape, mk_poly_basis_inverse_hilbert(P, rcond=rcond)
    e, d = np.zeros((2, N))
    for i in range(N):
        e[i], d[i] = np.polyval(P[i], 1), np.polyval(PI[i], 1)
    return np.outer(e, d)


def mk_poly_basis_reencoder(P, rcond=1e-6, dt=1e-4):
    N = int(
        (1.0 + dt + 1e-12) / dt)  # #samples; 1e-12 prevents rounding errors
    xs = np.linspace(0, 1, N)
    H = np.array([np.polyval(P[i, ::-1], xs) for i in range(P.shape[0])])
    HInv = np.linalg.pinv(H.T, rcond=rcond) / dt
    return np.outer(H[:, -1], HInv[:, -1])


def mk_poly_sys_reencoder(A, B, rcond=1e-6, dt=1e-4):
    N = int(
        (1.0 + dt + 1e-12) / dt)  # #samples; 1e-12 prevents rounding errors
    xs = np.linspace(0, 1, N)
    H = np.array([scipy.linalg.expm(A * xs[j]) @ B for j in range(N)]).T
    HInv = np.linalg.pinv(H.T, rcond=rcond) / dt
    return np.outer(H[:, -1], HInv[:, -1])


## Legendre Delay Network (LDN)


def mk_ldn_lti(q, dtype=np.float, rescale=False):
    """
    Generates the A, B matrices of the linear time-invariant (LTI) system
    underlying the LDN.

    The returned A is a q x q matrix, the returned B is a vector of length q.
    Divide the returned matrices by the desired window length theta.

    See Aaron R. Voelker's PhD thesis for more information:
    https://hdl.handle.net/10012/14625 (Section 6.1.3, p. 134)

    If `rescale` is True, a less numerically stable version of the LDN is
    returned that exactly traces out the shifted Legendre polynomials,
    including scaling factors.
    """
    qs = np.arange(q)
    A = -np.ones((q, q), dtype=dtype)
    for d in range(1, q, 2):  # iterate over odd diagonals
        A[range(d, q), range(0, q - d)] = 1
    B = np.ones((q, ), dtype=dtype)
    B[1::2] = -1
    if rescale:
        return (2 * qs[None, :] + 1) * A, B
    else:
        return (2 * qs[:, None] + 1) * A, \
               (2 * qs + 1) * B


def mk_leg_lti(q, dtype=np.float):
    """
    Assembles an LTI system that has the Legendre polynomials as an impulse
    response.
    """
    qs = np.arange(q)
    A = np.zeros((q, q), dtype=dtype)
    for d in range(1, q, 2):  # iterate over odd diagonals
        A[range(d, q), range(0, q - d)] = 1
    B = np.ones((q, ), dtype=dtype)
    B[1::2] = -1
    return (4 * qs[None, :] + 2) * A, B


## Chebyshev LTI code


def mk_cheb_lti(q, dtype=np.float):
    qs = np.arange(q)
    A = np.zeros((q, q), dtype=dtype)
    for d in range(1, q, 2):  # iterate over odd diagonals
        A[range(d + 1, q), range(1, q - d)] = 2
        A[d, 0] = 1
    B = np.ones((q, ), dtype=dtype)
    B[1::2] = -1
    return (2 * qs[:, None]) * A, B


## Legendre Delay Network Basis


def mk_ldn_basis_euler(q, N=None, normalize=True):
    """
    This function is the attempt at generating a LDN basis using naive Euler
    integration. This produces horribly wrong results.
    
    For reference only, DO NOT USE. Use `mk_ldn_basis` instead.
    """
    q, N = int(q), int(q) if N is None else int(N)
    A, B = mk_ldn_lti(q)
    At, Bt = A / N + np.eye(q), B / N
    res = np.zeros((q, N))
    Aexp = np.eye(q)
    for i in range(N):
        res[:, N - i - 1] = Aexp @ Bt
        Aexp = At @ Aexp
    return (res / np.linalg.norm(res, axis=1)[:, None]) if normalize else res


def mk_ldn_basis(q, N=None, normalize=True):
    """
    Generates the LDN basis for q basis vectors and N input samples.  Set
    `normalize` to `False` to obtain the exact LDN impulse response, otherwise
    a normalized basis transformation matrix as defined in the TR is returned.
    """
    return mk_lti_basis(*mk_ldn_lti(q), N, normalize=normalize)


## Discrete Legendre Orthogonal Polynomial Basis and Related Code


def mk_leg_basis(q, N=None):
    """
    Creates a non-orthogonal basis by simply sampling the Legendre polynomials.
    """
    q, N = int(q), int(q) if N is None else int(N)
    xs0 = np.linspace(0.0, 1.0, N + 1)[:-1]
    xs1 = np.linspace(0.0, 1.0, N + 1)[1:]
    res = np.zeros((q, N))
    for n in range(q):
        Pn = np.polynomial.Legendre([0] * n + [1], [1, 0]).integ()
        res[n] = Pn(xs1) - Pn(xs0)
    return res / np.linalg.norm(res, axis=1)[:, None]


def mk_dlop_basis_linsys(q, N=None):
    """
    Constructs a matrix of "Discrete Legendre Orthogonal Polynomials" (DLOPs).
    q is the number of polynomials to generate, N is the number of samples for
    each polynomial.

    This is function is for reference only and should not be used. It is
    unstable for q > 30 (if N > q the function may be stable longer).

    This function uses a rather inefficient approach that directly relies on
    the definition of a Legendre Polynomial (a set of orthogonal Polynomials
    with Pi(1) = 1.0) to generate the basis.

    In each iteration i, this function adds a new polynomial of degree i to the
    set of already computed polynomials. The polynomial coefficients are
    determined by solving for coefficients that generate discrete sample points
    that are orthogonal to the already sampled basis vectors.

    The returned basis is made orthogonal by dividing by the norm of each
    discrete polynomial.
    """
    # Construct the sample points
    q, N = int(q), int(q) if N is None else int(N)
    qs, Ns = np.arange(q), np.arange(N)
    xs = 2.0 * Ns / (N - 1.0) - 1.0

    # Evaluate the individual monomials (this is a Vandermonde matrix)
    M = np.power(xs[:, None], qs[None, :])

    # Create the matrix. The first basis vector is "all ones"
    res = np.zeros((q, N))
    res[0] = 1.0

    # Solve for polynomial coefficients up to degree q such that the newly
    # added basis vector is orthogonal to the already created basis vectors,
    # and such that the last sample is one.
    for i in range(1, q):
        A = np.zeros((i + 1, i + 1))
        b = np.zeros((i + 1, ))
        b[-1] = 1.0
        A[:i, :] = res[:i, :] @ M[:, :i + 1]
        A[i, :] = M[0, :i + 1]
        coeffs = np.linalg.solve(A, b)
        res[i] = M[:, :i + 1] @ coeffs

    return res / np.linalg.norm(res, axis=1)[:, None]


def mk_dlop_basis_direct(q, N=None):
    """
    Slow, direct implementation of the DLOP basis according to

    Neuman, C. P., & Schonbach, D. I. (1974).
    Discrete (legendre) orthogonal polynomials—A survey.
    International Journal for Numerical Methods in
    Engineering, 8(4), 743–770.
    https://doi.org/10.1002/nme.1620080406

    Note that this code relies on the fact that Python 3 always uses
    "big ints" or ("long" in Python 2 terms). The integers used in this
    function will likely not fit into 32- or 64-bit integers; so be careful
    when porting this code to a different programing language.
    """
    q, N = int(q), int(q) if N is None else int(N)
    res = np.zeros((q, N))
    for m in range(q):
        # Usa a common denominator instead of dividing by
        # fading_factorial(N - 1, j), where "j" is the inner loop variable.
        # Instead we divide all terms by fading_factorial(N - 1, m) and
        # multiply the terms by the additional terms that we're dividing by.
        # This way we can perform the final division numer / denom computing
        # the float output at the very end; everything up to this point is
        # precise integer arithmetic.
        denom = fading_factorial(N - 1, m)
        for K in range(N):
            numer = 0
            for j in range(m + 1):
                # Main equation from the paper. The last term corrects for the
                # common denominator.
                c = nCr(m, j) * nCr(m + j, j) * \
                   fading_factorial(K, j) * \
                   fading_factorial(N - 1 - j, m - j)
                numer += c if (j % 2 == 0) else -c
            res[m, K] = numer / denom
        res[m]

    return res / np.linalg.norm(res, axis=1)[:, None]


def mk_dlop_basis_recurrence(q, N=None):
    """
    Computes the DLOP basis using the Legendre recurrence relation as described
    in the section "Generation Scheme" of Neuman & Schonbach, 1974, pp. 758-759
    (see above for the full reference).

    Do NOT use this function. This function is numerically unstable and only
    included as a reference. Use `mk_dlop_basis` instead
    """

    # Fill the first rows
    q, N = int(q), int(q) if N is None else int(N)
    res = np.zeros((q, N))
    if q > 0:
        res[0] = np.ones(N)
    if q > 1:
        res[1] = np.linspace(1, -1, N)

    # Iterate over all columns
    for K in range(N):
        # Compute the initial coefficients for the recurrence relation
        c0, c1, c2 = 0, N - 2 * K - 1, N - 1
        δ0, δ1, δ2 = N - 1, 2 * c1, N - 1

        # Iterate over all rows
        for m in range(2, q):
            δ0, δ1, δ2 = δ0 + 2, δ1, δ2 - 2
            c0, c1, c2 = c0 + δ0, c1 + δ1, c2 + δ2
            res[m, K] = (c1 * res[m - 1, K] - c0 * res[m - 2, K]) / c2

    return res / np.linalg.norm(res, axis=1)[:, None]


def mk_dlop_basis(q, N=None, eps=1e-7):
    """
    Same as `mk_dlop_basis_recurrence`, but updates all columns at once using
    numpy.
    """

    # Fill the first rows
    q, N = int(q), int(q) if N is None else int(N)
    res = np.zeros((q, N))
    if q > 0:
        res[0] = np.ones(N) / np.sqrt(N)
    if q > 1:
        res[1] = np.linspace(1, -1, N) * np.sqrt((3 * (N - 1)) / (N * (N + 1)))

    # Pre-compute the coefficients c0, c1. See Section 4.4 of the TR.
    Ks = np.arange(0, N, dtype=np.float)[None, :]
    ms = np.arange(2, q, dtype=np.float)[:, None]
    α1s = np.sqrt(  ((2 * ms + 1) * (N - ms)) \
                  / ((2 * ms - 1) * (N + ms)))
    α2s = np.sqrt(  ((2 * ms + 1) * (N - ms) * (N - ms + 1)) \
                  / ((2 * ms - 3) * (N + ms) * (N + ms - 1)))
    β1s = α1s * ((2 * ms - 1) * (N - 2 * Ks - 1) / (ms * (N - ms)))
    β2s = α2s * ((ms - 1) * (N + ms - 1) / (ms * (N - ms)))

    # The mask is used to mask out columns that cannot become greater than one
    # again. This prevents numerical instability.
    mask = np.ones((q, N), dtype=np.bool)

    # Evaluate the recurrence relation
    for m in range(2, q):
        # A column K can only become greater than zero, if one of the
        # cells in the two previous rows was significantly greater than zero.
        mask[m] = np.logical_or(mask[m - 1], mask[m - 2])

        # Apply the recurrence relation
        res[m] = (  (β1s[m - 2]) * res[m - 1] \
                  - (β2s[m - 2]) * res[m - 2]) * mask[m]

        # Mask out cells that are smaller than some epsilon
        mask[m] = np.abs(res[m]) > eps

    return res


## Fourier and Cosine Basis


def mk_fourier_basis(q, N=None):
    """
    Generates the q x N matrix F that can be used to compute a Fourier-like
    transformation of a real-valued input vector of length N.  The first
    result dimension will be the DC offset.  Even result dimensions are the
    real (sine) Fourier coefficients, odd dimensions are the imaginary (cosine)
    coefficients.
    
    Beware that this is a only a Fourier transformation for q = N, and even
    then not a "proper" Fourier transformation because the transformation
    matrix is normalized to be orthogonal.  So be careful when comparing the
    results of this function to "standard" Fourier transformations.
    """
    q, N = int(q), int(q) if N is None else int(N)
    qs, Ns = np.arange(q)[:, None], np.arange(N)[None, :]
    freq = ((qs + 1) // 2)  # 0, 1, 1, 2, 2, ...
    phase = (qs % 2)  # 0, 1, 0, 1, 0, ...
    F = np.cos(
        2.0 * np.pi * freq * (Ns + 0.5) / N + \
        0.5 * np.pi * phase)
    F[0] /= np.sqrt(2)
    F[-1] /= np.sqrt(2) if (q % 2 == 0 and N == q) else 1.0
    return F * np.sqrt(2 / N)


def mk_fourier_basis_derivative(q, N=None):
    """
    Returns the derivative of the Fourier series.
    """
    q, N = int(q), int(q) if N is None else int(N)
    qs, Ns = np.arange(q)[:, None], np.arange(N)[None, :]
    freq = ((qs + 1) // 2)  # 0, 1, 1, 2, 2, ...
    phase = (qs % 2)  # 0, 1, 0, 1, 0, ...
    F = -2.0 * np.pi * freq * np.sin(
        2.0 * np.pi * freq * (Ns + 0.5) / N + \
        0.5 * np.pi * phase)
    F[0] /= np.sqrt(2)
    F[-1] /= np.sqrt(2) if (q % 2 == 0 and N == q) else 1.0
    return F * np.sqrt(2 / N)


def mk_cosine_basis(q, N=None):
    """
    Generates the q x N matrix C which can be used to compute the orthogonal
    DCT-II, everyone's favourite basis transformation.  As with the
    `mk_fourier_basis` function above, this code only returns a canonical
    DCT basis if q = N.
    """
    q, N = int(q), int(q) if N is None else int(N)
    qs, Ns = np.arange(q)[:, None], np.arange(N)[None, :]
    C = np.cos((Ns + 0.5) / N * qs * np.pi)
    C[0] /= np.sqrt(2)
    return C * np.sqrt(2 / N)


def mk_haar_basis(q, N=None):
    """
    Generates the Haar wavelets. Note that the resulting matrix is not
    orthogonal exactly if N is not a power of two.
    """
    def subdiv(r0, r1):
        if r1 - r0 > 0:
            c = r0 + (r1 - r0 + 1) // 2
            yield (r0, c, r1)
            for L, R in zip(subdiv(r0, c), subdiv(c, r1)):
                yield L
                yield R

    q, N = int(q), int(q) if N is None else int(N)
    res = np.zeros((q, N))
    res[0] = 1
    for q, (i0, i1, i2) in zip(range(1, q), subdiv(0, N)):
        res[q, i0:i1] = 1
        res[q, i1:i2] = -1
    return res / np.linalg.norm(res, axis=1)[:, None]


## Low-pass filtered bases


def lowpass_filter_basis(T, qp=None, filter_ctor=mk_fourier_basis):
    """
    Takes a basis T with shape q x N and returns a basis that additionally
    applies a low-pass filter to the N-dimensional input, such that the input
    is represented by qp Fourier coefficients. This is equivalent to
    low-pass filtering the basis T itself, i.e., representing the discrete
    basis functions in terms of the Fourier basis.

    This function has no effect if qp = N; in this case, there is no
    potential for information loss.

    The "filter_ctor" parameter can be used to represent the given basis T in
    terms of another function basis.
    """
    (q, N), qp = T.shape, (T.shape[0] if qp is None else int(qp))
    F = filter_ctor(qp, N)
    return T @ np.linalg.pinv(F) @ F

