import scipy.linalg
import dlop_ldn_function_bases as bases

def mk_mod_fourier_lti(q, fac=0.9, Ninternal=1000):
    def mk_fourier_oscillator(q, mul=1.0):
        B = (np.arange(0, q) + 1) % 2
        A = np.zeros((q, q))
        for k in range(1, q):
            ki = (k + 1) // 2
            fk = 2.0 * np.pi * mul * ki
            A[2 * ki - 1, 2 * ki - 1] = 0
            A[2 * ki - 1, 2 * ki + 0] = fk
            A[2 * ki + 0, 2 * ki - 1] = -fk
            A[2 * ki + 0, 2 * ki + 0] = 0
        return A, B

    assert q % 2 == 1

    A, B = mk_fourier_oscillator(q, mul=0.9)
    Ad, Bd = np.zeros((q, q)), np.zeros((q, ))

    Ad[1:, 1:], Bd[1:] = bases.discretize_lti(1.0 / Ninternal, A[1:, 1:],
                                              B[1:])
    Bd[0] = 1e-3
    Ad[0, 0] = 1.0

    H = bases.mk_lti_basis(Ad,
                           Bd,
                           Ninternal,
                           from_discrete_lti=True,
                           normalize=False)
    enc = H[:, 0]
    dec = np.linalg.pinv(H, rcond=1e-2)[0]

    Ad = Ad - np.outer(enc, dec) @ Ad
    Bd = Bd - np.outer(enc, dec) @ Bd

    A = np.real(scipy.linalg.logm(Ad)) * Ninternal
    return A, B


def mexpfin(A, o):
    if o == 0:
        return np.eye(A.shape[0])
    res = np.zeros_like(A)
    f = 1.0
    for i in range(1, o + 1):
        res += np.linalg.matrix_power(A, i) * f
        f /= (i + 1)
    return res


def mk_h(q, N, p=4):
    Ao, Bo = mk_mod_fourier_lti(q, N)
    A, B = Ao / N, Bo / N
    A = mexpfin(A, p)
    At, Bt = A + np.eye(q), B
    H = np.zeros((q, 3 * N))
    Aexp = np.eye(q)
    for i in range(3 * N):
        H[:, i] = Aexp @ Bt
        Aexp = At @ Aexp
    H = H / np.linalg.norm(H, axis=1)[:, None]
    return H


fig, axs = plt.subplots(1, 4, figsize=(8.0, 1.25))

q = 101
Ns = np.geomspace(200, 1000, axs.size, dtype=int)
for i, ax in enumerate(axs.flat):
    H = mk_h(q, Ns[i])
    ts = np.linspace(0, H.shape[1] / Ns[i], H.shape[1])

    scale = np.percentile(np.abs(H.flatten()), 95)
    H /= scale

    mpl.rcParams['path.simplify_threshold'] = 0.5
    for k, j in enumerate(range(1, q, 3)):
        ax.plot(ts, H[j], zorder=-j, color=cm.get_cmap('magma')((k % 10) / 10), lw=1.25)

    _, S, _ = np.linalg.svd(H)
    S /= np.max(S)

    ax.plot(0.1,
            0.9,
            'o',
            color=utils.grays[0],
            markersize=8,
            transform=ax.transAxes)
    ax.text(0.1,
            0.89,
            f"\\textbf{{{i + 1}}}",
            color="white",
            transform=ax.transAxes,
            ha="center",
            va="center")
    ax.text(0.95,
            0.85,
            "$N = {}$\n$\\Sigma = {:0.1f}$".format(Ns[i], np.sum(S)),
            ha="right",
            va="baseline",
            color="black",
            transform=ax.transAxes,
            bbox={
                "pad": 0.25,
                "color": "white",
            })
    ax.set_ylim(-2, 2)

    ax.set_xticks(np.linspace(0, H.shape[1] / Ns[i], 4, dtype=int))
    ax.set_xticks(np.linspace(0, H.shape[1] / Ns[i], 7), minor=True)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])

utils.save(fig)

