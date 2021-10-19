import scipy.linalg
import dlop_ldn_function_bases as bases
import tqdm

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

def mk_ldn_basis(q, N, Nmul=3):
    A, B = bases.mk_ldn_lti(q, rescale=True)
    return bases.mk_lti_basis(A, B, N, N * Nmul)

def mk_mod_fourier_basis(q, N, Nmul=3):
    N_tot = int(1.1 * N)
    H = bases.mk_fourier_basis(q, int(N_tot))[:, (N_tot - N):]
    A, B = reconstruct_lti(H, dampen="erasure")
    return bases.mk_lti_basis(A, B, N, N * Nmul)

# LDN LTI
# Mod. Fourier LTI
# Fourier

# Cosine
# DLOP
# Haar

basis_funs = [
    mk_ldn_basis,
    mk_mod_fourier_basis,
    bases.mk_fourier_basis,
]

basis_names = [
    "LDN", "Mod. Fourier", "Other"
]

fig, axs = plt.subplots(1, 3, figsize=(7.5, 1.25), gridspec_kw={
    "hspace": 1.0,
    "wspace": 0.5,
}, squeeze=False)

for i_row in range(axs.shape[0]):
    for i_col in range(axs.shape[1]):
        i = i_row * axs.shape[1] + i_col
        ax = axs[i_row, i_col]

        if i != 1:
            qs = np.arange(1, 65, 1)
        else:
            qs = np.arange(1, 65, 2) # Mod. Fourier doesn't like even q
        Sigmas = np.zeros(qs.shape)
        for j, q in tqdm.tqdm(enumerate(qs)):
            H = basis_funs[i](q, 1024)
            _, S, _ = np.linalg.svd(H)
            Sigmas[j] = np.sum(S) / np.max(S)

        a, _ = np.polyfit(qs, Sigmas, deg=1)
        if i == 2:
            utils.annotate(ax, 40, a * 40 - 2, 50, a * 40 - 2 - 10 / a, f"$\\Sigma = q$")
        else:
            utils.annotate(ax, 40, a * 40 - 3, 50, a * 40 - 3 - 10 / a, f"$\\Sigma \\approx {a:0.2f} q$")

            ax.plot(qs, qs, 'k--', lw=0.5)

        if i != 1:
            ax.plot(qs, Sigmas, 'k')
        else:
            ax.plot(qs, Sigmas, 'ko', markersize=2)


        if i < 2:
            ax.plot(0.95,
                    1.13,
                    'o',
                    color=utils.blues[0],
                    markeredgecolor='k',
                    markeredgewidth=0.7,
                    transform=ax.transAxes,
                    clip_on=False)
        else:
            ax.plot(0.875,
                    1.13,
                    'h',
                    color=utils.oranges[1],
                    markeredgecolor='k',
                    markeredgewidth=0.7,
                    transform=ax.transAxes,
                    clip_on=False)
            ax.plot(0.95,
                    1.13,
                    '^',
                    color=utils.yellows[1],
                    markeredgecolor='k',
                    markeredgewidth=0.7,
                    transform=ax.transAxes,
                    clip_on=False)


        ax.set_xlim(0, 64)
        ax.set_ylim(0, 64)
        ax.set_xticks(np.linspace(1, 64, 3, dtype=int))
        ax.set_xticks(np.linspace(1, 64, 5, dtype=int), minor=True)
        ax.set_yticks(np.linspace(1, 64, 3, dtype=int))
        ax.set_yticks(np.linspace(1, 64, 5, dtype=int), minor=True)

        ax.set_title(f"\\textbf{{{basis_names[i]}}}", x=0.4 if i == 2 else 0.5)
        ax.set_xlabel("Order $q$")
        ax.set_ylabel("Basis quality $\\Sigma$")

utils.save(fig)
