def mk_cont_fourier_basis(n):
    if n == 0:
        return lambda x: np.ones_like(x)
    elif n % 2 == 1:
        return lambda x: np.sin(((n + 1) // 2) * 2.0 * np.pi * x) * np.sqrt(2)
    elif n % 2 == 0:
        return lambda x: np.cos((n // 2) * 2.0 * np.pi * x) * np.sqrt(2)


def mk_cont_cosine_basis(n):
    if n == 0:
        return lambda x: np.ones_like(x)
    else:
        return lambda x: np.cos(n * np.pi * x) * np.sqrt(2)


def mk_cont_legendre_basis(n):
    p = np.polynomial.Legendre([0] * n + [1], [0, 1])
    return lambda x: np.sqrt((2 * n + 1)) * p(x)


def mk_cont_haar_basis(n):
    if n == 0:
        return lambda x: np.ones_like(x)
    else:
        phi = 2**int(np.log2(n))

        def w(x):
            xs = (phi * x - n + phi)
            return np.sqrt(phi) * np.logical_and(
                xs >= 0.0, xs <= 1.0) * (-np.sign(xs - 0.5))

        return w


fig, axs = plt.subplots(7,
                        3,
                        figsize=(7.75, 3.0),
                        sharex=True,
                        sharey=True,
                        gridspec_kw={
                            "wspace": 0.1,
                            "hspace": 0.75,
                        })

N, N_smpls = 7, 1001
for j, (basis, basis_name, basis_sym) in enumerate(
        zip((mk_cont_fourier_basis, mk_cont_cosine_basis,
             mk_cont_legendre_basis),
            ('Fourier series', 'Cosine series', 'Legendre polynomials'),
            ('f', 'c', '{\\tilde{p}}'))):
    xs = np.linspace(0, 1, N_smpls)
    for i, ax in enumerate(axs[:, j].flat):
        color = cm.get_cmap('viridis')(i / (N - 1)) * np.array((0.9, 0.9, 0.9, 1.0))
        #        color = cm.get_cmap('tab10')(i * 0.1)
        #color = 'k'
        ax.plot(xs, basis(i)(xs), color=color, linewidth=1.5, clip_on=False)
        ax.plot(1.1 * (xs - 0.05),
                np.zeros_like(xs),
                'k:',
                linewidth=0.5,
                zorder=-100,
                clip_on=False)
        ax.set_ylim(-2, 2)
        ax.set_xlim(0, 1)
        ax.set_title('${}_{{{}}}(x)$'.format(basis_sym, i), y=0.75)
        for spine in ['left', 'bottom', 'right', 'top']:
            ax.spines[spine].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot([0, 0], [-0.5, 0.5], 'k-', lw=0.5, clip_on=False)
        ax.plot([1, 1], [-0.5, 0.5], 'k-', lw=0.5, clip_on=False)
        if i == 0:
            ax.text(0.0,
                    2.0,
                    '\\textbf{{{}}}'.format(chr(ord('A') + j)),
                    fontsize=12,
                    va='baseline',
                    ha='left',
                    transform=ax.transAxes)
            ax.text(0.5,
                    2.0,
                    '\\textbf{{{}}}'.format(basis_name),
                    va="baseline",
                    ha="center",
                    transform=ax.transAxes)

utils.save(fig)

