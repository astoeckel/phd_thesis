import tqdm
import gen_2d_fun
import function_bases as bases
import scipy.optimize

N = 127
d = 10
REG = 1e-3

basis = bases.mk_dlop_basis(d, N)

A = basis * np.sqrt(0.5 * N)
A2D = np.einsum('ij,kl->ikjl', A, A)
A1D = np.concatenate((
    A2D[0, :].reshape(d, -1),
    A2D[:, 0].reshape(d, -1),
)).T
A2D_flat = A2D.reshape(d, d, N * N).T.reshape(N * N, d * d)

def lstsq(A, Y):
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n, m = A.shape
    d = Y.shape[1]
    ATA = A.T @ A + n * REG * np.eye(m)
    D = np.zeros((m, d))
    for i in range(d):
        D[:, i] = np.linalg.solve(ATA, A.T @ Y[:, i])
    return D


def solve_additive(basis, tar, rng):
    return basis @ lstsq(basis, tar)


def solve_multiplicative(basis, tar, rng):
    def f(w):
        w1, w2 = w.reshape(2, -1)
        return (basis @ w1) * (basis @ w2)

    def E(w):
        return np.mean(np.square(f(w) - tar)) + np.sqrt(REG) * np.mean(np.square(w))

    d = basis.shape[1] // 2
    Ds, Es = [[None] * 10 for _ in range(2)]
    for i in range(len(Ds)):
        w = rng.randn(4 * d)
        Ds[i] = scipy.optimize.minimize(E, w, tol=1e-3, method='BFGS').x
        Es[i] = E(Ds[i])
    return f(Ds[np.argmin(Es)])

fig = plt.figure(figsize=(6.7, 4.0))
gs1 = fig.add_gridspec(10, 6, left=0.05, top=0.95, bottom=0.0375, right=0.4, wspace=0.1, hspace=0.1)
gs2 = fig.add_gridspec(4, 4, left=0.5, top=0.9, bottom=0.05, right=0.95, wspace=0.45, hspace=0.75)

for i in range(gs1.nrows):
    for j in range(gs1.ncols):
        A = np.outer(basis[i], basis[j])
        A /= np.percentile(np.abs(A), 95)
        ax = fig.add_subplot(gs1[i, j])
        ax.imshow(A, cmap="viridis", vmin=-1.0, vmax=1.0, origin='lower', extent=[-1, 1, -1, 1])
        #ax.contour(A, vmin=-1.0, vmax=1.0, levels=np.linspace(-1, 1, 3), colors=['white'], linestyles=['--'], linewidths=[0.7])
        ax.set_aspect(1)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        utils.remove_frame(ax)

        if j == 0:
            ax.set_ylabel(f"$i = {i}$")
        if i == gs1.nrows - 1:
            ax.set_xlabel(f"$j = {j}$")

        if i == 0 and j == 0:
            ax.plot([0, gs1.ncols * (1.0 + gs1.wspace) + 0.3], [-0.05, -0.05], 'k:', clip_on=False, transform=ax.transAxes)
            ax.plot([1.085, 1.085], [1.0, -(gs1.nrows * (1.0 + gs1.hspace) - 1.1)], 'k:', clip_on=False, transform=ax.transAxes)

rhos = np.logspace(-1, 0.5, 4)
fs = [
    gen_2d_fun.gen_2d_fun(gen_2d_fun.mk_2d_flt(1.0/rhos[i], N), N, rng=np.random.RandomState(478287)) for i in range(4)
]

for i, j in tqdm.tqdm([(i, j) for i in range(4) for j in range(4)]):
    ax = fig.add_subplot(gs2[i, j])

    X = fs[j]
    rng = np.random.RandomState(47818)
    if i == 0:
        Y = X
    elif i == 1:
        Y = solve_additive(A1D, X.reshape(-1), rng=rng).reshape(N, N)
    elif i == 2:
        Y = solve_multiplicative(A1D, X.reshape(-1), rng=rng).reshape(N, N)
    elif i == 3:
        Y = solve_additive(A2D_flat, X.reshape(-1), rng=rng).reshape(N, N)

    xs = np.linspace(-1, 1, N)
    ax.imshow(Y, vmin=-2.0, vmax=2.0, cmap='inferno', extent=[-1, 1, -1, 1], origin='lower')
    ax.contour(xs, xs, Y, vmin=-2.0, vmax=2.0, levels=np.linspace(-2, 2, 5), colors=['white'], linestyles=['--'], linewidths=[0.7])

    if i != 0:
        E = np.sqrt(np.mean(np.square(X - Y)))
        ax.text(0.05, 0.95, "{:0.1f}\\%".format(E * 100), size=8, va="top", ha="left", transform=ax.transAxes, bbox={
            "color": "white",
            "pad": 0.01,
        })

    ax.set_aspect(1)

    ax.set_xticks(np.linspace(-1, 1, 3))
    ax.set_xticks(np.linspace(-1, 1, 5), minor=True)

    ax.set_yticks(np.linspace(-1, 1, 3))
    ax.set_yticks(np.linspace(-1, 1, 5), minor=True)

    if j != 0:
        ax.set_yticklabels([])
    if i != 3:
        ax.set_xticklabels([])
    if i == 0:
        ax.set_title("$\\rho^{{-1}} = {:0.2f}$".format(rhos[j]))

    utils.outside_ticks(ax)

fig.text(0.025, 0.96, "\\textbf{A}", size=12, va="bottom", ha="left", transform=fig.transFigure)
fig.text(0.5 * (gs1.left + gs1.right), 0.96, "\\textbf{Basis functions} (1D and 2D)", va="bottom", ha="center", transform=fig.transFigure)

for i in range(4):
    y = 0.96 if i == 0 else (0.93 - i * 0.24)
    fig.text(gs2.left - 0.0525, y, "\\textbf{{{}}}".format(chr(ord('B') + i)), size=12, va="bottom", ha="left", transform=fig.transFigure)
    fig.text(0.5 * (gs2.left + gs2.right), y, "\\textbf{{{}}} {}".format(*[
        ("Target functions", ""),
        ("Additive network", "$f_\mathrm{add}$ (1D basis; 20 DOF)"),
        ("Multiplicative network", "$f_\mathrm{den}$ (1D basis; 40 DOF)"),
        ("Additive network", "$f_\mathrm{mlp}$ (2D basis; 100 DOF)"),
    ][i]), va="bottom", ha="center", transform=fig.transFigure)

utils.save(fig)
