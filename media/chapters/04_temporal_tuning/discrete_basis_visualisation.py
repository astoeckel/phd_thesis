import dlop_ldn_function_bases as bases


def plot_basis(ax, B):
    s = np.percentile(np.abs(B), 95)
    ax.imshow(B,
              interpolation='nearest',
              extent=(-0.5, B.shape[1] - 0.5, B.shape[0] - 0.5, -0.5),
              vmin=-s,
              vmax=s,
              cmap='RdBu')
    xs = [int(x) if i == 0 or i == 4 else int(10 * round(x / 10)) for i, x, in enumerate(np.linspace(0, B.shape[1] - 1, 5))]
    ax.set_xticks(xs)

    ys = [int(y) if i == 0 or i == 4 else int(10 * round(y / 10)) for i, y, in enumerate(np.linspace(0, B.shape[0] - 1, 5))]
    ax.set_yticks(ys)
    ax.set_ylabel("Basis function $n$")
    ax.set_aspect(B.shape[1] / B.shape[0])
    ax.set_xlabel("Sample $k$", labelpad=3.0)


fig, axs = plt.subplots(1, 3, figsize=(7.5, 1.4))

q, N = 40, 40
F = bases.mk_fourier_basis(q, N)
C = bases.mk_cosine_basis(q, N)
P = bases.mk_dlop_basis(q, N)

plot_basis(axs[0], F)
axs[0].set_title("\\textbf{Fourier basis} $F_n(k; N)$")

plot_basis(axs[1], C)
axs[1].set_title("\\textbf{Cosine basis} $F_n(k; N)$")

plot_basis(axs[2], P)
axs[2].set_title("\\textbf{DLOP basis} $F_n(k; N)$")

for i in range(3):
    axs[i].text(-0.34,
                1.071,
                "\\textbf{{{}}}".format(chr(ord('A') + i)),
                size=12,
                transform=axs[i].transAxes)

utils.save(fig)

