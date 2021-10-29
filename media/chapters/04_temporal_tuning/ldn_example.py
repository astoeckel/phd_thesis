import scipy.linalg
import dlop_ldn_function_bases as bases


def eval_lti(A, B, ts):
    return np.array([scipy.linalg.expm(A * t) @ B for t in ts])


def matshow(ax, A):
    s = np.max(np.abs(A))
    ax.imshow(A,
              vmin=-s,
              vmax=s,
              cmap='RdBu',
              extent=(0.5, A.shape[1] + 0.5, A.shape[0] + 0.5, 0.5),
              interpolation='none')
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            color = "white" if (np.abs(A[i, j]) / s) > 0.75 else "k"
            ax.text(j + 1,
                    i + 1,
                    str(int(A[i, j])),
                    va="center",
                    ha="center",
                    size=8,
                    color=color)
    ax.set_xticks(np.arange(1, A.shape[1] + 1, 2))
    ax.set_yticks(np.arange(1, A.shape[0] + 1, 2))
    ax.set_xlabel("Column $j$")
    ax.set_ylabel("Row $i$")
    utils.outside_ticks(ax)


fig, axs = plt.subplots(1, 4, figsize=(7.55, 1.25), gridspec_kw={
    "wspace": 0.4,
    "width_ratios": [4, 5, 4, 5],
})

T, dt = 1.5, 1e-3
ts = np.linspace(0.0, T, 1000)

H = bases.mk_dlop_basis(6, 1000)
H /= H[:, 0][:, None]

A, B = bases.mk_leg_lti(6)

matshow(axs[0], A)

axs[1].plot(ts, eval_lti(A, B, ts))
axs[1].plot(np.linspace(1, 0, H.shape[1]), H.T, 'k:', lw=0.5)
axs[1].axvline(1.0, color='k', linestyle='--', linewidth=0.7)
axs[1].set_xlim(0, T)
axs[1].set_ylim(-1.2, 1.2)
axs[1].set_xlabel("Time $t$ (s)")
axs[1].set_ylabel("$\\mathfrak{e}_i(t)$")
axs[1].set_xticks(np.linspace(0, 1.5, 7), minor=True)

A, B = bases.mk_ldn_lti(6, rescale=True)

matshow(axs[2], A)

axs[3].plot(ts, eval_lti(A, B, ts))
axs[3].plot(np.linspace(1, 0, H.shape[1]), H.T, 'k:', lw=0.5)
axs[3].axvline(1.0, color='k', linestyle='--', linewidth=0.7)
axs[3].set_xlim(0, T)
axs[3].set_ylim(-1.2, 1.2)
axs[3].set_xlabel("Time $t$ (s)")
axs[3].set_ylabel("$\\mathfrak{e}_i(t)$")
axs[3].set_xticks(np.linspace(0, 1.5, 7), minor=True)

fig.text(0.085, 0.95, "\\textbf{A}", size=12, va="baseline", ha="left")
fig.text(0.5, 0.95, "\\textbf{B}", size=12, va="baseline", ha="left")
fig.text(0.3, 0.95, "\\textbf{Legendre system}", va="baseline", ha="center")
fig.text(0.725, 0.95, "\\textbf{Windowed Legendre system (LDN)}", va="baseline", ha="center")

fig.align_labels(axs)

utils.save(fig)

