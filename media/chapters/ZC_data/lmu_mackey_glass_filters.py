import lmu_utils

BASES = [
    (lmu_utils.mk_ext_ldn_basis, "ldn"),  # 0
    (lmu_utils.mk_mod_fourier_basis, "mod_fourier"),  # 1
    (lmu_utils.mk_ext_fourier_basis, "fourier"),  # 3
    (lmu_utils.mk_ext_cosine_basis, "cosine"),  # 4
    (lmu_utils.mk_ext_haar_basis, "haar"),  # 5
    (lmu_utils.mk_ext_dlop_basis, "dlop"),  # 2
]

LABEL_MAP = {
    "ldn": "LDN",
    "mod_fourier": "Mod.~Fourier",
    "dlop": "DLOP",
    "fourier": "Fourier",
    "cosine": "Cosine",
    "haar": "Haar",
}

QS = [5, 9, 17]

Nm = 3

fig, axs = plt.subplots(len(BASES),
                        len(QS),
                        figsize=(8.0, 3.5),
                        gridspec_kw={
                            "wspace": 0.1,
                            "hspace": 0.75,
                        })

for i, (basis_ctor, name) in enumerate(BASES):
    for j, q in enumerate(QS):
        H = basis_ctor(q, q, Nm)[:, ::-1]
        for k in range(q):
            if i < 2:
                axs[i, j].plot(np.arange(1, q * Nm + 1),
                               H[k],
                               '-s',
                               zorder=-k,
                               lw=0.7,
                               markersize=1,
                               clip_on=False)
            else:
                # Save space in the resulting PDF file. Only plot the first
                # half of the filter.
                axs[i, j].plot(np.arange(1, q + 2),
                               H[k, :(q + 1)],
                               '-s',
                               zorder=-k,
                               lw=0.7,
                               markersize=1,
                               clip_on=False)
                if k == 0:
                    axs[i, j].plot(np.arange(q, q * Nm + 1),
                                   H[k, (q - 1):],
                                   '-s',
                                   zorder=-k,
                                   lw=0.7,
                                   markersize=1,
                                   clip_on=False,
                                   color='tab:blue')

        axs[i, j].set_xlim(1, q * Nm)
        axs[i, j].spines["left"].set_visible(False)
        axs[i, j].set_yticks([])
        if i + 1 < len(BASES):
            axs[i, j].set_xticklabels([])
        else:
            axs[i, j].set_xlabel("Sample index $t$")

        axs[i, j].axvline(q, color='k', linestyle='--')

        axs[i, j].text(1.0,
                       1.125,
                       f"$q = {q}$",
                       ha="right",
                       va="baseline",
                       transform=axs[i, j].transAxes,
                       size=8)
        if j == 0:
            axs[i, j].set_title(f"\\textbf{{{LABEL_MAP[name]}}}",
                                x=-0.01,
                                y=0.825,
                                ha="left")

utils.save(fig)

