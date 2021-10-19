import h5py
import dlop_ldn_function_bases as bases

with h5py.File(utils.datafile("ldn_integrator_analysis.h5"), 'r') as f:
    qs = f["qs"][()]
    Ns = f["Ns"][()]
    Es = f["errs"][()][1]
    lambdas_rk2 = f["lambdas"][()][2]
    lambdas_rk4 = f["lambdas"][()][3]
    sigmas_ref = f["sigmas"][()][0]
    sigmas_rk2 = f["sigmas"][()][2]
    sigmas_rk4 = f["sigmas"][()][3]


def do_plot(ax, cax, sigmas, lambdas, usx=90):
    # Hatches
    for i in range(50):
        q0, N0 = 2 * i, 2 * i
        ax.plot([20 * 2 * i, N0], [0, q0], 'k', linewidth=0.1, zorder=-10)

    sigmas_cpy = np.copy(sigmas)
    sigmas[lambdas > 1.0] = np.nan

    CS = ax.contourf(
        Ns,
        qs,
        100 * (sigmas / sigmas_ref),
        levels=np.linspace(0, 120, 13),
        cmap="inferno_r",
    )
    ax.contour(Ns, qs, lambdas, levels=[1], widths=[1.5], colors=['k'])
    ax.contour(Ns, qs, sigmas / sigmas_ref, levels=[0.99], widths=[1.5], colors=['white'])
    ax.text(825, 6, "Rel. $\\Sigma > 99\\%$", color="white", va="center", ha="center")

    for C in CS.collections:
        C.set_edgecolor("face")

    ax.plot(Ns[:800], np.sqrt(Ns[:800]), 'k--', lw=0.7, color="white")
    ax.plot(Ns[910:], np.sqrt(Ns[910:]), 'k--', lw=0.7, color="white")
    ax.text(600, 20, "$\\sqrt{N}$", color="white")

    ax.plot(qs, qs, 'k--', linewidth=0.5)

    ax.set_xlim(1, 1000)
    ax.set_ylim(1, 50)
    ax.set_xticks(np.linspace(1, 1000, 6, dtype=np.int))
    ax.set_yticks([1, 10, 20, 30, 40, 50])
    ax.set_xlabel('Sample count $N$')
    ax.set_ylabel('State dimensions $q$')

    utils.annotate(ax, usx, 45, 175, 40, "Unstable")

    ax.text(830, 29.5, "Stable", va="center", ha="center", color="white")

    utils.outside_ticks(ax)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_position(('outward', 2.5))

    cb = plt.colorbar(CS, cax=cax, orientation='vertical')
    for spine in ['left', 'bottom', 'right', 'top']:
        cax.spines[spine].set_visible(False)
    cax.set_ylabel('$\\Sigma$ relative to ZOH')
    cb.outline.set_visible(False)
    cax.set_yticklabels(["{:d}\\%".format(int(d)) for d in np.linspace(0, 120, 7)])


fig, axs = plt.subplots(1,
                        5,
                        figsize=(6.9, 2.0),
                        gridspec_kw={
                            "width_ratios": [1.0, 0.03, 0.4, 1.0, 0.03],
                            "wspace": 0.1,
                        })
utils.remove_frame(axs[2])

do_plot(axs[0], axs[1], sigmas_rk2, lambdas_rk2)
#axs[3].plot(Ns, 3.5 * np.sqrt(Ns))
axs[0].text(185, 32, "$\\approx \\frac{7}{2} \sqrt{N}$", va="center", ha="center")

do_plot(axs[3], axs[4], sigmas_rk4, lambdas_rk4, usx=75)
axs[3].text(185, 32, "$\\approx \\frac{9}{2} \sqrt{N}$", va="center", ha="center")

axs[0].set_title("\\textbf{LDN Midpoint} ($\\mat{\\tilde A}^{\\{2\\}}$)")
axs[0].text(-0.21,
        1.055,
        "\\textbf{A}",
        size=12,
        va="baseline",
        ha="left",
        transform=axs[0].transAxes)

axs[3].set_title("\\textbf{LDN Fourth-Order Runge Kutta} ($\\mat{\\tilde A}^{\\{4\\}}$)")
axs[3].text(-0.21,
        1.055,
        "\\textbf{B}",
        size=12,
        va="baseline",
        ha="left",
        transform=axs[3].transAxes)


utils.save(fig)

