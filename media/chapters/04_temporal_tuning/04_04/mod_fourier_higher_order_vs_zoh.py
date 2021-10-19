import h5py
import dlop_ldn_function_bases as bases

with h5py.File(utils.datafile("mod_fourier_integrator_analysis.h5"), 'r') as f:
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
    for i in range(100):
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

    ax.contour(Ns,
               qs,
               100 * (sigmas / sigmas_ref),
               levels=[100],
               widths=[1.5],
               colors=['white'])
    ax.contour(Ns, qs, lambdas, levels=[1], widths=[1.5], colors=['k'])

    for C in CS.collections:
        C.set_edgecolor("face")

    ax.plot(2.0 * qs, 2.0 * qs, 'k--', linewidth=0.5)

    ax.set_xlim(1, 1000)
    ax.set_ylim(1, 100)
    ax.set_xticks(np.linspace(1, 1000, 6, dtype=np.int))
    ax.set_yticks([1, 20, 40, 60, 80, 100])
    ax.set_xlabel('Sample count $N$')
    ax.set_ylabel('State dimensions $q$')

    if not usx is None:
        utils.annotate(ax, 225, 70, 225, 70, "Unstable")

    ax.text(830, 29.5, "Stable", va="center", ha="center", color="white")

    utils.outside_ticks(ax)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_position(('outward', 2.5))

    cb = plt.colorbar(CS, cax=cax, orientation='vertical')
    for spine in ['left', 'bottom', 'right', 'top']:
        cax.spines[spine].set_visible(False)
    cax.set_ylabel('$\\Sigma$ relative to ZOH')
    cb.outline.set_visible(False)
    cax.set_yticklabels(
        ["{:d}\\%".format(int(d)) for d in np.linspace(0, 120, 7)])


fig, axs = plt.subplots(1,
                        5,
                        figsize=(6.9, 2.0),
                        gridspec_kw={
                            "width_ratios": [1.0, 0.03, 0.4, 1.0, 0.03],
                            "wspace": 0.1,
                        })
utils.remove_frame(axs[2])

do_plot(axs[0], axs[1], sigmas_rk2, lambdas_rk2)
#axs[0].plot(Ns, 1.33 * np.power(Ns, 2.0 / 3.0))
axs[0].text(600,
            80,
            "$\\approx \\frac{4}{3} \sqrt[3]{N^2}$",
            va="center",
            ha="center")

#axs[0].plot(Ns, 0.66 * np.power(Ns, 2.0 / 3.0))
axs[0].text(400,
            20,
            "$\\approx \\frac{2}{3} \sqrt[3]{N^2}$",
            va="center",
            ha="center",
            color='white')

do_plot(axs[3], axs[4], sigmas_rk4, lambdas_rk4, usx=None)
axs[3].text(130, 80, "$\\approx N$", va="center", ha="center")

#axs[3].plot(Ns, 1.0 * np.power(Ns, 0.75))
axs[3].text(400,
            60,
            "$\\approx \sqrt[4]{N^3}$",
            va="center",
            ha="center",
            color="white")

axs[0].set_title(
    "\\textbf{Modified Fourier Midpoint} ($\\mat{\\tilde A}_\\mathrm{F}^{\{2\}}$)"
)
axs[0].text(-0.21,
            1.055,
            "\\textbf{A}",
            size=12,
            va="baseline",
            ha="left",
            transform=axs[0].transAxes)

axs[3].set_title(
    "\\textbf{Modified Fourier Runge-Kutta} ($\\mat{\\tilde A}_\\mathrm{F}^{\\{4\\}}$)"
)
axs[3].text(-0.21,
            1.055,
            "\\textbf{B}",
            size=12,
            va="baseline",
            ha="left",
            transform=axs[3].transAxes)

q = 101
Ns = np.geomspace(200, 1000, 4, dtype=int)
for i, N in enumerate(Ns):
    axs[3].plot(N, q, 'o', color=utils.grays[0], markersize=8, clip_on=False)
    axs[3].text(N,
                q,
                f"\\textbf{{{i + 1}}}",
                color="white",
                ha="center",
                va="center",
                clip_on=False)

utils.save(fig)
