import h5py
import dlop_ldn_function_bases as bases

with h5py.File(utils.datafile("ldn_integrator_analysis.h5"), 'r') as f:
    qs = f["qs"][()]
    Ns = f["Ns"][()]
    Es = f["errs"][()][1]
    lambdas = f["lambdas"][()][1]
    sigmas_ref = f["sigmas"][()][0]
    sigmas = f["sigmas"][()][1]


def plot_markers(ax, Ns=np.geomspace(20, 1000, 6, dtype=int), q=20):
    for i, N in enumerate(Ns):
        ax.plot(N,
                q,
                'o',
                color=utils.grays[0],
                markersize=8,
                clip_on=False,
                zorder=1000)
        ax.text(N,
                q - 0.25,
                f"\\textbf{{{i + 1}}}",
                color="white",
                ha="center",
                va="center",
                clip_on=False,
                zorder=1001)


fig, axs = plt.subplots(1,
                        5,
                        figsize=(6.9, 2.0),
                        gridspec_kw={
                            "width_ratios": [1.0, 0.03, 0.4, 1.0, 0.03],
                            "wspace": 0.1,
                        })
utils.remove_frame(axs[2])

ax, cax, _, _, _ = axs

# Draw the hatched background
for i in range(50):
    q0, N0 = 2 * i, 2 * i
    ax.plot([20 * 2 * i, N0], [0, q0], 'k', linewidth=0.1, zorder=-10)
CS = ax.contourf(Ns,
                 qs,
                 Es,
                 vmin=0.0,
                 vmax=1.0,
                 levels=np.linspace(0, 1, 11),
                 cmap='viridis')
for C in CS.collections:
    C.set_edgecolor("face")
ax.contour(Ns, qs, Es, levels=[1.00], colors=['k'])
ax.contour(Ns, qs, Es, levels=[0.1], colors=['white'])
ax.plot(Ns[:670], np.sqrt(Ns[:670]), '--', color='white', lw=0.7)
ax.plot(Ns[970:], np.sqrt(Ns[970:]), '--', color='white', lw=0.7)
ax.text(600, 20, "$\\sqrt{N}$", color="white")
ax.plot(qs, qs, 'k--', linewidth=0.5)
ax.text(210,
        40,
        'NRMSE $E > 1$\n(high error)',
        va='center',
        ha='center',
        fontsize=8,
        bbox={
            "color": "white",
            "pad": 0.5,
            "alpha": 1.0,
        })
ax.text(650,
        7,
        'NRMSE $E < 0.1$ (low error;\ncan safely use Euler)',
        va='center',
        ha='center',
        fontsize=8,
        color='white')
ax.text(750,
        30,
        'NRMSE $0.1 \\leq E < 1$\n(moderate error)',
        va='center',
        ha='center',
        fontsize=8,
        color='white')
ax.text(900,
        14.5,
        '$\\approx \\frac{3}{5}\\sqrt{N}$',
        va='center',
        ha='center',
        fontsize=8,
        color='white')
ax.text(490,
        44,
        '$\\approx \\frac{17}{10} \\sqrt{N}$',
        va='center',
        ha='center',
        fontsize=8,
        color='black',
        bbox={
            "color": "white",
            "pad": 0.25,
            "alpha": 1.0,
        })
ax.set_xlim(1, 1000)
ax.set_ylim(1, 50)
ax.set_xticks(np.linspace(1, 1000, 6, dtype=np.int))
ax.set_yticks([1, 10, 20, 30, 40, 50])
ax.set_xlabel('Sample count $N$')
ax.set_ylabel('State dimensions $q$')

ax.set_title("\\textbf{LDN Euler impulse response error}")
ax.text(-0.21,
        1.055,
        "\\textbf{A}",
        size=12,
        va="baseline",
        ha="left",
        transform=ax.transAxes)

plot_markers(ax)

utils.outside_ticks(ax)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_position(('outward', 2.5))

cb = plt.colorbar(CS, cax=cax, orientation='vertical')
for spine in ['left', 'bottom', 'right', 'top']:
    cax.spines[spine].set_visible(False)
cax.set_ylabel('NRMSE $E$')
cb.outline.set_visible(False)

_, _, _, ax, cax = axs

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

for C in CS.collections:
    C.set_edgecolor("face")

plot_markers(ax)

ax.plot(Ns[:800], np.sqrt(Ns[:800]), 'k--', lw=0.7, color="white")
ax.plot(Ns[910:], np.sqrt(Ns[910:]), 'k--', lw=0.7, color="white")
ax.text(600, 21, "$\\sqrt{N}$", color="white")

#ax.plot(Ns, 0.6 * np.sqrt(Ns), color='white', lw=1.5)
#ax.text(900,
#        14.5,
#        '$\\approx \\frac{3}{5}\\sqrt{N}$',
#        va='center',
#        ha='center',
#        fontsize=8,
#        color='white')

ax.plot(qs, qs, 'k--', linewidth=0.5)

#ax.plot(Ns, 2.0 * np.power(Ns, 0.5), 'k-', lw=1.7)

ax.set_xlim(1, 1000)
ax.set_ylim(1, 50)
ax.set_xticks(np.linspace(1, 1000, 6, dtype=np.int))
ax.set_yticks([1, 10, 20, 30, 40, 50])
ax.set_xlabel('Sample count $N$')
ax.set_ylabel('State dimensions $q$')

ax.text(175,
        40,
        "Unstable",
        va="center",
        ha="center",
        bbox={
            "color": "white",
            "pad": 0.25,
        })
ax.text(830, 29.5, "Stable", va="center", ha="center", color="white")
ax.text(420, 35, "$\\approx 2 \sqrt{N}$", va="center", ha="center")

utils.outside_ticks(ax)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_position(('outward', 2.5))

cb = plt.colorbar(CS, cax=cax, orientation='vertical')
for spine in ['left', 'bottom', 'right', 'top']:
    cax.spines[spine].set_visible(False)
cax.set_ylabel('$\\Sigma$ relative to ZOH')
cb.outline.set_visible(False)
cax.set_yticklabels(["{:d}\\%".format(int(d)) for d in np.linspace(0, 120, 7)])

#levels = np.linspace(-3, 0, 7)
#vmin, vmax = np.min(levels), np.max(levels)
#cax.pcolormesh([0, 1],
#               np.power(10.0, levels),
#               np.array((levels, levels)).T,
#               shading='flat',
#               vmin=vmin,
#               vmax=vmax,
#               cmap='Oranges')
#cax.set_yscale('log')
#cax.set_ylim(1e-3, 1e0)
#cax.yaxis.set_ticks_position('right')
#cax.yaxis.set_label_position('right')
#for spine in ['left', 'bottom', 'right', 'top']:
#    cax.spines[spine].set_visible(False)
#cax.set_xticks([])
#cax.set_xlabel("")
#cax.set_ylabel("Eigenvalue $|\\lambda_1|$")

ax.set_title("\\textbf{LDN Asymptotic stability and basis quality}")
ax.text(-0.21,
        1.055,
        "\\textbf{B}",
        size=12,
        va="baseline",
        ha="left",
        transform=ax.transAxes)

utils.save(fig)

