import scipy.optimize

F = 96485.3321233
R = 8.31446261815324


def nernst(z, T, Xin, Xout):
    return -(R * (273.15 + T)) / (z * F) * np.log(Xin / Xout) * 1e3


def goldman(T, Ps, XIs, XOs):
    return ((R * (273.15 + T)) / F) * np.log(
        sum(Ps[i] * XOs[i] for i in range(3)) / sum(Ps[i] * XIs[i]
                                                    for i in range(3))) * 1e3


# Data from Kandel et al. Table 6-1
X1In, X1Out = 400, 20  # K+
X2In, X2Out = 50, 440  # Na+
X3In, X3Out = 560, 40  # Cl-

# Compute the reversal potentials
E1 = nernst(1, 20, X1In, X1Out)
E2 = nernst(1, 20, X2In, X2Out)
E3 = nernst(1, 20, X3In, X3Out)

# Some shorthands
Ps = np.linspace(0, 2, 100)
eqn1 = lambda p1, p2, p3: goldman(20.0, [p1, p2, p3], [X1In, X2In, X3In],
                                  [X1Out, X2Out, X3Out])
eqn2 = lambda g1, g2, g3: (g1 * E1 + g2 * E2 + g3 * E3) / (g1 + g2 + g3)

# Weighting
P1, P2, P3 = 1.0, 1.0, 1.0

# Find g1, g2, g3
def err(G):
    (g1, g2), g3 = G, 1.0
    return (np.sum(np.square(eqn1(0, P2, P3) - eqn2(0, g2, g3))) +
            np.sum(np.square(eqn1(P1, 0, P3) - eqn2(g1, 0, g3))) +
            np.sum(np.square(eqn1(P1, P2, 0) - eqn2(g1, g2, 0))) +
            3 * np.sum(np.square(eqn1(P1, P2, P3) - eqn2(g1, g2, g3))))


g1, g2, g3 = 1.0, 1.0, 1.0
g1, g2 = scipy.optimize.minimize(err, (g1, g2,), bounds=((0.01, None), (0.01, None))).x

fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.1))

#cs = ['#204a87ff', '#f57900ff', '#4e9a06ff']
cs = ['k'] * 3
lw = 0.75


axs[0].plot(Ps, eqn1(P1 * Ps, P2, P3), color=cs[0], linewidth=lw)
axs[1].plot(Ps, eqn1(P1, P2 * Ps, P3), color=cs[1], linewidth=lw)
axs[2].plot(Ps, eqn1(P1, P2, P3 * Ps), color=cs[2], linewidth=lw)

axs[0].plot(Ps, eqn2(g1 * Ps, g2, g3), color=cs[0], linestyle='--', linewidth=lw)
axs[1].plot(Ps, eqn2(g1, g2 * Ps, g3), color=cs[1], linestyle='--', linewidth=lw)
axs[2].plot(Ps, eqn2(g1, g2, g3 * Ps), color=cs[2], linestyle='--', linewidth=lw)

axs[0].plot([0], [eqn1(0, P2, P3)], 'o', markersize=3, color=cs[0])
axs[0].plot([0], [eqn2(0, g2, g3)], 'o', markersize=3, fillstyle='none', color=cs[0])
axs[0].plot([1], [eqn1(P1, P2, P3)], 'o', markersize=3, color=cs[0])
axs[0].plot([1], [eqn2(g1, g2, g3)], 'o', markersize=3, fillstyle='none', color=cs[0])

axs[1].plot([0], [eqn1(P1, 0, P3)], 'o', markersize=3, color=cs[1])
axs[1].plot([0], [eqn2(g1, 0, g3)], 'o', markersize=3, fillstyle='none', color=cs[1])
axs[1].plot([1], [eqn1(P1, P2, P3)], 'o', markersize=3, color=cs[1])
axs[1].plot([1], [eqn2(g1, g2, g3)], 'o', markersize=3, fillstyle='none', color=cs[1])

axs[2].plot([0], [eqn1(P1, P2, 0)], 'o', markersize=3, color=cs[2])
axs[2].plot([0], [eqn2(g1, g2, 0)], 'o', markersize=3, fillstyle='none', color=cs[2])
axs[2].plot([1], [eqn1(P1, P2, P3)], 'o', markersize=3, color=cs[2])
axs[2].plot([1], [eqn2(g1, g2, g3)], 'o', markersize=3, fillstyle='none', color=cs[2])

utils.annotate(axs[0], 1.05, eqn1(P1, P2, P3) + 5.0, 1.25, eqn1(P1, P2, P3) + 15.0, "$P_{\mathrm{K}^+} = 1 : P_{\mathrm{Na}^+} = 1 : P_{\mathrm{Cl}^-} = 1$", va="bottom", ha="center")
utils.annotate(axs[0], 0.05, eqn1(0, P2, P3) - 5.0, 1.0, eqn1(0, P2, P3) - 35.0, "$P_{\mathrm{K}^+} = 0 : P_{\mathrm{Na}^+} = 1 : P_{\mathrm{Cl}^-} = 1$", va="top", ha="center")

utils.annotate(axs[1], 1.0, eqn1(P1, P2, P3) + 5.0, 1.0, eqn1(P1, P2, P3) + 15.0, "$P_{\mathrm{K}^+} = 1 : P_{\mathrm{Na}^+} = 1 : P_{\mathrm{Cl}^-} = 1$", va="bottom", ha="center")
utils.annotate(axs[1], 0.05, eqn1(P1, 0, P3) - 0.0, 0.15, eqn1(P1, 0, P3) - 0.0, "$P_{\mathrm{K}^+} = 1 : P_{\mathrm{Na}^+} = 0 : P_{\mathrm{Cl}^-} = 1$", va="center", ha="left")

utils.annotate(axs[2], 1.05, eqn1(P1, P2, P3) + 5.0, 1.2, eqn1(P1, P2, P3) + 20.0, "$P_{\mathrm{K}^+} = 1 : P_{\mathrm{Na}^+} = 1 : P_{\mathrm{Cl}^-} = 1$", va="bottom", ha="center")
utils.annotate(axs[2], 0.05, eqn1(P1, P2, 0) - 5.0, 1.0, eqn1(P1, P2, 0) - 45.0, "$P_{\mathrm{K}^+} = 1 : P_{\mathrm{Na}^+} = 1 : P_{\mathrm{Cl}^-} = 0$", va="top", ha="center")

axs[0].set_title('$\\mathrm{K}^+$ Channel')
axs[1].set_title('$\\mathrm{Na}^+$ Channel')
axs[2].set_title('$\\mathrm{Cl}^-$ Channel')

l1 = mpl.lines.Line2D([], [], color='k', linestyle='-')
l2 = mpl.lines.Line2D([], [], color='k', linestyle='--')

fig.legend([l1, l2], ['Goldman equation', 'Circuit model'], ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.15))

for i, ax in enumerate(axs.flat):
    ax.set_ylim(-80, 20)
    ax.set_xticks(np.arange(0, 2.1, 1))
    ax.set_xticks(np.arange(0, 2.1, 0.25), minor=True)
    ax.set_yticks(np.arange(-80, 20.1, 20))
    ax.set_yticks(np.arange(-80, 20.1, 10), minor=True)
    ax.set_xlabel('Permeability/conductance factor')
    ax.plot()
    if i == 0:
        ax.set_ylabel('Equilibrium potential $E$ (mV)')
    else:
        ax.set_yticklabels([])

#ax.legend([l1, l2], ['$\\mathrm{K}^+$', '$\\mathrm{Na}^+$'], ncol=2)
#ax.plot(Ps, eqn1(P1, P2, P3 * Ps))

#ax.set_prop_cycle(None)

#ax.plot(Ps, eqn2(g1 * Ps, g2, g3), '--')
#ax.plot(Ps, eqn2(g1, g2 * Ps, g3), '--')
#ax.plot(Ps, eqn2(g1, g2, g3 * Ps), '--')

utils.save(fig)

