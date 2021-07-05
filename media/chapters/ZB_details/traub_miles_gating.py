import scipy.optimize


def plot_equilibrium_point(
    ax,
    alpha,
    beta,
    Vth=-50.0,
    Vmin=-50.0,
    Vmax=50.0,
    Zmin=0.0,
    Zmax=1.0,
    resQ=10,  # 10 instead of 11 sampling to prevent NaNs...
    resE=100):  # 100 instead of 101 to prevent NaNs...

    VsQ = np.linspace(Vmin, Vmax, resQ)
    ZsQ = np.linspace(Zmin, Zmax, resQ)
    VssQ, ZssQ = np.meshgrid(VsQ, ZsQ)

    dZ = lambda V, Z: (alpha(V - Vth) * (1.0 - Z) - beta(V - Vth) * Z)
    dZss = dZ(VssQ, ZssQ)
    ax.quiver(VsQ, ZsQ, np.zeros_like(dZss), dZss)

    # Plot the equilibrium point
    Vs = np.linspace(Vmin, Vmax, resE)
    Zs = np.zeros(resE)
    for i, V in enumerate(Vs):
        Zs[i] = scipy.optimize.root_scalar(lambda Z: dZ(V, Z), x0=0.0,
                                           x1=1.0).root
    ax.plot(Vs, Zs, color='k')


def plot_time_constants(ax,
                        alpha,
                        beta,
                        Vth=-50.0,
                        Vmin=-50.0,
                        Vmax=50.0,
                        Zmin=0.0,
                        Zmax=1.0,
                        resE=100):  # 100 instead of 101 to prevent NaNs...
    Vs = np.linspace(Vmin, Vmax, resE)
    taus = alpha(Vs) + beta(Vs)
    ax.plot(Vs, taus, color='k')
    ax.set_yscale('log')
    ax.set_ylim(1e-1, 100.0)
    ax.set_xlim(Vmin * 1.1, Vmax * 1.1)
    ax.set_xticks(np.arange(Vmin, Vmax + 0.1, 50.0))
    ax.set_xticks(np.arange(Vmin, Vmax + 0.1, 12.5), minor=True)

    ax.plot(0.1, 0.9175, "k^", markersize=4, transform=ax.transAxes)
    ax.plot(0.1, 0.09, "kv", markersize=4, transform=ax.transAxes)
    ax.text(0.15, 0.95, "Faster", va="top", ha="left", transform=ax.transAxes)
    ax.text(0.15,
            0.05,
            "Slower",
            va="bottom",
            ha="left",
            transform=ax.transAxes)


alpha_m = lambda V: 0.32 * (13.0 - V) / (np.exp((13.0 - V) / 4.0) - 1.0)
beta_m = lambda V: 0.28 * (V - 40.0) / (np.exp((V - 40.0) / 5.0) - 1.0)

alpha_h = lambda V: 0.128 * (np.exp((17.0 - V) / 18.0))
beta_h = lambda V: 4 / (np.exp((40 - V) / 5) + 1)

alpha_n = lambda V: 0.032 * (15.0 - V) / (np.exp((15.0 - V) / 5.0) - 1.0)
beta_n = lambda V: 0.5 * np.exp((10 - V) / 40)

fig, axs = plt.subplots(2,
                        3,
                        figsize=(7.45, 5.0),
                        gridspec_kw={
                            "wspace": 0.3,
                            "hspace": 0.5,
                        })

(ax1, ax2, ax3) = axs[0]

plot_equilibrium_point(ax1, alpha_m, beta_m)
ax1.set_xlabel("Membrane potential $v$ (mV)")
ax1.set_ylabel("Equilibrium $m_\infty(v)$")
ax1.set_title("Gating variable $m$ (depolarisation)")

plot_equilibrium_point(ax2, alpha_h, beta_h)
ax2.set_xlabel("Membrane potential $v$ (mV)")
ax2.set_ylabel("Equilibrium $h_\infty(v)$")
ax2.set_title("Gating variable $h$ (gating for $m$)")

plot_equilibrium_point(ax3, alpha_n, beta_n)
ax3.set_xlabel("Membrane potential $v$ (mV)")
ax3.set_ylabel("Equilibrium $n_\infty(v)$")
ax3.set_title("Gating variable $n$ (repolarisation)")

fig.text(-0.275,
         1.029,
         "\\textbf{A}",
         fontdict={"size": 12},
         transform=ax1.transAxes,
         va="bottom",
         ha="left")

(ax1, ax2, ax3) = axs[1]

plot_time_constants(ax1, alpha_m, beta_m)
ax1.set_xlabel("Membrane potential $v$ (mV)")
ax1.set_ylabel("Time constant $\\tau_m(v)^{-1}$")
ax1.set_title("Gating variable $m$ (depolarisation)")

plot_time_constants(ax2, alpha_h, beta_h)
ax2.set_xlabel("Membrane potential $v$ (mV)")
ax2.set_ylabel("Time constant $\\tau_h(v)^{-1}$")
ax2.set_title("Gating variable $h$ (shutoff for $m$)")

plot_time_constants(ax3, alpha_n, beta_n)
ax3.set_xlabel("Membrane potential $v$ (mV)")
ax3.set_ylabel("Time constant $\\tau_n(v)^{-1}$")
ax3.set_title("Gating variable $n$ (repolarisation)")

fig.text(-0.275,
         1.029,
         "\\textbf{B}",
         fontdict={"size": 12},
         transform=ax1.transAxes,
         va="bottom",
         ha="left")

utils.save(fig)

