import nengo


def sim(A, B, us, tau=100e-3, dt=1e-3):
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    # Compute the derivative
    dus = np.concatenate(((0, ), (us[1:] - us[:-1]) / dt))
    usp = np.array((us, dus)).T

    AH = tau * A + np.eye(A.shape[0])
    BH = tau * B

    with nengo.Network() as model:
        nd = nengo.Node(lambda t: usp[int(np.clip(t / dt, 0, len(usp) - 1))])
        ens = nengo.Ensemble(n_neurons=1000, dimensions=1)
        nengo.Connection(nd, ens, transform=BH, synapse=tau)
        nengo.Connection(ens, ens, transform=AH, synapse=tau)

        p = nengo.Probe(ens, synapse=0.005)

    T = len(us) * dt
    with nengo.Simulator(model, dt=dt) as sim:
        sim.run(T)

    return sim.trange(), sim.data[p]


np.random.seed(489789)

dt, tau = 1e-3, 100e-3
T = 1.0
N = int(T / dt + 1e-9)
ts = np.arange(0, N) * dt
us = sum(
    np.sin(2.0 * np.pi * ts * i) * [0.0, 0.7, 0.2, -0.3][i] for i in range(4))

us = np.zeros(N)
us[ts > 0.25] = 1.0
us[ts > 0.75] = -1.0

fig, axs = plt.subplots(1, 3, figsize=(6, 1.5))

axs[0].plot(ts, us, 'k--', label='$u(t)$', linewidth=0.7)
axs[0].plot(*sim(-1.0 / tau, [1.0 / tau, 0.0], us, tau=tau),
            label='$\\hat x(t)$',
            color=utils.blues[0])
axs[0].set_title(
    "\\textbf{Without recurrence}\n$\\tau \\dot{x}(t) = u(t) - x(t)$")

axs[1].plot(ts, us, 'k--', label='$u(t)$', linewidth=0.7)
axs[1].plot(*sim(0.0, [1.0, 0.0], us, tau=tau),
            label='$\\hat x(t)$',
            color=utils.blues[0])
axs[1].set_title("\\textbf{Integrator}\n$\\dot{x}(t) = u(t)$")

axs[2].plot(ts, us, 'k--', label='$u(t)$', linewidth=0.7)
axs[2].plot(*sim(0.0, [0.0, 1.0], us, tau=tau),
            label='$\\hat x(t)$',
            color=utils.blues[0])
axs[2].set_title("\\textbf{Pass-through}\n$\\dot{x}(t) = \\dot u(t)$")

for i in range(3):
    axs[i].set_xlim(0.0, T)
    axs[i].set_xticks(np.linspace(0, T, 5), minor=True)
    axs[i].set_yticks(np.linspace(-1, 1, 5), minor=True)
    axs[i].set_xlabel("Time $t$")
    axs[i].text(-0.125 if i == 0 else 0.0,
                1.1875,
                "\\textbf{{{}}}".format(chr(ord('B') + i)),
                size=12,
                ha="left",
                va="baseline",
                transform=axs[i].transAxes)
    if i > 0:
        axs[i].set_yticklabels([])

axs[2].legend(loc="upper right", bbox_to_anchor=(0.7, 1.62), ncol=2, handlelength=1.0, handletextpad=0.5, columnspacing=2.0)


utils.save(fig)

