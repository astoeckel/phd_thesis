import nengo
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

np.random.seed(53781)

# Sample a different tau for each neuron
N = 100
taus = np.logspace(-1, 1, N)

# Sample the impulse response
T = 50.0
dt = 1e-3
ts = np.arange(0, T, dt)
N_smpls = len(ts)
F = np.exp(-ts[:, None] / taus[None, :])

# Sample characteristic delays
N_delays = 73
thetas = 5.0 * np.power(np.linspace(0.1, 1, N_delays),
                        3.0)  # Sparsify later delays

# Low-pass filters of order "theta". Can be decoded well from the low-pass filters
G = (ts[:, None]**thetas[None, :]) * np.exp(-ts[:, None])
G /= np.sum(G, axis=0)[None, :]

# Compute the linear operator that decodes these delays
D = np.linalg.lstsq(F, G, rcond=1e-6)[0]
D /= np.max(
    F @ D, axis=0)[None, :]  # Normalise all decoded delays to a maximum of one

# Simulate a Nengo network to compute more-or-less realistic activities
with nengo.Network() as model:
    us = nengo.Node(lambda t: np.exp(-t / taus))
    xs = nengo.Ensemble(n_neurons=N_delays,
                        dimensions=N_delays,
                        encoders=np.eye(N_delays),
                        intercepts=nengo.dists.Uniform(0.4, 0.6),
                        max_rates=nengo.dists.Uniform(20, 50),
                        noise=nengo.processes.FilteredNoise(
                            synapse=0.1,
                            dist=nengo.dists.Gaussian(mean=0.0, std=0.1)))

    nengo.Connection(us, xs, transform=D.T)

    p_us = nengo.Probe(us, synapse=None)
    p_xs = nengo.Probe(xs.neurons, synapse=None)

with nengo.Simulator(model) as sim:
    sim.run(5.0)

# Plot stuff
fig, axs = plt.subplots(1,
                        3,
                        figsize=(7.4, 1.85),
                        gridspec_kw={
                            "wspace": 0.4,
                        })

ts = sim.trange()
As = sim.data[p_xs]
As = nengo.Lowpass(tau=0.2).filtfilt(As, dt=1e-3)
As_max = np.max(As, axis=0)
As_max_idcs = np.argmax(As, axis=0)
idcs = np.argsort(As_max_idcs)
As_max_ts = ts[As_max_idcs[idcs]]

axs[0].imshow((As[:, idcs] / As_max[idcs]).T,
              extent=(0, 5.0, N_delays + 0.5, 0.5),
              cmap='inferno',
              interpolation='nearest')
axs[0].plot(As_max_ts,
            np.arange(0.5, N_delays + 0.5),
            linewidth=3,
            color='white',
            alpha=1)
axs[0].plot(As_max_ts,
            np.arange(0.5, N_delays + 0.5),
            linestyle=':',
            linewidth=1,
            color='k',
            alpha=1)
axs[0].set_aspect('auto')
axs[0].set_xlabel("Time $t - t_0$ (s)")
axs[0].set_ylabel("Neuron index $i$")
axs[0].set_title("\\textbf{Model activity over time}")
axs[0].set_xticks(np.linspace(0, 5, 6))

As_len = np.linalg.norm(As, axis=1)
As_norm = As / As_len[:, None]
axs[1].imshow(As_norm @ As_norm.T,
              cmap='inferno',
              origin='upper',
              extent=(0, 5, 5, 0))
axs[1].plot(ts, ts, linewidth=3, color='white', alpha=1)
axs[1].plot(ts, ts, linestyle=':', linewidth=1, color='k', alpha=1)
axs[1].set_xticks(np.linspace(0, 5, 6))
axs[1].set_xlabel("Time $t - t_0$ (s)")
axs[1].set_ylabel("Time $t - t_0$ (s)")
axs[1].set_aspect('auto')
axs[1].set_title("\\textbf{Model activity similarity}")

ts = np.arange(0, T, dt)
dt = (ts[-1] - ts[0]) / (len(ts) - 1)
for idx, i in enumerate(np.linspace(1, 10, 4)):
    ys = (ts**i) * np.exp(-ts / 0.1)
    ys /= np.sum(ys) * dt
    axs[2].plot(ts, ys, 'k-', linewidth=1)

#    D = np.linalg.lstsq(F, ys, rcond=1e-6)[0]
#    ys_dec = F @ D
#    axs[2].plot(ts, ys_dec, 'k--', linewidth=0.5)

    x0 = i * 0.1
    y0 = ys[int(x0 / dt)]
    utils.annotate(axs[2],
                   x0 + 0.04,
                   y0 + 0.08,
                   x0 + 0.2,
                   y0 - 0.05 * (idx)**2 + bool(idx) * 0.6 +
                   (1.0 - bool(idx)) * 0.2,
                   "$\\theta = {:0.1f}$".format(x0),
                   ha="left")
axs[2].set_xlim(0, 2)
axs[2].set_ylim(0, 4)
axs[2].set_xticks(np.linspace(0, 2, 3))
axs[2].set_xticks(np.linspace(0, 2, 5), minor=True)
axs[2].set_xlabel("Time $t$ (s)")
axs[2].set_ylabel("Impulse response $h_{\\theta}(t)$")
axs[2].set_title("\\textbf{Impulse responses}")

for i in range(3):
    utils.outside_ticks(axs[i])
    axs[i].text(-0.25 if i == 0 else -0.22,
                1.0625,
                '\\textbf{{{}}}'.format(chr(ord('A') + i)),
                size=12,
                ha="left",
                va="baseline",
                transform=axs[i].transAxes)

utils.save(fig)

