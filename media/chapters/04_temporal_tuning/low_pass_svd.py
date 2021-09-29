import scipy.signal

T, dt = 2.002, 1e-3
q=20
ts_flt = np.arange(-T / 2, T / 2, dt)
taus = np.geomspace(2e-3, 100e-3, q)
#taus = np.linspace(2e-3, 100e-3, q)

def mk_analysis():
    flts = np.array([(ts_flt >= 0.0) * (np.exp(-ts_flt / tau))
                     for tau in taus]).T

    U, S, V = np.linalg.svd(flts)
    #U, S, V = np.linalg.svd(flts - np.mean(flts, axis=0))

    return flts, U, S

fig, axs = plt.subplots(1, 3, figsize=(7.5, 1.8), gridspec_kw={
    "wspace": 0.4,
})

flts, U, S = mk_analysis()

cax = fig.add_axes([0.175, 0.8, 0.15, 0.05])
cax.spines["left"].set_visible(False)
cax.set_yticks([])

cax.pcolormesh(taus, [0, 1], np.array((np.linspace(0, 1, q), np.linspace(0, 1, q))), shading='auto')
cax.set_xscale('log')
#cax.text(0.0, -1.4, "$\\tau$", va="baseline", ha="left", transform=cax.transAxes)
cax.set_xlabel("Time-constant $\\tau$", size=8, labelpad=0.0)

for i in range(q):
#    color = mpl.cm.get_cmap('viridis')((taus[i] - taus[0]) / (taus[-1] - taus[0]))
    color = mpl.cm.get_cmap('viridis')((i + 1) / q)
    axs[0].plot(ts_flt * 1e3, flts[:, i], color=color, zorder=-i)
axs[0].set_xlim(0, 300)
axs[0].set_ylim(0, 1)
axs[0].set_xlabel("Time $t$ (ms)")
axs[0].set_title("{Synaptic low-pass filters}")
axs[0].set_ylabel("Impulse response $h_i(t)$")
axs[0].text(-0.25, 1.057, "\\textbf{A}", size=12, transform=axs[0].transAxes)

for i in range(6):
    axs[1].plot(ts_flt * 1e3, U[:, i], zorder=-i)
axs[1].set_xlim(0, 300)
axs[1].set_ylim(-0.3, 0.3)
axs[1].set_xlabel("Time $t$ (ms)")
axs[1].set_title("{Orthogonal basis (PCA)}")
axs[1].set_ylabel("Impulse response $h_i(t)$")
axs[1].text(-0.29, 1.057, "\\textbf{B}", size=12, transform=axs[1].transAxes)

sigma = S[:6] / S[0]
bidcs = np.arange(1, len(sigma) + 1)
axs[2].plot(bidcs, sigma, 'k:', linewidth=0.7)
for i in range(len(bidcs)):
    color = mpl.cm.get_cmap('tab10')(i / 10)
    axs[2].plot(bidcs[i],
                sigma[i],
                'o',
                markersize=5,
                zorder=10,
                color=color,
                clip_on=False)
axs[2].set_ylim(0, 1)
axs[2].set_xticks(bidcs)
axs[2].set_ylabel("Singular value $\\sigma_i$")
axs[2].set_xlabel("Basis function index $i$")
axs[2].set_title("{Singular values}")
axs[2].text(-0.24, 1.057, "\\textbf{C}", size=12, transform=axs[2].transAxes)

utils.save(fig)
