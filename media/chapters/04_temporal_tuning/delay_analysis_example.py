import h5py
from basis_delay_analysis_common import *
import dlop_ldn_function_bases as bases
import scipy.stats

fig, axs = plt.subplots(5,
                        2,
                        figsize=(7.9, 4.0),
                        gridspec_kw={
                            "hspace": 0.5,
                            "height_ratios": [1, 2, 0.5, 1, 2]
                        })
utils.remove_frame(axs[2, 0])
utils.remove_frame(axs[2, 1])

BASES = [("lowpass", None), ("mod_fourier", "erasure"), ("cosine", "erasure"),
         ("legendre", "erasure")]

T = 10.0
q = 11
q_plot = min(q, 6)
dt = 1e-2

for i, (basis, window) in enumerate(BASES):
    # Generate a basis
    print(basis, window)
    ts_H, H = mk_impulse_response(basis, window, q=q, dt=dt)

    ax_basis = axs[3 * (i // 2) + 0, i % 2]
    ax_reconstruction = axs[3 * (i // 2) + 1, i % 2]

    #    ax_basis.set_xticks(np.linspace(0, 10, 6))
    #    ax_basis.set_xticks(np.linspace(0, 10, 11), minor=True)
    ax_basis.spines["bottom"].set_visible(False)
    ax_basis.set_xticks([])
    ax_basis.set_xticklabels([])
    ax_basis.set_xlim(0, T)

    ax_reconstruction.set_xticks(np.linspace(0, 10, 6))
    ax_reconstruction.set_xticks(np.linspace(0, 10, 11), minor=True)
    ax_reconstruction.set_xlim(0, T)
    ax_reconstruction.set_xlabel("Time $t$ (s)")

    ax_basis.axhline(0.0, linestyle=':', lw=0.5, color="grey", zorder=-1000)
    ax_basis.spines["left"].set_visible(False)
    ax_basis.set_yticks([])

    ax_reconstruction.axhline(0.0,
                              linestyle=':',
                              lw=0.5,
                              color="grey",
                              zorder=-1000)
    ax_reconstruction.spines["left"].set_visible(False)
    ax_reconstruction.set_yticks([])

    # Generate some test and training data
    ts = np.arange(0, T, dt)
    N_sig = len(ts)
    rng = np.random.RandomState(4109)
    n_thetas = 20
    thetas, xs, ys, xs_train, ys_train = generate_full_dataset(
        n_thetas, 1, 10, N_sig, 1.0 / dt, rng)

    # Plot the test data
    xs_conv = convolve(H.T, xs[:1, 0])[0]
    for k in range(q_plot):
        k_plt = int((q - 1) * k / (q_plot - 1)) if basis == "lowpass" else k
        ax_basis.plot(ts, xs_conv[:, k_plt], zorder=-k)

    # Iterate over each delay and compute a decoder
    Es = []
    for j in range(n_thetas):
        color = cm.get_cmap('inferno')(1.0 - j / (n_thetas - 1))
        D = np.linalg.lstsq(convolve(H.T, xs_train[j]).reshape(-1, q),
                            ys_train[j].reshape(-1),
                            rcond=1e-4)[0]
        ys_rec = xs_conv @ D
        Es.append(ys_rec - ys[j])
        ax_reconstruction.plot(ts, xs_conv @ D, color=color, zorder=-j)

    ax_reconstruction.plot(ts, xs[0, 0], 'k--', lw=0.5)
    ax_reconstruction.plot([0.8, 1.8], [3.2, 3.2],
                           'k-',
                           lw=1.5,
                           clip_on=False,
                           solid_capstyle='butt')
    ax_reconstruction.text(1.3,
                           3.5,
                           "$\\theta$",
                           color='k',
                           ha="center",
                           va="bottom")

    ax_basis.text(-0.017,
                  1.27,
                  "\\textbf{{{}}}".format(chr(ord('A') + i)),
                  size=12,
                  va="baseline",
                  ha="left", transform=ax_basis.transAxes)

    if i == 0:
        mx, my = 2.2, -100.0
        ly = 3.0
    else:
        mx, my = 3.5, [-100.0, -50.0, -60.0, -30.0][i]
        ly = 4.0

    utils.annotate(ax_basis, mx, my, mx + 0.8, 1.5 * my, "$\\vec m(t)$")
    utils.annotate(ax_reconstruction, 2.8, 2.5, 3.75, ly, "$u(t)$")
    ax_reconstruction.plot([4.25, 4.75], [ly, ly], 'k--', lw=0.5)

    utils.annotate(ax_reconstruction, 5.5, 0.5, 6.0, ly, "$u(t - \\theta')$")
    xsp = np.linspace(6.8, 7.3, 11)
    for j, (x0, x1) in enumerate(zip(xsp[:-1], xsp[1:])):
        color = cm.get_cmap('inferno')(1.0 - j / (len(xsp) - 1))
        ax_reconstruction.plot([x0, x1], [ly, ly], color=color)


    rms = np.sqrt(np.mean(np.square(ys[0])))
    rmse = np.sqrt(np.mean(np.square(Es)))
    nrmse = rmse / rms
    ax_reconstruction.text(1.0,
                           1.175,
                           f"$E = {nrmse * 100:0.1f}\\%$",
                           va="top",
                           ha="right",
                           transform=ax_reconstruction.transAxes)

    ax_basis.set_title("\\textbf{{{}}} {}".format(
        *{
            "lowpass": ("Low-pass filters", ""),
            "mod_fourier": ("Modified Fourier",
                            "(with information erasure)"),
            "cosine": ("Cosine", "(with information erasure)"),
            "legendre": ("LDN", ""),
        }[basis]))

utils.save(fig)

