import scipy.linalg

import dlop_ldn_function_bases as bases
import halton
from lstsq_cstr import lstsq_cstr


def eval_lti(A, B, ts):
    return np.array([scipy.linalg.expm(A * t) @ B for t in ts])


#c1 = np.array(mpl.colors.to_rgb("tab:blue"))
#c2 = np.array(mpl.colors.to_rgb("tab:orange"))
c1 = np.array(mpl.colors.to_rgb(utils.blues[0]))
c2 = np.array(mpl.colors.to_rgb(utils.oranges[1]))
c3 = np.sqrt(1 / 2) * c1 + (1.0 - np.sqrt(1 / 2)) * c2

dt = 1e-2
T = 5.0
ts = np.arange(0, T, dt)
N = len(ts)
H = bases.mk_fourier_basis(100, N)

A, B = bases.mk_ldn_lti(6)
M = eval_lti(A, B, ts)
M_H = np.linalg.pinv(H.T) @ M


def H_evelope(q, sigma=2.0, T=T):
    fs = np.zeros(q)
    fs[0] = 0.0
    for i in range(1, q):
        fs[i] = ((i + 1) // 2) / T

    ps = np.exp(-fs**2 / sigma**2)
    return fs, ps


def mk_sig(ps):
    q = H.shape[0]
    X_F = ps * np.random.normal(0, 1, q)

    # Generate the Fourier coefficients, scale the function to an RMS of 0.5
    xs = (H.T @ X_F)[::-1]  # This could be an FFT
    peak_min, peak_max = np.min(xs), np.max(xs)
    offs = -0.5 * (peak_max + peak_min)
    scale = 2.0 / (peak_max - peak_min)
    X_F[0] += offs
    X_F *= scale

    return X_F, H.T @ X_F


def power_spectrum(xi):
    q = len(xi)
    fs = np.zeros((q + 1) // 2)
    Xs = np.zeros((q + 1) // 2)
    Xs[0] = np.abs(xi[0])
    for i in range(1, (q + 1) // 2):
        fs[i] = i / T
        Xs[i] = np.hypot(xi[2 * i + 0], xi[2 * i + 1])
    return fs, Xs


def draw_circle(ax):
    phis = np.linspace(-np.pi, np.pi, 1001)
    ax.plot(np.cos(phis), np.sin(phis), 'k--', lw=0.5, clip_on=False)
    for x in [-1, 1]:
        ax.plot([x, x], [-0.05, 0.05], 'k-', lw=1.25, clip_on=False)
    for y in [-1, 1]:
        ax.plot([-0.05, 0.05], [y, y], 'k-', lw=1.25, clip_on=False)
    ax.plot([-1.1, 1.1], [0.0, 0.0],
            linestyle=':',
            color='grey',
            lw=0.5,
            clip_on=False)
    ax.plot([0.0, 0.0], [-1.1, 1.1],
            linestyle=':',
            color='grey',
            lw=0.5,
            clip_on=False)
    ax.set_xlim(-0.575, 1.175)
    ax.set_ylim(-0.875, 0.875)
    ax.set_aspect(1)


fig, axs = plt.subplots(2,
                        4,
                        figsize=(7.45, 4.25),
                        gridspec_kw={
                            "wspace": 0.5,
                            "hspace": 0.45,
                            "width_ratios": [3, 3, 3, 1]
                        })

phis = np.linspace(-np.pi, np.pi, 1001)
e = np.array((-np.sqrt(1 / 2), np.sqrt(1 / 2)))
for i in range(2):
    np.random.seed(12355)

    color = [c1, c2][i]

    As = []
    fs, ps = H_evelope(H.shape[0])
    for k in range(2187):
        xi, xs = mk_sig(ps)
        if i == 1:
            xi_orig, xs_orig = xi, xs[::-1]
            ξ = halton.halton_ball(2, k)
            xi = lstsq_cstr(np.diag(1.0 / ps), xi_orig / ps, M_H[:, :2].T,
                            ξ / dt)
            xs = (H.T @ xi)[::-1]
        else:
            xs = xs[::-1]

        a = np.array([np.convolve(xs, M[:, j], 'valid')[0]
                      for j in range(2)]) * dt
        As.append(a)
    As = np.array(As)

    axs[i, 0].plot(fs, 10.0 * np.log10(ps), 'k--', lw=1.0, zorder=100)
    axs[i, 0].set_xlim(fs[0], fs[-1])
    axs[i, 0].set_xlabel("Frequency $F_\\ell$ ($\\mathrm{Hz}$)")
    axs[i, 0].set_ylabel("$\\sqrt{X_{\\ell - 1}^2 + X_{\ell}^2}$ $\\mathrm{(dB)}$")
    axs[i, 0].set_xlim(0, 4)
    axs[i, 0].set_ylim(-15, 10)
    if i == 0:
        axs[i, 0].set_title("Power spectrum", y=1.05)

    fs, Xs = power_spectrum(xi)
    if i == 1:
        _, Xs_orig = power_spectrum(xi_orig)
    for j in range(len(fs)):
        axs[i, 0].plot(fs[j],
                       10.0 * np.log10(Xs[j]),
                       "o",
                       markersize=3,
                       fillstyle="none",
                       color=color,
                       zorder=10)
        if i == 1:
            axs[i, 0].plot(fs[j],
                           10.0 * np.log10(Xs_orig[j]),
                           "o",
                           markersize=3,
                           fillstyle="none",
                           alpha=0.5,
                           color=color,
                           zorder=9)
        axs[i, 0].plot([fs[j], fs[j]], [-20, 10.0 * np.log10(Xs[j])],
                       "k:",
                       linewidth=0.5,
                       color='grey')

    if i == 1:
        axs[i, 1].plot(ts, xs_orig, color='k', lw=0.5, linestyle=':')
    axs[i, 1].plot(ts, xs, color=color, lw=1.0)
    axs[i, 1].set_xlim(0, T)
    axs[i, 1].set_xticks(np.arange(0, 5.1, 2))
    axs[i, 1].set_xticks(np.arange(0, 5.1, 1), minor=True)
    axs[i, 1].set_xlabel("Time $t$ ($\\mathrm{s}$)")
    axs[i, 1].set_ylabel("$\\mathfrak{x}_k(t)$")
    axs[i, 1].set_ylim(-1.2, 1.2)
    if i == 0:
        axs[i, 1].set_title("Time-domain", y=1.05)

    draw_circle(axs[i, 2])
    utils.remove_frame(axs[i, 2])
    axs[i, 2].arrow(-0.85 * e[0],
                    -0.85 * e[1],
                    1.7 * e[0],
                    1.7 * e[1],
                    width=0.02 * 3,
                    head_width=0.075 * 1.5,
                    linewidth=0.0,
                    color='white',
                    clip_on=False,
                    zorder=100)
    axs[i, 2].arrow(-0.85 * e[0],
                    -0.85 * e[1],
                    1.7 * e[0],
                    1.7 * e[1],
                    width=0.02,
                    head_width=0.075,
                    linewidth=0.0,
                    color='k',
                    clip_on=False,
                    zorder=101)
    for proj in [-0.5, 0.0, 0.5]:
        axs[i, 2].plot([proj * e[0] - 0.05 * e[1], proj * e[0] + 0.05 * e[1]],
                       [proj * e[1] + 0.05 * e[0], proj * e[1] - 0.05 * e[0]],
                       'k-',
                       lw=0.7,
                       zorder=102)

    axs[i, 2].scatter(As[::7, 0],
                      As[::7, 1],
                      marker='+',
                      color='k',
                      s=20,
                      lw=0.7,
                      clip_on=False)

    axs[i, 2].scatter(As[-1, 0],
                      As[-1, 1],
                      marker='o',
                      color=color,
                      s=50,
                      clip_on=False,
                      zorder=103)
    axs[i, 2].scatter(As[-1, 0],
                      As[-1, 1],
                      marker='+',
                      color='white',
                      s=25,
                      lw=1.0,
                      clip_on=False,
                      zorder=104)
    if i == 0:
        axs[i, 2].set_title("Basis and encoder projection", y=1.118, x=0.8)

    text = "$e^\\mathrm{t}_{i}$"
    ha = ("center" if np.abs(e[0]) < 0.1 else
          ("right" if e[0] < 0.0 else "left"))
    va = ("center" if np.abs(e[1]) < 0.1 else
          ("bottom" if e[1] > 0.0 else "top"))
    axs[i, 2].text(1.06 * e[0], 1.06 * e[1], text, ha=ha, va=va)

    proj = As[-1, :2] @ e
    axs[i, 2].plot([As[-1, 0], proj * e[0]], [As[-1, 1], proj * e[1]],
                   'white',
                   lw=2,
                   zorder=100)
    axs[i, 2].plot([As[-1, 0], proj * e[0]], [As[-1, 1], proj * e[1]],
                   ':',
                   color=color,
                   lw=0.7,
                   zorder=102)
    axs[i, 2].plot([proj * e[0]], [proj * e[1]],
                   'x',
                   color=color,
                   markersize=5,
                   lw=0.7,
                   zorder=102)

    violin_parts = axs[i, 3].violinplot(As[:, :2] @ e[:2])
    violin_parts['cbars'].set_color('k')
    violin_parts['cmins'].set_color('k')
    violin_parts['cmaxes'].set_color('k')
    for pc in violin_parts['bodies']:
        pc.set_facecolor(c3)

    axs[i, 3].set_ylim(-1.2, 1.2)
    axs[i, 3].spines["bottom"].set_visible(False)
    axs[i, 3].set_xticks([])
    axs[i, 3].set_ylabel("")

    axs[i, 3].scatter(1.0, proj, marker='o', color=color, s=50, zorder=102)
    axs[i, 3].scatter(1.0,
                      proj,
                      marker='+',
                      color='white',
                      s=25,
                      lw=1.0,
                      clip_on=False,
                      zorder=103)
    axs[i, 3].set_ylabel("$\\langle \\vec{e}^\\mathrm{t}_i, \\vec{x}^\\mathrm{t}_k \\rangle$")

fig.text(0.07, 0.95, "\\textbf{A}", size=12, ha="left", va="baseline")
fig.text(0.5, 0.95, "\\textbf{Na\\\"ively sampled input signals} $\\mathfrak{x}_k$", ha="center", va="baseline")

fig.text(0.07, 0.45, "\\textbf{B}", size=12, ha="left", va="baseline")
fig.text(0.5, 0.45, "\\textbf{Uniform activation sampling}", ha="center", va="baseline")

utils.save(fig)

