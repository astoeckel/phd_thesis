import scipy.linalg

import dlop_ldn_function_bases as bases


def eval_lti(A, B, ts):
    return np.array([scipy.linalg.expm(A * t) @ B for t in ts])


dt = 1e-2
T = 2.0
ts = np.arange(0, T, dt)

A, B = bases.mk_ldn_lti(6, rescale=True)
M = eval_lti(A, B, ts)

fig, axs = plt.subplots(1,
                        2,
                        figsize=(7.0, 2.25),
                        gridspec_kw={"wspace": 0.575})

#c1 = np.array(mpl.colors.to_rgb("tab:blue"))
#c2 = np.array(mpl.colors.to_rgb("tab:orange"))
c1 = np.array(mpl.colors.to_rgb(utils.blues[0]))
c2 = np.array(mpl.colors.to_rgb(utils.oranges[1]))

for i in range(2, A.shape[0]):
    axs[0].plot(ts, M[:, i], color="k", linestyle="-", linewidth=0.5)

axs[0].plot(ts, M[:, 1], linewidth=1.5, color=c2)
axs[0].plot(ts, M[:, 0], linewidth=1.5, color=c1)
axs[0].plot(ts, M[:, 1], linewidth=1.5, linestyle=':', color=c2)

axs[0].set_xlim(0, T)
axs[0].spines["left"].set_visible(False)
axs[0].set_yticks([])
axs[0].axhline(0.0, linestyle=':', color='grey', lw=0.5)
axs[0].set_xlabel("Time $t$ (s)")

utils.annotate(axs[0], 0.95, 0.75, 1.25, 0.75, "$\\mathfrak{h}_1(t)$")
utils.annotate(axs[0],
               0.69,
               0.4,
               1.0,
               -0.2,
               "$\\mathfrak{h}_2(t)$",
               ha="left",
               va="top")

utils.remove_frame(axs[1])


def draw_circle(ax):
    phis = np.linspace(-np.pi, np.pi, 1001)
    ax.plot(np.cos(phis), np.sin(phis), 'k--', lw=0.5)
    for x in [-1, 1]:
        ax.plot([x, x], [-0.05, 0.05], 'k-', lw=1.25)
    for y in [-1, 1]:
        ax.plot([-0.05, 0.05], [y, y], 'k-', lw=1.25)
    ax.plot([-1.1, 1.1], [0.0, 0.0], linestyle=':', color='grey', lw=0.5)
    ax.plot([0.0, 0.0], [-1.1, 1.1], linestyle=':', color='grey', lw=0.5)
    ax.plot([0.0, 0.0], [-0.05, 0.05], 'k-', lw=1.25)
    ax.plot([-0.05, 0.05], [0.0, 0.0], 'k-', lw=1.25)
    ax.set_aspect(1)


draw_circle(axs[1])

axs[1].set_xlim(-1.5, 1.5)
axs[1].set_ylim(-1.2, 1.5)

for i, phi in enumerate(np.linspace(-np.pi, np.pi, 9)[:-1]):
    e0, e1 = np.cos(phi), np.sin(phi)
    x0, y0 = 0.75, 0.44
    w, h = 0.1, 0.1
    r0 = 0.2
    r1 = 0.39

    color = c1 * np.abs(e0) + c2 * (1.0 - np.abs(e0))

    ax = fig.add_axes([x0 + r0 * e0 - 0.5 * w, y0 + r1 * e1 - 0.5 * h, w, h])
    ys = e0 * M[:, 0] + e1 * M[:, 1]
    ax.plot(ts, ys, '-', color=color)
    utils.remove_frame(ax)
    ax.set_xlim(0, T)
    ax.axhline(0.0, linestyle=':', color='grey', lw=0.5)
    ax.text(1.9,
            max(0.25,
                np.max(ys) - 0.25),
            f"$\\mathfrak{{e}}_{{{i + 1}}}(t)$",
            size=8,
            ha="right",
            va="baseline")

    axs[1].plot([e0], [e1], 'ko', markersize=3, color=color)
    axs[1].arrow(0,
                 0,
                 0.825 * e0,
                 0.825 * e1,
                 width=0.0125,
                 head_width=0.05,
                 linewidth=0.0,
                 color='k')

    sqrt_text = lambda x: "\\sqrt{2}" if x > 0 else "-\\sqrt{2}"
    int_text = lambda x: int(x) if int(x) == x else (sqrt_text(x))
    #text = "$e_{{{}}} = ({}, {})$".format(i + 1, int_text(e0), int_text(e1))
    text = "$e^\\mathrm{{t}}_{{{}}}$".format(i + 1)

    ha = ("center" if np.abs(e0) < 0.1 else ("right" if e0 < 0.0 else "left"))
    va = ("center" if np.abs(e1) < 0.1 else ("bottom" if e1 > 0.0 else "top"))
    axs[1].text(1.06 * e0, 1.06 * e1, text, ha=ha, va=va)

axs[0].text(-0.035,
            1.075,
            "\\textbf{A}",
            size=12,
            transform=axs[0].transAxes,
            ha="left",
            va="baseline")
axs[0].text(0.5,
            1.075,
            "\\textbf{Impulse responses} $\\mathfrak{h}(t)$",
            transform=axs[0].transAxes,
            ha="center",
            va="baseline")

axs[1].text(-0.4,
            1.0725,
            "\\textbf{B}",
            size=12,
            transform=axs[1].transAxes,
            ha="left",
            va="baseline")
axs[1].text(
    0.5,
    1.0725,
    "\\textbf{Temporal encoders} $\\mathfrak{e}(t) = \\langle \\mathfrak{h}(t), \\vec{e}^\\mathrm{t} \\rangle$",
    transform=axs[1].transAxes,
    ha="center",
    va="baseline")

utils.save(fig)

