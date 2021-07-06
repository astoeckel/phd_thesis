import lif_utils

def gabor(x, y, f=2.0, sigma=0.25, theta=np.pi / 4, psi=0.0, x0=0.0, y0=0.0):
    xp = np.cos(theta) * x + np.sin(theta) * y - x0
    yp = -np.sin(theta) * x + np.cos(theta) * y - y0
    E = np.exp(-(np.square(xp) + np.square(yp)) / (2.0 * np.square(sigma)))
    S = np.cos(2.0 * np.pi * f * xp - psi)
    return E * S


xs, ys = np.linspace(-1, 1, 1024), np.linspace(-1, 1, 1024)
xss, yss = np.meshgrid(xs, ys)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.36, 3.0), gridspec_kw={
    'wspace': 0.33,
})
zss = gabor(xss, yss, sigma=0.4, f=1.0, theta=0.0)
ax1.imshow(zss,
           extent=[np.min(xs), np.max(xs),
                   np.min(ys), np.max(ys)],
           cmap='RdBu',
           vmin=-1,
           vmax=1)
ax1.set_ylabel('$\\xi_2$')
ax1.set_xlabel('$\\xi_1$')
for i in [-2, -1, 0, 1, 2]:
    ax1.text(
        -0.45 + 0.0125 * (i * i),
        0.25 * i,
        '$\\mathbf{-}$',
        va='center',
        ha='center',
        fontsize=12 - 2 * np.abs(i),
    )
    ax1.text(
        0.0,
        0.25 * i,
        '$\\mathbf{+}$',
        va='center',
        ha='center',
        color='white',
        fontsize=12 - 2 * np.abs(i),
    )
    ax1.text(
        0.45 - 0.0125 * (i * i),
        0.25 * i,
        '$\\mathbf{-}$',
        va='center',
        ha='center',
        fontsize=12 - 2 * np.abs(i),
    )
ax1.set_title('Receptive field $e(\\xi_1, \\xi_2)$')
ax1.text(-0.275, 1.048, '\\textbf{A}', fontsize=12, transform=ax1.transAxes)

ax2.set_xlim(-1, 1)
ax2.set_ylim(-1, 1)
ax2.set_aspect(1)
ax2.set_xlabel('$\\xi_1$')
ax2.set_ylabel('$\\xi_2$')
ax2.set_title('Stimulus $x(\\xi_1, \\xi_2)$')
ax2.text(-0.275, 1.048, '\\textbf{B}', fontsize=12, transform=ax2.transAxes)
for i, angle in enumerate(np.linspace(-np.pi / 2, np.pi / 2, 8)[:-1]):
    w = 0.2
    h = 1.5
    x0, x1, y0, y1 = -w / 2, w / 2, -h / 2, h / 2
    x00 = np.cos(angle) * x0 + np.sin(angle) * y0
    y00 = -np.sin(angle) * x0 + np.cos(angle) * y0

    x10 = np.cos(angle) * x1 + np.sin(angle) * y0
    y10 = -np.sin(angle) * x1 + np.cos(angle) * y0

    x01 = np.cos(angle) * x0 + np.sin(angle) * y1
    y01 = -np.sin(angle) * x0 + np.cos(angle) * y1

    x11 = np.cos(angle) * x1 + np.sin(angle) * y1
    y11 = -np.sin(angle) * x1 + np.cos(angle) * y1

    linestyle = '-' if i == 0 else '--'
    linewidth = 1.0 if i == 0 else 0.5
    ax2.plot([x00, x10, x11, x01, x00], [y00, y10, y11, y01, y00],
             color='k',
             lw=linewidth,
             ls=linestyle)

angles = np.linspace(0, np.pi / 6)
ax2.plot(0.9 * np.cos(angles), 0.9 * np.sin(angles), 'k-', linewidth=1.5)
ax2.arrow(0.9 * np.cos(angles)[-1],
          0.9 * np.sin(angles)[-1],
          -0.01 * np.sin(angles[-1]),
          0.01 * np.cos(angles[-1]),
          width=0.01,
          color='k')
ax2.text(np.cos(angles[4 * len(angles) // 7]),
         np.sin(angles[4 * len(angles) // 7]),
         '$\\varphi$',
         va='center',
         ha='center')

n_angles = 101
angles = np.linspace(0, np.pi, n_angles)
activations = np.zeros(n_angles)
for i, angle in enumerate(angles):
    zss = gabor(xss, yss, sigma=0.4, f=1.0, theta=-angle + np.pi/2)
    activations[i] = np.sum(zss[np.logical_and(
        np.logical_and(xss >= -w / 2, yss >= -h / 2),
        np.logical_and(xss <= w / 2, yss <= h / 2))])

# Some plausible neural non-linearity
activities = 10 * np.power(activations / np.max(activations), 2.0)

ax3.set_xlim(0, 180)
ax3.set_xticks(np.arange(0, 181, 45))
ax3.set_xticklabels(['{:0.0f}\\textdegree{{}}'.format(x) for x in np.arange(0, 181, 45)])
ax3.plot(angles * 180 / np.pi, activities, 'k-')
ax3.set_aspect(16.35)
ax3.set_xlabel('Angle $\\varphi$')
ax3.set_ylabel('Activities $a(\\varphi)$ ($\\mathrm{s}^{-1}$)')
ax3.set_title('Tuning curve $a(\\varphi)$')
ax3.text(-0.225, 1.048, '\\textbf{C}', fontsize=12, transform=ax3.transAxes)

utils.save(fig)

