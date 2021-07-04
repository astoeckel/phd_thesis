import nengo
from mpl_toolkits.mplot3d import Axes3D

# Neuron model
G = lambda J: nengo.LIFRate().rates(J.reshape(-1), gain=1.0, bias=0.0).reshape(
    J.shape)

# Select gain and bias
α = 1.0
Jbias = 1.0

# Select an encoding vector and make sure that its normalised
e = np.array([1.0, 1.0])
e = e / np.linalg.norm(e)

# Regularly sample the x- and y-axis into a 2D-grid
xs = np.linspace(-1.0, 1.0, 50)
ys = np.linspace(-1.0, 1.0, 50)
xss, yss = np.meshgrid(xs, ys)

# Also sample along the unit circle
ϕs = np.linspace(-np.pi, np.pi, 1000)
cxs, cys = np.cos(ϕs), np.sin(ϕs)

# Compute the neuron activities at each grid-point
G_2D = lambda x1, x2: G(α * (x1 * e[0] + x2 * e[1]) + Jbias)
zss = G_2D(xss, yss)
czs = G_2D(cxs, cys)

# Plot the activities
fig = plt.figure(figsize=(7.95, 3.5))
grid = mpl.gridspec.GridSpec(1, 2, fig, wspace=0.4, width_ratios=[0.6, 0.4])
ax1 = fig.add_subplot(grid[0], projection='3d')
ax1.plot_surface(xss, yss, zss, cmap=cm.get_cmap('viridis'))
ax1.plot(cxs, cys, czs, linewidth=2, color='white', zorder=5)
ax1.plot(cxs, cys, czs, ':', linewidth=2, color='black', zorder=5)
color = utils.oranges[1] #cm.get_cmap('tab10')(0.15)
ax1.plot([0], [0], G_2D(0, 0), 'o', color=color, zorder=10)
ax1.plot([e[0]], [e[1]], [G_2D(*e)], marker=(3, 0, 65), zorder=10, color=color)
ax1.plot(np.linspace(0, e[0], 100),
         np.linspace(0, e[1], 100),
         G_2D(np.linspace(0, e[0], 100), np.linspace(0, e[1], 100)),
         color=color,
         zorder=20)
ax1.set_xticks(np.linspace(-1, 1, 5))
ax1.set_yticks(np.linspace(-1, 1, 5))
ax1.set_xlabel("$x_1$")
ax1.set_ylabel("$x_2$")
ax1.set_zlabel('Activity $a$ ($\\mathrm{s}^{-1}$)')

ax1.dist = 12  # Distance of the camera from the plot

ax2 = fig.add_subplot(grid[1])
ax2.plot(ϕs, czs, clip_on=False, color=utils.blues[0])
ax2.plot(np.arctan2(e[1], e[0]), 0, '^', color=utils.oranges[1])
ax2.set_xticks(np.linspace(-np.pi, np.pi, 5))
ax2.set_xticklabels(
    ['$-\\pi$', '$-\\frac{\\pi}{2}$', '0', '$\\frac{\\pi}{2}$', '$\\pi$'])
ax2.set_xlabel('Angle on the unit circle $\\phi$ (rad)')
ax2.set_ylabel('Activity $a$ ($\\mathrm{s}^{-1}$)')
ax2.set_aspect(0.085)

fig.text(0.165,
         0.75,
         "\\textbf{A}",
         fontdict={"size": 12},
         transform=fig.transFigure,
         va='bottom',
         ha='left')

fig.text(0.595,
         0.75,
         "\\textbf{B}",
         fontdict={"size": 12},
         transform=fig.transFigure,
         va='bottom',
         ha='left')

utils.save(fig)

