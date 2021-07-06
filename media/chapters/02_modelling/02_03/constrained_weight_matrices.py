import lif_utils
import scipy.optimize

np.random.seed(48988)

# Generate some random post-population encoders
N_post = 31
N_pre = 32
#N_post = 11
#N_pre = 10
d_pre = d_post = 3
E_post = np.random.normal(0, 1, (N_post, d_post))
E_post /= np.linalg.norm(E_post, axis=1)[:, None]
E_pre = np.random.normal(0, 1, (N_pre, d_pre))
E_pre /= np.linalg.norm(E_pre, axis=1)[:, None]

# Generate some pre-population tuning curves
max_rates = np.random.uniform(50, 100, N_pre)
x_intercepts = np.random.uniform(-0.95, 0.95, N_pre)

J1 = lif_utils.lif_rate_inv(max_rates)
J0 = lif_utils.lif_rate_inv(1e-3)

gains = (J1 - J0) / (1 - x_intercepts)
biases = (J0 - x_intercepts * J1) / (1 - x_intercepts)

# Generate some samples
N_smpls = 1000
xs = np.random.normal(0, 1, (N_smpls, d_pre))
xs /= np.linalg.norm(xs, axis=1)[:, None]
xs *= np.power(np.random.uniform(0, 1, N_smpls)[:, None], 1 / d_pre)
Js = gains[None, :] * (xs @ E_pre.T) + biases[None, :]
As = lif_utils.lif_rate(Js)

# Compute the decoders
ys = xs


def loss_l1(D, σ=1):
    D = D.reshape(d_post, N_pre)
    return np.sum(
        np.square(ys - As @ D.T)) + np.square(σ) * N_smpls * np.sum(np.abs(D))


def loss_l2(D, σ=1):
    D = D.reshape(d_post, N_pre)
    return np.sum(np.square(ys - As @ D.T)) + np.square(σ) * N_smpls * np.sum(
        np.square(D))


D0 = np.random.randn(d_post, N_pre)
D_L1 = scipy.optimize.minimize(loss_l1,
                               x0=D0.flatten()).x.reshape(d_post, N_pre)
D_L2 = scipy.optimize.minimize(loss_l2,
                               x0=D0.flatten()).x.reshape(d_post, N_pre)

# Solve for non-negative weights, assuming that approx. the last ten pre-neurons are inhibitory
N_inh = int(0.3 * N_pre)
N0, N1, N2 = 0, N_pre - N_inh, N_pre
W_dale = np.zeros((N_post, N_pre))
J_tar = ys @ E_post.T
As_signed = np.concatenate((As[:, N0:N1], -As[:, N1:N2]), axis=1)
σ = 0.1
for i in range(N_post):
    W_dale[i] = scipy.optimize.nnls(
        As_signed.T @ As_signed + N_smpls * np.square(σ) * np.eye(N_pre),
        As_signed.T @ J_tar[:, i])[0]
W_dale[:, N1:N2] *= -1

print(np.sqrt(np.mean(np.square(ys - As @ D_L1.T))),
      np.sqrt(np.mean(np.square(ys - As @ D_L2.T))),
      np.sqrt(np.mean(np.square(J_tar - As @ W_dale.T))))

fig = plt.figure(figsize=(6.625, 2.0))

gss = [
    fig.add_gridspec(2,
                     2,
                     width_ratios=[1, 12],
                     height_ratios=[1, 12],
                     hspace=0.1,
                     wspace=0.1,
                     top=0.975,
                     bottom=0.25,
                     left=0.05 + i * (0.9 / 3),
                     right=0.05 + i * (0.9 / 3) + 0.75 / 3) for i in range(3)
]


def plot_weight_matrix(fig,
                       gs,
                       E=None,
                       D=None,
                       W=None,
                       ylabel_left=True,
                       ylabel_right=True):

    xticks = np.arange(0, N_pre + 1, 10)
    yticks = np.arange(0, N_pre + 1, 10)

    ax1, ax2, ax3 = None, None, None

    if not D is None:
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(D / np.percentile(np.abs(D), 95),
                   cmap='RdBu',
                   vmin=-1,
                   vmax=1,
                   interpolation=None)
        ax1.set_aspect('auto')
        ax1.set_xticklabels([])
        ax1.set_xticks(xticks)
        ax1.set_yticks([])
        ax1.set_title("Decoder $\\mathbf{D}$")
        utils.add_frame(ax1)

    if not E is None:
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.imshow(E / np.percentile(np.abs(E), 95),
                   cmap='RdBu',
                   vmin=-1,
                   vmax=1,
                   interpolation=None)
        ax2.set_aspect('auto')
        ax2.set_yticklabels([])
        ax2.set_xticks([])
        ax2.set_yticks(yticks)
        if ylabel_left:
            ax2.set_ylabel("Encoder $\\mathbf{E}$")
        utils.add_frame(ax2)

    if (W is None) and (not E is None) and (not D is None):
        W = E @ D

    if not W is None:
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.imshow(W / np.percentile(np.abs(W), 95),
                   cmap='RdBu',
                   vmin=-1,
                   vmax=1,
                   interpolation=None)
        ax3.set_aspect('auto')
        ax3.set_xlabel("Pre-neuron index $j$")
        ax3.set_xticks(xticks)
        ax3.set_yticks(yticks)
        ax3.yaxis.tick_right()
        ax3.yaxis.set_label_position("right")
        if ylabel_right:
            ax3.set_ylabel("Post-neuron index $i$")
        else:
            ax3.set_yticklabels([])
        ax3.set_xlim(-0.5, N_pre - 0.5)
        ax3.set_ylim(N_post - 0.5, -0.5)
        utils.outside_ticks(ax3)
        utils.add_frame(ax3)

    return ax1, ax2, ax3


plot_weight_matrix(fig,
                   gss[0],
                   E_post,
                   D_L2,
                   ylabel_left=True,
                   ylabel_right=False)
plot_weight_matrix(fig,
                   gss[1],
                   E_post,
                   D_L1,
                   ylabel_left=False,
                   ylabel_right=False)
_, _, ax3 = plot_weight_matrix(fig,
                               gss[2],
                               W=W_dale,
                               ylabel_left=False,
                               ylabel_right=True)
ax3.set_title("Weights $\\mat W$", y=1.135)
ax3.plot([-0.5, N_pre - N_inh - 0.5], [-2, -2],
         '-',
         linewidth=3,
         solid_capstyle='butt',
         color=mpl.cm.get_cmap('RdBu')(0.9),
         clip_on=False)
ax3.text(0.5 * (N_pre - N_inh), -2.5, "Exc.", va="bottom", ha="center", size=8)
ax3.plot([N_pre - N_inh - 0.5, N_pre - 0.5], [-2, -2],
         '-',
         linewidth=3,
         solid_capstyle='butt',
         color=mpl.cm.get_cmap('RdBu')(0.1),
         clip_on=False)
ax3.text(N_pre - N_inh * 0.5, -2.5, "Inh.", va="bottom", ha="center", size=8)

fig.text(0.015,
         1.1,
         "\\textbf{A}",
         transform=fig.transFigure,
         size=12,
         va="bottom",
         ha="left")
fig.text(0.3475,
         1.1,
         "\\textbf{B}",
         transform=fig.transFigure,
         size=12,
         va="bottom",
         ha="left")
fig.text(0.65,
         1.1,
         "\\textbf{C}",
         transform=fig.transFigure,
         size=12,
         va="bottom",
         ha="left")

fig.text(0.185,
         1.1,
         "\\textbf{$L_2$-regularised LSTSQ}",
         va="bottom",
         ha="center")
fig.text(0.485,
         1.1,
         "\\textbf{$L_1$-regularised LSTSQ}",
         va="bottom",
         ha="center")
fig.text(0.785,
         1.1,
         "\\textbf{$L_2$-regularised NNLS}",
         va="bottom",
         ha="center")

gss_hist = [
    fig.add_gridspec(1,
                     2,
                     width_ratios=[1, 12],
                     hspace=0.1,
                     wspace=0.1,
                     top=-0.025,
                     bottom=-0.1625,
                     left=0.05 + i * (0.9 / 3),
                     right=0.05 + i * (0.9 / 3) + 0.75 / 3) for i in range(3)
]

ax_hist1 = fig.add_subplot(gss_hist[0][0, 1])
ax_hist2 = fig.add_subplot(gss_hist[1][0, 1])
ax_hist3 = fig.add_subplot(gss_hist[2][0, 1])


def plot_weight_statistic(ax, W):
    p = np.percentile(np.abs(W), 95)
    W = W.flatten() / p
    hist, edges = np.histogram(W,
                               bins=np.linspace(-1, 1, 12),
                               weights=np.ones_like(W) / W.size)

    for i in range(len(edges) - 1):
        ax.bar(x=0.5 * (edges[i] + edges[i + 1]),
               height=hist[i],
               width=edges[i + 1] - edges[i],
               color=mpl.cm.get_cmap('RdBu')(i / (len(edges) - 2)))
        if i == (len(edges) - 1) // 2:
            utils.annotate(ax,
                           0,
                           min(hist[i] - 0.05, 0.2),
                           -0.4,
                           0.175,
                           "{:0.1f}\\%".format(hist[i] * 100),
                           va="center",
                           ha="right")
        ax.bar(
            x=0.5 * (edges[i] + edges[i + 1]),
            height=hist[i],
            width=edges[i + 1] - edges[i],
            fill=False,
            edgecolor='k',
            linewidth=0.75,
        )

    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
    ax.set_ylim(0, 0.25)
    ax.set_xlim(-1, 1)
    ax.set_xticks([-1, 0, 1])
    ax.set_xticks([-1, -0.5, 0, 0.5, 1], minor=True)
    ax.set_yticks([0, 0.25])
    ax.set_yticks([0, 0.12, 0.25], minor=True)
    ax.set_xlabel("Weight $w / P_{95}$")
    utils.outside_ticks(ax)


plot_weight_statistic(ax_hist1, E_post @ D_L2)
plot_weight_statistic(ax_hist2, E_post @ D_L1)
plot_weight_statistic(ax_hist3, W_dale)

utils.save(fig)

