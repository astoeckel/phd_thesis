import lif_utils
from nonneg_common import *


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def forkrng(rng=np.random):
    return np.random.RandomState(rng.randint((1 << 31)))


def ratio_formatter(x):
    p1 = x
    p2 = (1.0 - x)
    base = 100
    a1 = int(np.round(p1 * base / 5)) * 5
    a2 = int(np.round(p2 * base / 5)) * 5
    d = gcd(a1, a2)
    b1 = a1 // d
    b2 = a2 // d
    return f"${b1}\\!\\!:\\!\\!{b2}$"


def run_single(rng=np.random,
               N_pre=101,
               N_post=101,
               N_smpls=1001,
               sigma=0.1,
               f=lambda x: x,
               p_exc=0.5,
               do_decode_bias=True):
    gains_pre, biases_pre, encoders_pre = mk_ensemble(N_pre, rng=forkrng(rng))
    gains_post, biases_post, encoders_post = mk_ensemble(N_post,
                                                         rng=forkrng(rng))

    # Sample the source space
    xs = np.linspace(-1, 1, N_smpls).reshape(-1, 1)

    # Compute the target values
    ys = f(xs)

    # Determine the pre-activities
    Js_pre = gains_pre[None, :] * (xs @ encoders_pre.T) + biases_pre[None, :]
    As_pre = lif_utils.lif_rate(Js_pre)

    # Determine the target currents
    Js_post = gains_post[None, :] * (ys @ encoders_post.T) + biases_post[
        None, :]
    As_post = lif_utils.lif_rate(Js_post)
    if do_decode_bias:
        Js_tar = Js_post
    else:
        Js_tar = Js_post - biases_post[None, :]

    # Compute the weight matrix
    W = decode_currents(As_pre,
                        Js_tar,
                        p_exc=p_exc,
                        sigma=sigma,
                        rng=forkrng(rng))

    # Compute the decoded currents
    Js_dec = As_pre @ W.T

    # Compute the error between the target activities and the actual activities
    As_dec = lif_utils.lif_rate(
        Js_dec + (0 if do_decode_bias else biases_post[None, :]))

    return xs, As_post, As_dec


fig, axs = plt.subplots(1,
                        7,
                        figsize=(7.45, 1.0),
                        gridspec_kw={
                            "wspace": 0.2,
                            "width_ratios": [1, 1, 1, 0.5, 1, 1, 1],
                        })

i = 0
FUNCTIONS = [lambda x: x, lambda x: 2.0 * np.square(x) - 1.0]
for f in FUNCTIONS:
    for p_exc in [0, 0.5, 1.0]:
        xs, As_post, As_dec = run_single(np.random.RandomState(48711),
                                         p_exc=p_exc,
                                         f=f)
        _, _, As_dec2 = run_single(np.random.RandomState(48711),
                                   p_exc=p_exc,
                                   f=f,
                                   do_decode_bias=False)

        axs[i].plot(xs,
                    As_dec2[:, ::15],
                    'k--',
                    color='#606060',
                    linewidth=0.5)
        axs[i].plot(xs, As_post[:, ::15], 'k:', linewidth=0.75)
        axs[i].plot(xs, As_dec[:, ::15], 'k', linewidth=1)

        axs[i].add_patch(
            mpl.patches.Circle((0.1, 0.9),
                               0.05,
                               facecolor=mpl.cm.get_cmap('RdBu')(p_exc),
                               transform=axs[i].transAxes,
                               edgecolor='k',
                               linewidth=0.5,
                               zorder=100,
                               clip_on=False))
        axs[i].text(0.2,
                    0.89,
                    ratio_formatter(p_exc),
                    va='center',
                    ha='left',
                    transform=axs[i].transAxes)

        i += 1
    i += 1

utils.remove_frame(axs[3])

for i in range(len(axs)):
    if i == 3:
        continue
    axs[i].set_ylim(0, 100)
    axs[i].set_yticks([0, 50, 100])
    axs[i].set_yticks(np.arange(0, 101, 25), minor=True)
    if (i == 0) or (i == 4):
        axs[i].set_ylabel("$\\mathbf{a}^\mathrm{post}$ ($s^{-1}$)")
    else:
        axs[i].set_yticklabels([])
    axs[i].set_xlim(-1, 1)
    axs[i].set_xticks([-1, 0, 1])
    axs[i].set_xticks(np.linspace(-1, 1, 5), minor=True)
    axs[i].set_xlabel('$x$')

#axs[1].plot(As_post_id_2[:, ::7], 'k--', linewidth=0.75)
#axs[1].plot(As_dec_id_2[:, ::7], 'k', linewidth=0.75)

fig.legend( [
    mpl.lines.Line2D([0], [0], color='k', lw=0.75, linestyle=':'),
    mpl.lines.Line2D([0], [0], color='k', lw=1.0, linestyle='-'),
    mpl.lines.Line2D([0], [0], color='#606060', lw=0.5, linestyle='--')
], [
    "Target post-tuning",
    "Actual post-tuning",
    "Actual post-tuning (with intrinsic biases)",
], ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.3))

axs[0].text(-0.55, 1.1, '\\textbf{D}', va='top', ha='left', size=12, transform=axs[0].transAxes)
axs[4].text(-0.55, 1.1, '\\textbf{E}', va='top', ha='left', size=12, transform=axs[4].transAxes)

utils.save(fig)

