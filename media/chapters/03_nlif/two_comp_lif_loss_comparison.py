import scipy.optimize

from nef_synaptic_computation.multi_compartment_lif import *
import nef_synaptic_computation.lif_utils as lif_utils


def mk_two_comp_lif():
    neuron = Neuron()

    soma = Compartment(soma=True)
    soma.add_channel(CondChan(Erev=-65e-3, g=50e-9, name="leak"))

    dendrites = Compartment(name="dendrites")
    dendrites.add_channel(CondChan(Erev=20e-3, name="exc"))
    dendrites.add_channel(CondChan(Erev=-75e-3, name="inh"))
    dendrites.add_channel(CondChan(Erev=-65e-3, g=50e-9, name="leak"))

    neuron.add_compartment(soma)
    neuron.add_compartment(dendrites)
    neuron.connect("soma", "dendrites", 30e-9)

    return neuron.assemble()


# Generate some data

N = 30
model = mk_two_comp_lif()
gEs = np.linspace(0, 500e-9, N)
gIs = np.linspace(0, 500e-9, N)
gEss, gIss = np.meshgrid(gEs, gIs)
smpls = np.array((gEss.flatten(), gIss.flatten()))
As = model.iSom_empirical(smpls)['rate']

# Reconstruct currents
valid = As > 0
smpls = smpls.T[valid] * 1e9
As = As[valid]
Js = lif_utils.lif_rate_inv(As)
print("N = {}".format(np.sum(valid)))

def erf(a0, a1, a2, b0, b2):
    return np.sqrt(
        np.mean(np.square(Js - ((b0 + smpls[:, 0] - b2 * smpls[:, 1]) /
                                (a0 + a1 * smpls[:, 0] + a2 * smpls[:, 1]))),
                axis=-1))


def erf_subs(a0, a1, a2, b0, b2):
    return np.sqrt(
        np.mean(np.square(Js * (a0 + a1 * smpls[:, 0] + a2 * smpls[:, 1]) -
                          b0 - smpls[:, 0] + b2 * smpls[:, 1]),
                axis=-1)) / np.sqrt(np.mean(np.square((a0 + a1 * smpls[:, 0] + a2 * smpls[:, 1]))))



# Solve for parameters using NNLs
A = np.array(
    (Js, Js * smpls[:, 0], Js * smpls[:, 1], -np.ones_like(Js), smpls[:, 1])).T
b = np.array(smpls[:, 0])

# Compute the first parameter estimate
a0, a1, a2, b0, b2 = scipy.optimize.nnls(A.T @ A, A.T @ b)[0]

a0b, a1b, a2b, b0b, b2b = a0, a1, a2, b0, b2
for i in range(4):
    print(i, erf(a0b, a1b, a2b, b0b, b2b))
    w = (1.0 / (a0b + a1b * smpls[:, 0] + a2b * smpls[:, 1]))[:, None]
    a0b, a1b, a2b, b0b, b2b = scipy.optimize.nnls((w * A).T @ (w * A), (w * A).T @ (w[:, 0] * b))[0]

# Update the estimate
a0p, a1p, a2p, b0p, b2p = scipy.optimize.minimize(lambda x: erf(*x),
                                                  x0=[a0, a1, a2, b0, b2],
                                                  bounds=[(0.0, None)] * 5).x

print(erf(a0, a1, a2, b0, b2))
print(erf(a0p, a1p, a2p, b0p, b2p))
print(100.0 * (erf(a0, a1, a2, b0, b2) / erf(a0p, a1p, a2p, b0p, b2p) - 1.0))
print(100.0 * (erf(a0b, a1b, a2b, b0b, b2b) / erf(a0p, a1p, a2p, b0p, b2p) - 1.0))

def mk_range(x, r=0.1, N=50):
    if np.abs(x) < 1e-3:
        return np.linspace(0, r, N)
    else:
        return x * np.geomspace(1.0 - r, 1.0 + r, N)


def visualise_error_function(axs, f, swap=False):
    E_ref = f(a0, a1, a2, b0, b2)
    for i in range(5):
        for j in range(4):
            if i > 0:
                ax = axs[i - 1, j]
            else:
                ax = None

            params = [a0, a1, a2, b0, b2]
            paramsp = [a0p, a1p, a2p, b0p, b2p]

            rs = [0.1, 0.03, 0.15, 6.0, 0.1]
            ps = list(paramsp) if swap else list(params)
            ri = mk_range(paramsp[i], rs[i])
            rj = mk_range(paramsp[j], rs[j])
            sfss1, sfss2 = np.meshgrid(ri, rj)

            ps[i] = sfss1[..., None]
            ps[j] = sfss2[..., None]
            zs = f(*ps)

            vmax = 0.1

            if i > j and ax:
                C = ax.contour(rj,
                               ri,
                               zs.T,
                               vmin=0,
                               vmax=vmax,
                               levels=7,
                               colors=['white'],
                               linestyles=[':'],
                               linewidths=[0.5])
                ax.contourf(rj,
                            ri,
                            zs.T,
                            vmin=0,
                            vmax=vmax,
                            levels=C.levels,
                            cmap='inferno')

                ax.scatter([paramsp[j]], [paramsp[i]],
                           marker='o',
                           color='white',
                           clip_on=False,
                           zorder=100,
                           s=15,
                           linewidth=2.25)
                ax.scatter([paramsp[j]], [paramsp[i]],
                           marker='o',
                           color='k',
                           clip_on=False,
                           zorder=100,
                           s=15,
                           linewidth=1.0)

                ax.scatter([params[j]], [params[i]],
                           marker='+',
                           color='white',
                           clip_on=False,
                           zorder=100,
                           s=35,
                           linewidth=2.0)
                ax.scatter([params[j]], [params[i]],
                           marker='+',
                           color='k',
                           clip_on=False,
                           zorder=100,
                           s=25,
                           linewidth=1.0)

            if not ax is None:
                ax.set_aspect((rj[0] - rj[-1]) / (ri[0] - ri[-1]))
                ax.set_xlim(rj[0], rj[-1])
                ax.set_ylim(ri[0], ri[-1])
                ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3, steps=[2, 5]))
                ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3, steps=[2, 5]))
                ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
                ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))

                utils.outside_ticks(ax)
                labels = ["$a_0$", "$a_1$", "$a_2$", "$b_0$", "$b_2$"]
                if (i == 4) and (j < 4):
                    ax.set_xlabel(labels[j])
                else:
                    ax.set_xticklabels([])
                if (j == 0) and (i > 0):
                    ax.set_ylabel(labels[i])
                else:
                    ax.set_yticklabels([])

            if i <= j and ax:
                utils.remove_frame(ax)
                xs = np.linspace(-1, 1, 2)
                xss, _ = np.meshgrid(xs, xs)
                ax.contourf(xs, xs, -0.6 * np.ones_like(xss), levels=1, cmap='Greys', vmin=-1, vmax=1)
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_aspect(1)


fig = plt.figure(figsize=(6.5, 3.5))
gs1 = fig.add_gridspec(4, 4, left=0.06, right=0.45, wspace=0.2, hspace=0.2)
gs2 = fig.add_gridspec(4, 4, left=0.55, right=0.94, wspace=0.2, hspace=0.2)

axs1 = np.array([[fig.add_subplot(gs1[i, j]) for j in range(4)]
                 for i in range(4)])
axs2 = np.array([[fig.add_subplot(gs2[i, j]) for j in range(4)]
                 for i in range(4)])

axs1[0, 2].set_title("\\textbf{Loss function} $E$", x=-0.1, y=1.07)
axs2[0, 2].set_title("\\textbf{Substitute loss function} $E'$", x=-0.1, y=1.07)
axs1[0, 0].text(-0.85,
                1.15,
                "\\textbf{A}",
                size=12,
                va="bottom",
                ha="left",
                transform=axs1[0, 0].transAxes)
axs2[0, 0].text(-0.85,
                1.12,
                "\\textbf{B}",
                size=12,
                va="bottom",
                ha="left",
                transform=axs2[0, 0].transAxes)

fig.align_labels(axs=axs1)
fig.align_labels(axs=axs2)

visualise_error_function(axs1, erf, swap=True)
visualise_error_function(axs2, erf_subs)

utils.save(fig)

