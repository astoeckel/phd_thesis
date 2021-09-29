import scipy.linalg
import numpy as np


def reconstruct_lti(H, T=1.0, dampen=False, return_discrete=False):
    # Fetch the number of state dimensions and the number of samples
    q, N = H.shape

    # Canonicalise "dampen"
    if dampen and isinstance(dampen, str):
        dampen = {dampen}
    elif isinstance(dampen, bool):
        dampen = {"erasure"} if dampen else set()
    if (not isinstance(dampen,
                       set)) or (len(dampen - {"lstsq", "erasure"}) > 0):
        raise RuntimeError("Invalid value for \"dampen\"")

    # Time-reverse H
    Hrev = H[:, ::-1]

    # Compute the samples
    X = Hrev[:, :-1].T
    Y = (Hrev[:, 1:].T - Hrev[:, :-1].T) * N / T

    # Estimate the discrete system At, Bt
    At = np.linalg.lstsq(X, Y, rcond=None)[0].T
    Bt = Hrev[:, 0]

    # Add the Euler update
    At = np.eye(q) + At * T / N

    if "erasure" in dampen:
        enc, dec = enc, dec = H[:, 0], np.linalg.pinv(H)[0]
        At = At - np.outer(enc, dec) @ At
        Bt = Bt - np.outer(enc, dec) @ Bt

    if return_discrete:
        return At, Bt

    # Undo discretization (this is the inverse of discretize_lti)
    A = np.real(scipy.linalg.logm(At)) * N / T
    B = scipy.linalg.expm(-0.5 * A * T / N) @ Bt * np.sqrt(N)

    return A, B


def eval_lti(A, B, ts):
    return np.array([scipy.linalg.expm(A * t) @ B for t in ts])


def compute_errs(mk_basis_fun, q=11, q_plot=6, N=1000, dampen=False):
    T, dt = 1.0, 1 / N
    H = mk_basis_fun(q, N)

    ts = np.arange(0, 2 * T, dt)
    A, B = reconstruct_lti(H, dampen=dampen)
    H_ref = H[:, ::-1] * np.sqrt(N)
    H_rec = eval_lti(A, B, ts).T

    rmss = np.sqrt(np.mean(np.square(H_ref), axis=1))
    rmses = np.sqrt(np.mean(np.square(H_ref - H_rec[:, :N]), axis=1))

    zero_rms = np.sqrt(np.mean(np.square(H_rec[:, N:]), axis=1))

    return rmses / rmss, zero_rms


def plot_lti_basis(
    ax,
    mk_basis_fun,
    q=11,
    q_plot=6,
    N=1000,
    T=1.1,
    dt=1e-3,
    dampen=False,
):
    ts_H = np.linspace(0, 1, N)[::-1]
    H = mk_basis_fun(q, N)

    ts = np.arange(0, T, dt)
    A, B = reconstruct_lti(H, dampen=dampen)

    ys = eval_lti(A, B, ts)[:, :q_plot]
    for i in range(q_plot):
        ax.plot(ts, ys[:, i])
    ax.plot(ts_H, H.T[:, :q_plot] * np.sqrt(N), 'k:', lw=0.5)

    return A, B


def plot_errors_and_impulse_response(axs,
                                     funs,
                                     names,
                                     T=1.1,
                                     err_min=0.5e-3,
                                     y_letter=1.068,
                                     dampen=False,
                                     plot_zero_errs=False,
                                     utils=None):
    for j, fun in enumerate(funs):
        qs = np.arange(
            2, 12, 1,
            dtype=int)  #np.unique(np.geomspace(1, 21, 11, dtype=int))
        for i, q in enumerate(qs):
            errs, zero_errs = compute_errs(fun, q=q, dampen=dampen)
            if q > 1:
                errs = errs[1:]
                zero_errs = zero_errs[1:]
            for k in (range(2) if plot_zero_errs else range(1)):
                es = errs if k == 0 else zero_errs
                axs[k, j].plot(i,
                               np.mean(es),
                               '+',
                               color=['k', utils.blues[0]][k],
                               fillstyle='none',
                               markersize=5,
                               clip_on=False)
                axs[k,
                    j].errorbar(i,
                                np.mean(es),
                                np.array(
                                    (np.mean(es) - np.min(es),
                                     np.max(es) - np.mean(es))).reshape(2, -1),
                                color=['k', utils.blues[0]][k],
                                capsize=3.0,
                                lw=0.5,
                                clip_on=False)
        for k in (range(2) if plot_zero_errs else range(1)):
            axs[k, j].set_ylim([err_min, 1e-3][k], [1.1, 1.][k])
            axs[k, j].set_yscale('log')
            axs[k, j].set_ylabel(["NRMSE $E$", "RMS $E'$"][k])
            axs[k, j].set_xticks(np.arange(0, len(qs)))
            if (k == 1) or not plot_zero_errs:
                axs[k, j].set_xticklabels(qs)
                axs[k, j].set_xlabel("Number of basis functions $q$")
            else:
                axs[k, j].set_xticklabels([])

            axs[k, j].arrow(0.955,
                            0.135,
                            0.0,
                            -0.05,
                            width=0.0125,
                            linewidth=0.5,
                            facecolor='k',
                            edgecolor='white',
                            zorder=10,
                            transform=axs[k, j].transAxes)

        I = 2 if plot_zero_errs else 1
        A, B = plot_lti_basis(axs[I, j], fun, T=T, dampen=dampen)
        axs[I, j].text(1.0, 2.1, "$\\theta$", ha="center", va="bottom")
        axs[I, j].axvline(1.0, color='k', linestyle='--', linewidth=0.7)
        axs[I, j].set_xlim(0, T)
        axs[I, j].set_ylim(-2, 2)
        axs[I, j].set_yticks([-2, 0, 2])
        axs[I, j].set_yticks([-2, -1, 0, 1, 2], minor=True)
        axs[I, j].set_xlabel("Time $t$ (s)")
        axs[I, j].set_ylabel("$\\mathfrak{b}_i(t)$")

        if plot_zero_errs:
            axs[I, j].plot([0, 0.5], [1.0, 1.0],
                           'k--',
                           lw=1.5,
                           solid_capstyle='butt',
                           transform=axs[I, j].transAxes,
                           clip_on=False)
            axs[I, j].plot([0.5, 1.0], [1.0, 1.0],
                           '--',
                           lw=1.5,
                           color=utils.blues[0],
                           solid_capstyle='butt',
                           transform=axs[I, j].transAxes,
                           clip_on=False)

        axs[0, j].set_title("\\textbf{{{}}}".format(names[j]))
        axs[0, j].text(-0.265,
                       y_letter,
                       "\\textbf{{{}}}".format(chr(ord('A') + j)),
                       size=12,
                       transform=axs[0, j].transAxes,
                       ha="left",
                       va="baseline")

        axs[0, 0].get_figure().align_labels(axs[:, j])

