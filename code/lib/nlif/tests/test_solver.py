#!/usr/bin/env python3

#  libnlif -- Multi-compartment LIF simulator and weight solver
#  Copyright (C) 2020-2021  Andreas Stöckel
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np

from nlif.loss import make_loss_function
from nlif.solver import Solver
from nlif.tests.utils.nef import Ensemble


def do_compute_weights_single_neuron_ref(A_pre_exc,
                                         A_pre_inh,
                                         j_tar,
                                         j_th,
                                         ws,
                                         lambda_,
                                         nonneg,
                                         tol=1e-6,
                                         w_exc_init=None,
                                         w_inh_init=None):
    # Fetch the gradient of the loss function
    dL = make_loss_function(A_pre_exc,
                            A_pre_inh,
                            j_tar,
                            j_th,
                            ws,
                            lambda_,
                            grad=True,
                            pre_mult_ws=True)

    # Fetch the number of pre-neurons
    n_pre_exc = A_pre_exc.shape[1]
    n_pre_inh = A_pre_inh.shape[1]

    # Initialise w_exc and w_inh
    if w_exc_init is None:
        w_exc = np.zeros(n_pre_exc)
    else:
        w_exc = np.copy(w_exc_init)

    if w_inh_init is None:
        w_inh = np.zeros(n_pre_inh)
    else:
        w_inh = np.copy(w_inh_init)

    # Update the weights
    eta = 1e-3
    for i in range(10000):
        # Compute the gradient
        d_w_exc, d_w_inh = dL(w_exc, w_inh)

        # Compute the new weights
        w_exc_new = w_exc - eta * d_w_exc
        w_inh_new = w_inh - eta * d_w_inh

        # Apply the nonnegativity constraint
        if nonneg:
            w_exc_new = np.maximum(0, w_exc_new)
            w_inh_new = np.maximum(0, w_inh_new)

        # Compute the amount of change
        d_w_exc_abs = np.abs(w_exc_new - w_exc)
        d_w_inh_abs = np.abs(w_inh_new - w_inh)

        # Replace the weights with the old weights
        w_exc = w_exc_new
        w_inh = w_inh_new

        # Abort if the change is smaller than the target tolerance
        chg = np.max(np.concatenate((d_w_exc_abs, d_w_inh_abs))) / eta
        if i % 1000 == 0:
            print("Gradient descent i={}; chg={:0.4e}".format(i, chg))
        if chg < tol:
            return w_exc, w_inh, True

    return w_exc, w_inh, False


def do_test_bioneuronqp_single(Apre,
                               Jpost,
                               ws,
                               connection_matrix,
                               iTh,
                               nonneg,
                               reg,
                               renormalise,
                               tol=1e-6,
                               plot=False):
    # Check the dimensions
    assert Apre.ndim == 2
    assert Jpost.ndim == 2
    assert connection_matrix.ndim == 3
    assert Apre.shape[0] == Jpost.shape[0]
    assert connection_matrix.shape[0] == 2
    assert connection_matrix.shape[1] == Apre.shape[1]
    assert connection_matrix.shape[2] == Jpost.shape[1]
    n_samples = Apre.shape[0]
    n_pre = Apre.shape[1]
    n_post = Jpost.shape[1]

    # Solve for weights using bioneuronqp
    w_exc, w_inh = Solver().two_comp_solve(
        Apre=Apre,
        Jpost=Jpost,
        ws=ws,
        connection_matrix=connection_matrix,
        iTh=iTh,
        nonneg=nonneg,
        renormalise=renormalise,
        tol=tol,
        reg=reg,
        progress_callback=None,
        warning_callback=None,
        max_iter=10000,
    )

    # For each post-neuron, compute the weights using the above gradient descent
    # solver
    for i in range(n_post):
        # Fetch the excitatory and inhibitory pre-neurons
        A_pre_exc = Apre[:, connection_matrix[0, :, i]]
        A_pre_inh = Apre[:, connection_matrix[1, :, i]]

        # Fetch the target current
        j_tar = Jpost[:, i]

        # Compute the difference in weights
        w_exc_subs = w_exc[connection_matrix[0, :, i], i]
        w_inh_subs = w_inh[connection_matrix[1, :, i], i]

        # Compute the weights using the reference solver
        w_exc_ref, w_inh_ref, ok = do_compute_weights_single_neuron_ref(
            A_pre_exc=A_pre_exc,
            A_pre_inh=A_pre_inh,
            j_tar=j_tar,
            j_th=iTh,
            ws=ws,
            lambda_=reg,
            nonneg=nonneg,
            tol=tol,
            w_exc_init=w_exc_subs,
            w_inh_init=w_inh_subs,
        )

        if not ok:
            print("Maximum number of iterations reached")

        # Compute the maximum absolute error
        W0 = np.concatenate((w_exc_subs, w_inh_subs))
        W1 = np.concatenate((w_exc_ref, w_inh_ref))
        err = np.max(np.abs(W0 - W1))
        if err > tol:
            print("Error larger than tolerance! {} > {}".format(err, tol))
            ok = False

        # Make sure the error is within two orders of magnitude of the given
        # tolerance
        if not ok:
            print("Weights are returned by libbioneuronqp: ", w_exc_subs,
                  w_inh_subs)
            print("Weights after gradient descent: ", w_exc_ref, w_inh_ref)

        # Plot the error landscape, if requested
        if plot:
            import matplotlib.pyplot as plt

            L = nlif.loss.make_loss_function(A_pre_exc,
                                             A_pre_inh,
                                             j_tar,
                                             iTh,
                                             ws,
                                             reg,
                                             pre_mult_ws=True)
            L(w_exc_subs, w_inh_subs)
            L(w_exc_ref, w_inh_ref)

            n, n_exc, n_inh = W0.size, w_exc_subs.size, w_inh_subs.size
            fig, axs = plt.subplots(n, n, figsize=(12, 12))
            for i in range(n):
                for j in range(n):
                    ax = axs[i, j]
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if i + 1 == n:
                        ax.set_xlabel("$w_{{{}}}$".format(j))
                    if j == 0:
                        ax.set_ylabel("$w_{{{}}}$".format(i))

                    # Shorthands for the actual weights
                    w0x, w0y, w1x, w1y = W0[i], W0[j], W1[i], W1[j]

                    # Compute the range over which to evaluate the loss
                    # function
                    wx0, wy0 = min(w0x, w1x), min(w0y, w1y)
                    wx1, wy1 = max(w0x, w1x), max(w0y, w1y)
                    dwx, dwy = max(1e-2, wx1 - wx0), max(1e-2, wy1 - wy0)

                    # Compute the weight range
                    wsx = np.linspace(wx0 - dwx, wx1 + dwx, 35)
                    wsy = np.linspace(wy0 - dwy, wy1 + dwy, 36)
                    wssx, wssy = np.meshgrid(wsx, wsy)

                    # Compute the weights
                    WEs = np.zeros((n_exc, *wssx.shape))
                    WIs = np.zeros((n_inh, *wssx.shape))

                    for ii in range(n_exc):
                        if ii == i:
                            WEs[ii] = wssx
                        elif ii == j:
                            WEs[ii] = wssy
                        else:
                            WEs[ii] = np.ones(wssx.shape) * W1[ii]
                    for jj in range(n_inh):
                        if jj + n_exc == i:
                            WIs[jj] = wssx
                        elif jj + n_exc == j:
                            WIs[jj] = wssy
                        else:
                            WIs[jj] = np.ones(wssx.shape) * W1[jj + n_exc]

                    n_sweep = np.prod(wssx.shape)
                    E = L(WEs.reshape(n_exc, n_sweep),
                          WIs.reshape(n_inh, n_sweep)).reshape(wssx.shape)
                    C = ax.contourf(wsx, wsy, np.log10(E))
                    ax.contour(wsx,
                               wsy,
                               np.log10(E),
                               levels=C.levels,
                               colors=('white', ),
                               linestyles=('--', ))

                    ax.plot(W0[i],
                            W0[j],
                            'x',
                            markeredgewidth=2,
                            color='white')
                    ax.plot(W1[i],
                            W1[j],
                            '+',
                            markeredgewidth=2,
                            color='white')


#            fig.savefig('weight_space.png', bbox_inches='tight', transparent=True)
#            fig.savefig('weight_space.pdf', bbox_inches='tight', transparent=True)
            plt.show()

        assert ok


def do_test_from_dump(filename):
    data = np.load(filename, allow_pickle=True).item()

    do_test_bioneuronqp_single(
        Apre=data['Apre'],
        Jpost=data['Jpost'],
        ws=data['ws'],
        connection_matrix=data['connection_matrix'],
        iTh=data['iTh'],
        nonneg=data['nonneg'],
        renormalise=data['renormalise'],
        tol=data['tol'],
        reg=data['reg'],
        plot=True,
    )


def do_test_bioneuronqp(n_pre,
                        n_post,
                        n_samples,
                        ws,
                        iTh,
                        nonneg,
                        reg,
                        renormalise,
                        tol=1e-6):
    # Create two neuron ensembles
    ens1 = Ensemble(n_pre, 1)
    ens2 = Ensemble(n_post, 1)

    # Create some random connectivity
    connection_matrix = np.zeros((2, n_pre, n_post), dtype=np.bool)
    while np.sum(connection_matrix) == 0:
        connection_matrix = np.random.choice(
            [True, False], (2, n_pre, n_post)).astype(np.bool)

    # Sample the pre and post populations
    xs = np.linspace(-1, 1, n_samples).reshape(1, -1)
    Apre = ens1(xs).T
    Jpost = ens2.J(xs).T

    # Run the test code
    try:
        do_test_bioneuronqp_single(
            Apre=Apre,
            Jpost=Jpost,
            ws=ws,
            connection_matrix=connection_matrix,
            iTh=iTh,
            nonneg=nonneg,
            renormalise=renormalise,
            tol=tol,
            reg=reg,
        )
    except Exception as e:
        np.save(
            "test_bioneuronqp_dump.npy", {
                "Apre": Apre,
                "Jpost": Jpost,
                "ws": ws,
                "connection_matrix": connection_matrix,
                "iTh": iTh,
                "nonneg": nonneg,
                "renormalise": renormalise,
                "tol": tol,
                "reg": reg,
            })
        print("Problem dump saved to test_bioneuronqp_dump.npy")
        raise e


def test_bioneuronqp():
    np.random.seed(58198)

    for nonneg in [True, False]:
        for renormalise in [True, False]:
            for iTh in [None, -1.0, 0.0]:
                for reg in [1e-3, 1e-2, 1e-1]:
                    print("nonneg={} renormalise={} iTh={} reg={}".format(
                        nonneg, renormalise, iTh, reg))
                    for repeat in range(10):
                        # Select some random parameters
                        n_pre = np.random.randint(1, 10)
                        n_post = 1
                        n_samples = np.random.randint(50, 100)

                        # Select a random neuron model
                        ws = np.array(
                            [
                                0,  # a0
                                1,  # a1
                                np.random.uniform(-2, -0.5),  # a2
                                np.random.uniform(0.5, 1.5),  # b0
                                np.random.uniform(0, 1),  # b1
                                np.random.uniform()
                            ],  # b2
                            dtype=np.float64)

                        # Run the benchmark
                        do_test_bioneuronqp(n_pre, n_post, n_samples, ws, iTh,
                                            nonneg, reg, renormalise)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        for filename in sys.argv[1:]:
            do_test_from_dump(filename)
    else:
        test_bioneuronqp()

