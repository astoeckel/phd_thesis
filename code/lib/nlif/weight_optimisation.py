#  libnlif -- Multi-compartment LIF simulator and weight solver
#  Copyright (C) 2017-2021  Andreas Stöckel
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

import copy
import numpy as np
import tqdm

from .parameter_optimisation import loss as _loss
from .internal.adam import Adam
from .solver import Solver


def _check_shape(reduced_sys, As, Js, W, W_mask=None):
    # If W_mask is None, do an all-to-all connection
    if W_mask is None:
        W_mask = np.ones((reduced_sys.n_inputs, As.shape[-1]), dtype=np.bool)

    # Make sure that As, Js, W and W_mask are arrays
    As, Js, W, W_mask = np.asarray(As), np.asarray(Js), W, np.asarray(W_mask)

    # Make sure that the input dimensions match
    if W.ndim != 2:
        raise RuntimeError("W must be a two-dimensional matrix")
    if W.shape[0] != reduced_sys.n_inputs:
        raise RuntimeError(
            "First dimension of W must match the number of inputs")
    if As.shape[-1] != W_mask.shape[1]:
        raise RuntimeError(
            "Last dimension of W and As must be equal to the number of neurons"
        )
    if Js.shape != As.shape[:-1]:
        raise RuntimeError("Dimensionality mismatch between Js and As")
    if W.shape != W_mask.shape:
        raise RuntimeError("W and W_mask must have the same shape")

    # Reshape things for easier iteration
    output_shape = Js.shape
    Js = Js.reshape(-1)
    As = As.reshape(-1, As.shape[-1])
    N = len(Js)  # Number of samples

    return As, Js, W, W_mask, N, output_shape


def loss(reduced_sys, As, Js, W, J_th=None, reg=0.0):
    """
    Computes the loss function for the reduced system reduced_sys and the given
    pre-activities, weights and target currents. Implements sub-threshold
    relaxation if J_th is not set to None.
    """
    As, Js, W, _, N, output_shape = _check_shape(reduced_sys, As, Js, W)
    return (_loss(reduced_sys, As @ W.T, Js, J_th=J_th) +
            reg * N * np.sum(np.square(W))).reshape(output_shape)


def loss_gradient_numerical(reduced_sys,
                            As,
                            Js,
                            W,
                            W_mask=None,
                            J_th=None,
                            eta=1e-9,
                            reg=0.0):
    """
    Computes the weight gradient for the reduced system reduced_sys numerically.
    This function is slow, numerically unstable, and mainly for testing.
    Use `loss_gradient` instead.

    Returns a tensor dW containing gradients for each input sample.
    """
    As, Js, W, W_mask, N, output_shape = _check_shape(reduced_sys, As, Js, W,
                                                      W_mask)

    def dE(W1, W2):
        return (loss(reduced_sys, As, Js, W2, J_th, reg) -
                loss(reduced_sys, As, Js, W1, J_th, reg)) / eta

    k, m = reduced_sys.n_inputs, As.shape[-1]
    dW = np.zeros((N, k, m))

    for r, s in zip(*np.where(W_mask)):
        W1, W2 = np.copy(W), np.copy(W)
        W1[r, s] -= 0.5 * eta
        W2[r, s] += 0.5 * eta
        dW[:, r, s] = dE(W1, W2)

    return dW.reshape((*output_shape, k, m))


def loss_gradient(reduced_sys,
                  As,
                  Js,
                  W,
                  W_mask=None,
                  J_th=None,
                  reg=0.0,
                  return_ravelled=False):
    """
    Computes the parameter gradient for the reduced system reduced_sys
    numerically.

    Returns matrices dA, da_const, dB, db_const containing the mean gradient of
    the loss function over the entire batch.
    """
    As, Js, W, W_mask, N, output_shape = _check_shape(reduced_sys, As, Js, W,
                                                      W_mask)

    # Fetch some useful variables
    k, m = reduced_sys.n_inputs, As.shape[-1]
    dW = np.zeros((N, k, m))

    # Compute the inputs
    gs = As @ W.T

    for l in range(N):
        # Fetch some matrices that we'll need throughout the computation
        A_inv = np.linalg.inv(reduced_sys.A_dyn(gs[l]))
        b = reduced_sys.b_dyn(gs[l])
        v_eq = reduced_sys.v_eq(gs[l])
        Ap, Bp, c = reduced_sys.A, reduced_sys.B, reduced_sys.c
        J, g = Js[l], gs[l]

        # Compute the outer derivative
        J_hat = np.inner(reduced_sys.c, v_eq - reduced_sys.v_som)
        if (J_th is None) or (J > J_th):
            E = -2 * (J_hat - J)
        elif J_hat > J_th:
            E = -2 * (J_hat - J_th)
        else:
            E = 0

        # Compute the final derivative
        dW[l] = E * (
            np.einsum('ik,kj,kr,s,j,i->rs', A_inv, A_inv, Ap, As[l], b, c) +
            np.einsum('ij,jr,s,i->rs', A_inv, Bp, As[l], c)) * W_mask

        # Account for the regularisation term
        dW[l] += 2.0 * N * reg * W * W_mask

    if return_ravelled:
        return np.array([dW[i][W_mask]
                         for i in range(N)]).reshape(*output_shape, -1)
    else:
        return dW.reshape((*output_shape, k, m))


def _optimise_common(reduced_sys, As_train, Js_train, W, W_mask, As_test,
                     Js_test, J_th, N_epochs, normalise_error):
    # Clone the weights
    W = np.copy(W) * reduced_sys.in_scale

    # Prepare the input data
    if (As_test is None) != (Js_test is None):
        raise RuntimeError(
            "Noth As_test and Js_test must either be None or not None.")
    As_train, Js_train, W, W_mask, N_train, _ = _check_shape(
        reduced_sys, As_train, Js_train, W, W_mask)
    if not Js_test is None:
        As_test, Js_test, _, _, N_test, _ = _check_shape(
            reduced_sys, As_test, Js_test, W, W_mask)
    else:
        As_test, Js_test, N_test = None, None, None

    # Scale the output according to the reduced system scaling factors
    def rms(x):
        return np.sqrt(np.mean(np.square(x)))

    Js_train = Js_train * reduced_sys.out_scale
    rms_train = rms(Js_train) if normalise_error else 1.0

    if not Js_test is None:
        Js_test = Js_test * reduced_sys.out_scale
        rms_test = rms(Js_test) if normalise_error else 1.0

    # Initialise the errors
    errs_train, errs_test = np.ones((2, N_epochs + 1)) * np.nan

    def update_err(i):
        errs_train[i] = np.sqrt(
            np.mean(
                loss(res["reduced_sys"], As_train, Js_train, res["W"],
                     res["J_th"]))) / rms_train
        if not Js_test is None:
            errs_test[i] = np.sqrt(
                np.mean(
                    loss(res["reduced_sys"], As_test, Js_test, res["W"],
                         res["J_th"]))) / rms_test

    res = {
        "reduced_sys": reduced_sys,
        "W": W,
        "W_mask": W_mask,
        "As_train": As_train,
        "Js_train": Js_train,
        "N_train": N_train,
        "As_test": As_test,
        "Js_test": Js_test,
        "N_test": N_test,
        "J_th": None if J_th is None else J_th * reduced_sys.out_scale,
        "errs_train": errs_train,
        "errs_test": errs_test,
        "update_err": update_err,
    }

    update_err(0)

    return res


def optimise_sgd(
        reduced_sys,
        As_train,
        Js_train,
        W,
        W_mask=None,
        As_test=None,
        Js_test=None,
        J_th=None,
        N_epochs=100,
        N_batch=10,
        alpha=1e-5,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-7,  # From TensorFlow
        reg=1e-3,
        rng=np.random,
        progress=True,
        normalise_error=True):
    # Check some parameters, do some pre-processing
    data = _optimise_common(reduced_sys, As_train, Js_train, W, W_mask,
                            As_test, Js_test, J_th, N_epochs, normalise_error)

    # Fetch references at the parameter matrices
    p = (data["W"], )

    # Instantiate the optimiser and perform the actual optmisation
    optimiser = Adam(alpha=alpha, beta1=beta1, beta2=beta2, epsilon=epsilon)
    for i in tqdm.tqdm(range(N_epochs)) if progress else range(N_epochs):
        # Divide the input into batches
        sample_idcs = list(range(data["N_train"]))
        rng.shuffle(sample_idcs)
        batch_idcs = np.linspace(0,
                                 data["N_train"],
                                 int(np.ceil(data["N_train"] / N_batch)) + 1,
                                 dtype=int)
        batch_idcs0, batch_idcs1 = batch_idcs[:-1], batch_idcs[1:]

        # Iterate over each batch
        for i0, i1 in zip(batch_idcs0, batch_idcs1):
            # Compute the mean gradient
            dp = (np.mean(loss_gradient(data["reduced_sys"],
                                        data["As_train"][i0:i1],
                                        data["Js_train"][i0:i1],
                                        data["W"],
                                        data["W_mask"],
                                        data["J_th"],
                                        reg=reg),
                          axis=0), )

            # Update the parameters
            optimiser.step(p, dp)

            # Make sure that the weights are non-negative
            p[0][...] = np.maximum(p[0], 0.0)

        # Compute the error
        data["update_err"](i + 1)

    # Return the updated system as well as the recorded errors
    if Js_test is None:
        return data["W"] / reduced_sys.in_scale, data["errs_train"]
    else:
        return data["W"] / reduced_sys.in_scale, data["errs_train"], data[
            "errs_test"]


def optimise_bfgs(reduced_sys,
                  As_train,
                  Js_train,
                  W,
                  W_mask=None,
                  As_test=None,
                  Js_test=None,
                  J_th=None,
                  N_epochs=100,
                  reg=1e-3,
                  rng=np.random,
                  progress=True,
                  normalise_error=True):
    # Check some parameters, do some pre-processing
    data = _optimise_common(reduced_sys, As_train, Js_train, W, W_mask,
                            As_test, Js_test, J_th, N_epochs, normalise_error)

    # Assemble the optimisation bounds
    n1 = int(np.sum(data["W_mask"]))
    bounds = [(0.0, None) for _ in range(n1)]

    # Fetch the initial parameters
    x0 = data["W"][data["W_mask"]]

    # Loss function to minimise
    def f(x):
        Wp = np.copy(data["W"])
        Wp[data["W_mask"]] = x
        E = np.sum(
            loss(data["reduced_sys"], data["As_train"], data["Js_train"], Wp,
                 data["J_th"], reg))
        return E

    # Loss-function gradient
    def df(x):
        Wp = np.copy(data["W"])
        Wp[data["W_mask"]] = x
        dE = np.sum(loss_gradient(data["reduced_sys"],
                                  data["As_train"],
                                  data["Js_train"],
                                  Wp,
                                  data["W_mask"],
                                  data["J_th"],
                                  reg=reg,
                                  return_ravelled=True),
                    axis=0)
        return dE

    # Callback function writing the current parameter set to the output and
    # updating the recorded error
    with tqdm.tqdm(total=N_epochs, disable=not progress) as pbar:
        # Callback function responsible for tracking errors and updating the
        # progress bar
        i = [0]

        def callback(x):
            i[0] += 1
            data["W"][data["W_mask"]] = x
            data["update_err"](i[0])
            pbar.update(1)

        # Run the actual optimisation
        import scipy.optimize
        scipy.optimize.minimize(fun=f,
                                x0=x0,
                                jac=df,
                                method='L-BFGS-B',
                                bounds=bounds,
                                options={
                                    "maxiter": N_epochs,
                                    "ftol": 0.0,
                                    "gtol": 0.0,
                                    "disp": False,
                                },
                                callback=callback)

    # Return the updated system as well as the recorded errors
    if Js_test is None:
        return data["W"] / reduced_sys.in_scale, data["errs_train"]
    else:
        return data["W"] / reduced_sys.in_scale, data["errs_train"], data[
            "errs_test"]


def optimise_trust_region(reduced_sys,
                          As_train,
                          Js_train,
                          W=None,
                          W_mask=None,
                          As_test=None,
                          Js_test=None,
                          J_th=None,
                          N_epochs=50,
                          alpha1=1e0,
                          alpha2=1e0,
                          alpha3=1e-3,
                          reg1=1e-3,
                          reg2=1e-6,
                          gamma=0.95,
                          use_sanathanan_koerner=False,
                          progress=True,
                          normalise_error=True,
                          debug=False,
                          parallel_compile=True,
                          tol=1e-6):
    # If no weights are given, first solve an auxiliary problem to find some
    # weights
    if W is None:
        W0 = np.zeros((reduced_sys.n_inputs, As_train.shape[-1]))
        W, _ = optimise_trust_region(reduced_sys,
                                     As_train,
                                     Js_train * 0.0,
                                     W0,
                                     W_mask,
                                     N_epochs=1,
                                     reg1=1e-1,
                                     progress=False,
                                     normalise_error=False,
                                     debug=debug,
                                     parallel_compile=parallel_compile,
                                     tol=tol)

    # Check some parameters, do some pre-processing
    data = _optimise_common(reduced_sys, As_train, Js_train, W, W_mask,
                            As_test, Js_test, J_th, N_epochs, normalise_error)

    # Instantiate the optimiser and perform the actual optmisation
    solver = Solver(debug=debug, parallel_compile=parallel_compile)
    for i in tqdm.tqdm(range(N_epochs)) if progress else range(N_epochs):
        scale = np.power(gamma, -i)
        data["W"] = solver.nlif_solve_weights_iter(
            data["reduced_sys"],
            data["As_train"],
            data["Js_train"].reshape(1, *data["Js_train"].shape),
            data["W"].reshape(1, *data["W"].shape),
            data["W_mask"].reshape(1, *data["W_mask"].shape),
            reg1=reg1,
            reg2=reg2,
            alpha1=alpha1,
            alpha2=alpha2,
            alpha3=alpha3 * scale,
            J_th=data["J_th"],
            use_sanathanan_koerner=use_sanathanan_koerner,
            progress_callback=None,
            tol=tol)[0]
        data["update_err"](i + 1)

    # Return the updated system as well as the recorded errors
    if Js_test is None:
        return data["W"] / reduced_sys.in_scale, data["errs_train"]
    else:
        return data["W"] / reduced_sys.in_scale, data["errs_train"], data[
            "errs_test"]

