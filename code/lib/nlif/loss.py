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


###############################################################################
# REFERENCE IMPLEMENTATION OF THE LOSS FUNCTION                               #
###############################################################################


def make_loss_function(
        A_pre_exc,
        A_pre_inh,
        j_tar,
        j_th=None,
        ws=None,
        lambda_=0.0,
        grad=False,
        pre_mult_ws=False):
    """
    A reference implementation of the loss-function that is being optimized by
    libbioneuronqp.

    This function returns the loss function L(w+, w-) for the optimization
    problem specified by the parameters to this function. If desired, a function
    computing the Jacobi matrix is returned instead (this required autograd to
    be installed).

    This function is not explicitly used anywhere in the code, but the
    unit-tests shipping with libbioneuronqp ensure that the solution returned by
    libbioneuronqp correspond to a global minimum of this function by making
    sure that the Jacobi matrix is zero (this is sufficient, since this loss
    function is convex).


    Parameters
    ==========

    A_pre_exc:
        A (N x n+) matrix containing the excitatory pre-population activities.
    A_pre_inh:
        A (N x n-) matrix containing the inhibitory pre-population activities.
    j_tar:
        A N-dimensional vector containing the desired target currents.
    j_th:
        The threshold current of the post-neuron. Currents below this threshold
        should not evoke any neural activity in the post-neuron. If set to None,
        j_th is assumed to be -Infinity, which is equivalent to saying that all
        currents evoke some activity in the target.
    ws:
        A vector (a0, a1, a2, b0, b1, b2) containing the post synapse/neuron
        model parameters. Set to None or (0, 1, -1, 1, 0, 0) for a standard
        current-based LIF neuron.
    lambda_:
        The L2 regularisation factor.
    grad:
        If set to true, a function returning the Jacobi matrix (i.e. gradient
        for each input weight) is returned. The Python package "autograd" must
        be installed for this to work.
    pre_mult_ws:
        If true, directly applies the model parameters to the inputs A, a_tar,
        and j_th, instead of evaluating H in the returned loss function. This
        parameter mostly exists for debugging purposes.


    Returns
    =======

    A function L(w_exc, w_inh) that maps the given excitatory and inhibitory
    weights onto a loss value. The goal of weight optimization is to minimize
    the loss.

    If grad=False (the default), w_exc and w_inh may be matrices, in which case
    the loss is computed for multiple weight vectors at a time. The first matrix
    dimension must be equal to the number of pre-neurons.

    If grad=True, two matrices corresponding to the gradient of each coefficient
    of w_exc, w_inh is returned.
    """

    # Import autograd, if the gradient is requested
    if grad:
        import autograd
        import autograd.numpy as np
    else:
        import numpy as np

    # Make sure all given matrices have the right dimensionality
    assert A_pre_exc.ndim == 2
    assert A_pre_inh.ndim == 2
    assert j_tar.ndim == 1
    assert A_pre_exc.shape[0] == A_pre_inh.shape[0] == j_tar.shape[0]
    n_samples = A_pre_exc.shape[0] # = A_pre_inh.shape[0] = J_tar.shape[0]
    n_pre_exc = A_pre_exc.shape[1]
    n_pre_inh = A_pre_inh.shape[1]

    # Sume handy aliases for subscripting
    iw0, iw1, iw2 = 0, n_pre_exc, n_pre_exc + n_pre_inh

    # Destructure the given model weights
    if ws is None:
        ws = (0, 1, -1, 1, 0, 0)
    a0, a1, a2, b0, b1, b2 = ws

    # Apply the model parameters to the input arrays if pre_mult_ws is set
    # to true. Then set the model parameters to a neutral value.
    if pre_mult_ws:
        A_pre_exc = ((a1 - j_tar * b1).T * A_pre_exc.T).T
        A_pre_inh = ((a2 - j_tar * b2).T * A_pre_inh.T).T
        j_tar = j_tar * b0 - a0
        if not j_th is None:
            j_th = j_th * b0 - a0
        a0, a1, a2, b0, b1, b2 = (0.0, 1.0, 1.0, 1.0, 0.0, 0.0)

    # Split the pre-population activities, as well as the target currents
    # into super- and sub-threshold samples
    if j_th is None: # Same as j_th = -np.inf
        j_th = -np.inf
    is_subth = j_tar < j_th
    is_supth = j_tar >= j_th

    # Samples with a super-threshold target current, also called "valid"
    # constraints in the C++ code.
    A_pre_exc_supth = A_pre_exc[is_supth]
    A_pre_inh_supth = A_pre_inh[is_supth]
    j_tar_supth = j_tar[is_supth]

    # Samples with a sub-threshold target current, also called "invalid"
    # constraints in the C++ code.
    A_pre_exc_subth = A_pre_exc[is_subth]
    A_pre_inh_subth = A_pre_inh[is_subth]
    j_tar_subth = j_tar[is_subth]

    # Implementation of the synaptic nonlinearity model H[., .]
    def H(A_pre_exc, A_pre_inh, w_exc, w_inh):
        return (
            (a0 + a1 * A_pre_exc @ w_exc + a2 * A_pre_inh @ w_inh) /
            (b0 + b1 * A_pre_exc @ w_exc + b2 * A_pre_inh @ w_inh))

    # Actual implementation of the loss-function
    def _L(w_combined):
        # Split ws into w_exc and w_inh (this is for autograd)
        w_exc, w_inh = w_combined[iw0:iw1], w_combined[iw1:iw2]

        # Make sure the input arrays are 2D
        w_exc = np.atleast_1d(w_exc)
        w_inh = np.atleast_1d(w_inh)

        # Make sure the given matrices have the right shape
        assert w_exc.shape[0] == n_pre_exc
        assert w_inh.shape[0] == n_pre_inh

        # Compute j_hat, i.e., the actually decoded output current
        j_hat_supth = H(A_pre_exc_supth, A_pre_inh_supth, w_exc, w_inh)
        j_hat_subth = H(A_pre_exc_subth, A_pre_inh_subth, w_exc, w_inh)

        # Compute the quadratic super-threshold loss
        L_quad_sup = np.sum(
            np.square(j_hat_supth.T - j_tar_supth.T).T, axis=0)

        # Compute the quadratic sub-threshold loss
        L_quad_sub = np.sum(np.square(
            np.maximum(j_th, j_hat_subth).T - j_th).T, axis=0)

        # Compute the regularisation error
        L_reg = lambda_ * n_samples * (np.sum(np.square(w_exc), axis=0) +
                                       np.sum(np.square(w_inh), axis=0))

        return L_quad_sup + L_quad_sub + L_reg

    # Turns _L into a function that receives independent excitatory
    # and inhibitory weights
    def wrap_L(_L):
        def L(w_exc, w_inh):
            res = _L(np.concatenate((w_exc, w_inh)))
            return (res[iw0:iw1], res[iw1:iw2]) if grad else res
        return L

    # If "grad" is set to "true", use autograd to compute the Lagrangian
    # of the above function, otherwise just return L
    if grad:
        return wrap_L(autograd.grad(_L))
    else:
        return wrap_L(_L)

