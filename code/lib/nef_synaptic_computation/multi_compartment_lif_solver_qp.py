#!/usr/bin/env python3

#   This file is part of NEF Synaptic Computation
#   (c) Andreas Stöckel 2017, 2018
#
#   NEF Synaptic Computation is free software: you can redistribute it and/or
#   modify it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   NEF Synaptic Computation is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License along with
#   NEF Synaptic Computation.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np

################################################################################
# LINEAR AND QUADRATIC PROGRAMMING UTILITIES                                   #
################################################################################

import cvxopt


class CvxoptParamGuard:
    """
    Class used to set relevant cvxopt parameters and to reset them once
    processing has finished or an exception occurs.
    """

    def __init__(self, tol=1e-12, disp=False):
        self.options = {
            "abstol": tol,
            "feastol": tol,
            "reltol": 10 * tol,
            "show_progress": disp
        }

    def __enter__(self):
        # Set the given options, backup old options
        for key, value in self.options.items():
            if key in cvxopt.solvers.options:
                self.options[key] = cvxopt.solvers.options[key]
            cvxopt.solvers.options[key] = value
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the old cvxopt options
        for key, value in self.options.items():
            cvxopt.solvers.options[key] = value
        return self


def _solve_qp(Pqp,
              qqp,
              Gqp=None,
              hqp=None,
              Aqp=None,
              bqp=None,
              tol=1e-12,
              disp=True):
    """
    Solves the given quadtratic programing problem

    min    x^T P x + q^T x
    s.t.   Gx <= h
           Ax  = b

    """

    # Solve the QP problem
    with CvxoptParamGuard(tol=tol, disp=disp) as guard:
        res = cvxopt.solvers.qp(
            P=cvxopt.matrix(Pqp.astype(np.double)),
            q=cvxopt.matrix(qqp.astype(np.double)),
            G=None if Gqp is None else cvxopt.matrix(Gqp.astype(np.double)),
            h=None if hqp is None else cvxopt.matrix(hqp.astype(np.double)),
            A=None if Aqp is None else cvxopt.matrix(Aqp.astype(np.double)),
            b=None if bqp is None else cvxopt.matrix(bqp.astype(np.double)))

    return np.array(res["x"])


def _check_basis_pursuit_params(C, d, A, b, G, h):
    """
    Used internally to make sure that the given arguments have the right
    dimensionality.
    """

    # Replace zero-sized matrices with None
    C = None if (not (C is None)) and C.size == 0 else C
    d = None if (not (d is None)) and d.size == 0 else d
    A = None if (not (A is None)) and A.size == 0 else A
    b = None if (not (b is None)) and b.size == 0 else b
    G = None if (not (G is None)) and G.size == 0 else G
    h = None if (not (h is None)) and h.size == 0 else h

    # Make sure either both of the variables in the pairs G, h and A, b are
    # "None" or both of them are not
    assert not (C is None) and not (d is None), "C, d must not be None"
    assert (G is None) == (h is None), "Both G and h must be None"
    assert (A is None) == (b is None), "Both A and b must be None"

    # Make sure d, b, h are vectors
    assert (d is None) or (d.size == d.shape[0]), "d must be a vector"
    assert (b is None) or (b.size == b.shape[0]), "b must be a vector"
    assert (h is None) or (h.size == h.shape[0]), "h must be a vector"

    # Make sure A, C use the same number of variables
    if not A is None:
        assert A.shape[1] == C.shape[1], \
                "Second dimension of A, C, and G (number of variables in " + \
                "the system) must be the same"

    # Make sure C, d and G, h have the same number of constraints
    assert d.shape[0] == C.shape[0], \
            "First dimension of C, d (number of equality constraints) " + \
            "must be the same"
    if not G is None:
        assert G.shape[0] == h.shape[0], \
                "First dimension of G, h (number of inequality " + \
                " constraints) must be the same"

        # Make sure G and C have the same number of variables
        assert G.shape[1] == C.shape[1], \
                "Second dimension of A, C, and G (number of variables in " + \
                "the system) must be the same"

    return C, d, A, b, G, h


def solve_linearly_constrained_quadratic_loss(C,
                                              d,
                                              A=None,
                                              b=None,
                                              G=None,
                                              h=None,
                                              P=None,
                                              tol=1e-12,
                                              disp=False):
    """
    Solves a problem similar to the basis pursuit problem, but using the L2
    instead of the L1 norm, thus turning the problem into a quadratic program.

    min    || Cx  - d ||_2
    s.t.      Ax  = b
              Gx <= h

    This function is mainly meant for benchmarking the impact of using the L1
    instead of the L2 norm.
    """

    # Make sure the dimensionalities of the input matrices are correct
    C, d, A, b, G, h = _check_basis_pursuit_params(C, d, A, b, G, h)

    # Compute the matrices for the QP problem
    Pqp = C.T @ C
    qqp = -C.T @ d

    # Add the given P matrix to Pqp
    if not P is None:
        assert Pqp.shape == P.shape, "P has an incorrect shape"
        Pqp += P

    # Solve the QP problem
    return _solve_qp(Pqp, qqp, G, h, A, b, tol=tol, disp=disp)


def generate_orthogonal_linear_functions(xss, ymin, ymax):
    """
    Randomly projects xss with a linear projection. As many dimensions as
    possible are orthogonal in the projection. Rescales each dimension such
    that the values are within the range specified by ymin, ymax. Returns a
    (n x k) matrix, where n is the number of samples and k the number of output
    dimensions.

    xss: Samples in representation space, (n x m) where n is the number of
    samples and m the number of input dimensions.

    ymin: k-dimensional vector describing the minimum output value for each
    target dimension.

    ymax: k-dimensional vector describing the maximum output value for each
    target dimension.
    """

    assert np.all(ymin.shape == ymax.shape), \
        "ymin and ymax must have the same dimensionality"
    assert np.all(ymin <= ymax), \
        "ymin must be smaller than ymax along all dimensions"
    assert ymin.shape[0] == ymin.size, "ymin must be a vector"
    assert ymax.shape[0] == ymax.size, "ymax must be a vector"

    n_smpls = xss.shape[0]  # Number of samples
    n_in_dims = xss.shape[1]  # Number of input dimensions
    n_out_dims = ymin.shape[0]  # Number of output dimensions

    # Generate a random "orthonormal" matrix
    C = np.random.normal(0, 1, (n_in_dims, n_out_dims))
    C1, _, C2 = np.linalg.svd(C, full_matrices=False)
    C = C1 if C1.shape == C.shape else C2

    # Compute the target currents per compartment
    X = xss[...]
    Y = X @ C

    # Normalise to the given Ymin/Ymax range
    Ymin, Ymax = np.min(Y, axis=0), np.max(Y, axis=0)
    yscale = (ymax - ymin) / (Ymax - Ymin)
    yoffs = ymin - Ymin * yscale
    Y = Y * yscale[None, :] + yoffs[None, :]

    return Y


def compute_external_currents_from_voltages(V, MC, vsom):
    """
    For each compartment computes the external current required to arrive at the
    specified equilibrium potentials.

    Parameters
    ----------

    V: Matrix containing the desired membrane potentials for each compartment.
    Shape is (n_samples, n_comp).

    MC: matrix describing the connectivity of the entire dendtritic tree. Must
    be symmetric with zero diagonal. Shape is (n_comp + 1, n_comp + 1). Where
    the zeroth index corresponds to the soma. Individual entries must be in µS.

    vsom: Assumed constant somatic membrane potential.

    Return values
    -------------

    Matrix of shape (n_samples, n_comp ) with the external currents flowing
    into each compartment that will cause the given voltages.
    """

    # Make sure MC and V have the correct shape
    assert MC.shape[0] == MC.shape[1] == (V.shape[1] + 1)

    # Make sure MC is symmetric and the diagonal is zero
    assert np.all(np.abs(MC - MC.T) < 1e-15), "MC must be symmetric"
    assert np.all(np.diag(MC) == 0.0), "Diagonal of MC must be zero"

    # Fetch the number of samples and compartments
    n_smpls = V.shape[0]
    n_comp = V.shape[1]

    # Construct the current matrix
    J = np.zeros((n_smpls, n_comp + 1))
    for i in range(n_comp + 1):
        for j in range(n_comp + 1):
            Vi = V[:, i - 1] if i > 0 else np.ones(n_smpls) * vsom
            Vj = V[:, j - 1] if j > 0 else np.ones(n_smpls) * vsom
            J[:, i] += (Vi - Vj) * MC[i, j]
    return J



################################################################################
# PROBLEM DESCRIPTION                                                          #
################################################################################

def make_objective_qp(Jtar, Apre, MA, MAconst, MB, MBconst, MC, vsom, vmin, vmax, vrest,
                            is_subth=None, Wcon=None, regW=1e-9, regV=1e-9):
    """
    Constructs the objective function and its derivative for the given target
    currents, pre-synaptic activity, neuron model parameters, and synaptic
    connectivity. The objective function is the length of the gradient vector
    of the Lagrangian 𝓛(x, λ) = f(x) + λg(x).

    Jtar: target somatic current. Shape is (n_smpls, ).

    Apre: pre-popolation activities for each sample. Shape is (n_smpls, n_pre).

    MA: neuron-model specific parameter vector. Shape is (n_comp, n_input).

    MAconst: neuron-model specific parameter matrix. Shape is (n_comp, ).

    MB: neuron-model specific parameter matrix. Shape is (n_comp, n_input).

    MBconst: neuron-model specific parameter vector. Shape is (n_comp, ).

    MC: connectivity matrix describing the connectivity of the dendritic tree
    for the given neuron model,. Shape is (n_comp + 1, n_comp + 1). First
    row/column with index zero corresponds to the soma.

    vsom: assumed somatic voltage 

    is_subth: vector of boolean flags indicating whether the corresponding
    sample corresponds to a subthreshold current. For such subthreshold samples,
    this routine assumes that target voltages smaller than the given values will
    result in a subthreshold current. Shape is (n_smpls, ). Set this vector to
    "None" if all samples should be assumed to correspond to above-threshold
    currents.

    Wcon: Boolean connectivity matrix determining which pre-synaptic neuron is
    connected to which input. Shape is (n_input, n_pre). If None, all
    pre-synaptic neurons are connected to all input sites.

    reg: Weight regularisation.

    """

    # Construct Wcon if not given, make sure it is a matrix of bools
    if Wcon is None:
        Wcon = np.ones((MA.shape[1], Apre.shape[1]), dtype=np.bool)
    Wcon = Wcon.astype(np.bool)

    # Construct is_subth if not given, make sure it is a vector of bools
    if is_subth is None:
        is_subth = np.zeros((Jtar.shape[0]), dtype=np.bool)
    is_subth = is_subth.astype(np.bool)

    # Make sure all the input variables have the correct shape

    # Number of samples
    assert Jtar.shape[0] == is_subth.shape[0] == Apre.shape[0], \
        "Jtar, is_subth and Apre have the same first dimension (number " + \
        "of samples)"

    # Number of pre-synaptic neurons
    assert Apre.shape[1] == Wcon.shape[1], \
        "Apre and Wcon must have the same second dimension (number of " + \
        "pre-synaptic neurons)"

    # Number of compartments
    assert MA.shape[0] == MAconst.shape[0] == MB.shape[0] == \
           MBconst.shape[0] == MC.shape[0] == MC.shape[1], \
        "Incompatible Jtar, MC, MAconst, MA, MBconst, MB (number of " + \
        "compartments)"

    # Number of inputs
    assert MA.shape[1] == MB.shape[1] == Wcon.shape[0], \
        "Incompatible MA, MB, Wcon (number of neuron model inputs)"

    # Make sure MC is symmetric and the diagonal is zero
    assert np.all(np.abs(MC - MC.T) < 1e-15), "MC must be symmetric"
    assert np.all(np.diag(MC) == 0.0), "Diagonal of MC must be zero"

    # Make sure Jtar is a vector
    assert Jtar.size == Jtar.shape[0], "Jtar must be a vector"
    Jtar = Jtar.flatten()

    # Make sure vmin, vmax, vrest are vectors
    assert vmin.size == vmin.shape[0] == (MA.shape[0] - 1), "vmin must be a vector"
    assert vmax.size == vmax.shape[0] == (MA.shape[0] - 1), "vmax must be a vector"
    assert vrest.size == vrest.shape[0] == (MA.shape[0] - 1), "vmax must be a vector"

    # Fetch all the dimensions
    n_smpls = Jtar.shape[0]
    n_smpls_subth = np.sum(is_subth)
    n_smpls_supth = n_smpls - n_smpls_subth
    n_pre = Apre.shape[1]
    n_comp = MC.shape[0]
    n_input = MA.shape[1]

    # Synaptic weights
    n_syn = np.sum(Wcon)
    i0w, i1w = 0, 0 + n_syn

    # Slack variables for samples with subthreshold currents
    n_slack = n_smpls_subth
    i0s, i1s = i1w, i1w + n_slack

    # Variables corresponding to voltages
    n_volt = (n_comp - 1) * n_smpls
    i0v, i1v = i1s, i1s + n_volt

    # Total number of variables
    n_var = i1v

    # Compute the weight distribution matrices according to Wcon and Apre and
    # which compartment is directly affected by which synapse
    Ψ = np.zeros((n_smpls, n_input, n_syn))
    Ω = np.zeros((n_comp, n_syn))
    for i in range(n_smpls):
        i_syn = 0
        for j in range(n_input):
            for k in range(n_pre):
                if Wcon[j, k]:
                    # Copy the pre-synaptic activity to this specific synapse
                    Ψ[i, j, i_syn] = Apre[i, k]

                    # Determine which compartments are affected by this synapse
                    for l in range(n_comp):
                        if MA[l, j] != 0.0 or MB[l, j] != 0.0:
                            Ω[l, i_syn] = 1

                    # Increase the synapse index
                    i_syn += 1

    # Compute the variable distribution matrices
    def ᗧ(i0, i1, n, r=0):
        I = np.zeros((i1 - i0, n))
        I[list(range(i1 - i0)), list(range(i0 + (i1 - i0) * r, i0 + (i1 - i0) * (r + 1)))] = 1
        return I
    Iw, Is, Iv = ᗧ(i0w, i1w, n_var), ᗧ(i0s, i1s, n_var), ᗧ(i0v, i1v, n_var)
    Ivl = list(map(lambda i: ᗧ(i0v, i0v + (n_comp - 1), n_var, i), range(n_smpls)))

    # Multiplicators used to convert weights, currents, and voltages to a common
    # order of magnitude
    ωS, ωA, ωV = 1e9, 1e9, 1e3

    # Compute the slack variable distribution matrix
    S, i_s = np.zeros((n_smpls, n_smpls_subth)), 0
    for i in range(n_smpls):
        if is_subth[i]:
            S[i, i_s] = 1
            i_s += 1

    # Model parameters used in the somatic current computation
    MC_f = MC[1:, 0].T
    MM_f = (vsom * MA[0] + MB[0])
    M0_f = vsom * MAconst[0] + MBconst[0] - Jtar

    # Reduced model parameters used in the current distribution calculation
    MC_g = MC[1:, 1:]
    MA_g = MA[1:]
    MB_g = MB[1:]
    MAconst_g = MAconst[1:]
    MBconst_g = MBconst[1:] + vsom * MC[1:, 0] if n_comp > 1 else 0.0

    # Construct the primary objective function
    C_prim = np.zeros((n_smpls, n_var))
    for i in range(n_smpls):
        C_prim[i] = (MM_f @ Ψ[i] @ Iw + MC_f @ Ivl[i] + S[i] @ Is) * ωA
    d_prim = -np.copy(M0_f) * ωA

    # Objective functions describing the regularisation factors
    C_reg_w = Iw * ωS * regW
    C_reg_v = Iv * ωV * regV
    d_reg_w = np.zeros(n_syn)
    d_reg_v = np.tile(vrest, (n_smpls,)) * ωV * regV

    # Assemble the objective describing the non-orthogonality penalty
#    P = np.zeros((n_smpls * n_input, n_smpls * n_input))
#    for i in range(n_smpls):
#        for j0 in range(n_input):
#            for j1 in range(j0 + 1, n_input):
#                i0 = i * n_input + j0
#                i1 = i * n_input + j1
#                P[i0, i1] = 1
#    Ψr = Ψ.reshape((n_smpls * n_input, n_syn))
#    P = Iw.T @ Ψr.T @ P @ Ψr @ Iw

    # Inequality constraints describing the min/max values for each variable
    G_w_nonneg = -Iw * ωS
    h_w_nonneg = np.zeros(n_syn)
    G_v_min = -Iv * ωV
    h_v_min = -np.tile(vmin, (n_smpls,)) * ωV
    G_v_max = Iv * ωV
    h_v_max = np.tile(vmax, (n_smpls,)) * ωV
    G_s_nonneg = -Is * ωA
    h_s_nonneg = np.zeros(n_slack)

    # Assemble the constraints into a single matrix
    G = np.concatenate((G_w_nonneg, G_v_min, G_v_max, G_s_nonneg))
    h = np.concatenate((h_w_nonneg, h_v_min, h_v_max, h_s_nonneg))

    def mk_x0():
        x0 = np.zeros(n_var)
        x0[i0w:i1w] = 0.0
        x0[i0s:i1s] = 0.0
        x0[i0v:i1v] = np.tile(vrest, (n_smpls, ))
        return x0

    def mk_x_init(xss, dec_min=0.0, dec_max=1e-6, reg=1e-9):
        # Generate the original parameter vector
        x0 = mk_x0()

        # Compute initial weights that decode a set of orthogonal linear
        # functions from the pre-population
        fns = generate_orthogonal_linear_functions(
                xss, np.ones(n_input) * dec_min, np.ones(n_input) * dec_max)

        # Solve for weights that non-negatively decode the generated function
        sigma = (np.max(Ψ) ** 2) * reg
        for i in range(n_input):
            # Determine which synapses contribute to this input
            mask = (np.sum(Ψ[:, i, :], axis=0) > 0)
            n_syn_l = np.sum(1 * mask)

            # Setup the target function || C_tar @ x - d_tar ||
            C_tar = Ψ[:, i, mask] * ωS
            d_tar = fns[:, i] * ωS

            # Regularisation term
            C_reg = np.eye(n_syn_l) * sigma * ωS
            d_reg = np.zeros(n_syn_l) * ωS

            # Non-negativity
            G = -np.eye(n_syn_l) * ωS
            h = np.zeros(n_syn_l) * ωS

            # Solve the QP problem
            C = np.concatenate((C_tar, C_reg))
            d = np.concatenate((d_tar, d_reg))
            x0[i0w:i1w][mask] = solve_linearly_constrained_quadratic_loss(
                    C, d, None, None, G, h)[:, 0]

        return fix_v(x0)

    def mk_qp(x0=None, tr=0, w_mask=None):
        # Construct the initial parameter vector if none was given
        if x0 is None:
            x0 = mk_x0()

        # If w_mask is None, optimize all weights
        if w_mask is None:
            w_mask = np.ones(n_syn, dtype=np.bool)

        # If the mask is a 2D matrix and the last dimension is correct, combine
        # the arrays using logical OR
        if len(w_mask.shape) == 2 and w_mask.shape[1] == n_syn:
            w_mask = (np.sum(w_mask, axis=0) > 0).flatten()

        # Make sure x0 and w_mask are 1D arrays and have the correct size
        assert x0.size == x0.shape[0] == n_var, "x0 has the wrong size"
        assert w_mask.size == w_mask.shape[0] == n_syn, "w_mask has the wrong size"
        x0 = x0.flatten()
        w_mask = w_mask.astype(np.bool).flatten()

        # Assemble the trust region
        C_tr = np.concatenate((Iw * ωS, Is * ωA, Iv * ωV)) * tr
        d_tr = np.concatenate((Iw @ x0 * ωS, Is @ x0 * ωA, Iv @ x0 * ωV)) * tr

        # Assemble the matrix describing the objective function
        C = np.concatenate((C_prim, C_reg_w, C_reg_v, C_tr))
        d = np.concatenate((d_prim, d_reg_w, d_reg_v, d_tr))

        # Assemble the matrix describing the equality constraint
        A = np.zeros((n_volt, n_var))
        b = np.zeros((n_volt))
        for i in range(n_smpls):
            i0, i1 = i * (n_comp - 1), (i + 1) * (n_comp - 1)

            # First-order Taylor expansion of the term (MA_g @ Iw @ x) * (Ivl[i] @ x)
            f00 = MA_g @ Ψ[i] @ Iw
            f01 = f00 @ x0
            f10 = Ivl[i]
            f11 = f10 @ x0
            fx0 = f01 * f11
            Dfx0 = (f00 * f11[:, None] + f10 * f01[:, None])

            # Assemble the function
            A[i0:i1] = MB_g @ Ψ[i] @ Iw + MC_g @ Ivl[i] + np.diag(MAconst_g) @ Ivl[i] + Dfx0
            b[i0:i1] = -(MBconst_g + fx0 - Dfx0 @ x0)

        # Transform the system to not rely on weights that are currently masked
        # out
        x_mask = np.concatenate((w_mask, np.ones(n_var - n_syn, dtype=np.bool)))
        x_mask_neg = np.logical_not(x_mask)

        Cp = C[:, x_mask]
        dp = d - C[:, x_mask_neg] @ x0[x_mask_neg]
        Ap = A[:, x_mask]
        bp = b - A[:, x_mask_neg] @ x0[x_mask_neg]
        Gp = G[:, x_mask]
        hp = h - G[:, x_mask_neg] @ x0[x_mask_neg]

        return Cp, dp, Ap, bp, Gp, hp, x_mask

    def fix_v(x):
        # Make sure x0 has the correct size
        assert x.size == x.shape[0] == n_var, "x has the wrong size"

        # If we only have one compartment (the somatic compartment) there is
        # nothing we can do here
        if n_comp <= 1:
            return x

        # Make sure x is a 1D array
        x_shape, x = x.shape, x.flatten()

        for i in range(n_smpls):
            i0, i1 = i0v + i * (n_comp - 1), i0v + (i + 1) * (n_comp - 1)

            A = MC_g + np.diag(MAconst_g + MA_g @ Ψ[i] @ Iw @ x)
            b = MBconst_g + MB_g @ Ψ[i] @ Iw @ x
            x[i0:i1] = np.linalg.solve(-A, b)

        # Undo flattening x
        return x.reshape(x_shape)


    return {
        "mk_x0": mk_x0,
        "mk_x_init": mk_x_init,
        "mk_qp": mk_qp,
        "fix_v": fix_v,
        "Ω": Ω,
        "Ψ": Ψ,
        "i0w": i0w,
        "i1w": i1w,
        "i0s": i0s,
        "i1s": i1s,
        "i0v": i0v,
        "i1v": i1v
    }
