#   Temporal Basis Transformation Network
#   Copyright (C) 2020, 2021  Andreas Stöckel
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as
#   published by the Free Software Foundation, either version 3 of the
#   License, or (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
@file common.py

This file contains code shared between the TensorFlow Keras Layer and the
reference implementation. This is mostly some utility code that makes sure that
input dimensions are correct and that computes reshape operations for certain
inputs.

@author Andreas Stöckel
"""


class Mode:
    """
    Class used for the "Forward" and "Inverse" constants defined below.
    """
    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return self._name


Forward = Mode("Forward")
Inverse = Mode("Inverse")


def coerce_collapse(collapse):
    """
    Used to coerce the "collapse" parameter passed to TemporalBasisTrafo into a
    pre- and post-collapse
    """

    # If collapse was explicitly set to "True" or "False" return the
    if (collapse is True) or (collapse is False):
        return collapse, collapse
    elif hasattr(collapse, '__len__') and len(collapse) == 2:
        pre_collapse, post_collapse = collapse
        return bool(pre_collapse), bool(post_collapse)
    else:
        raise RuntimeError(
            "Invalid valid for collapse. Must either be a boolean or a tuple "
            "of booleans (pre_collapse, post_collapse)")


def coerce_params(H, units, pad, collapse, mode, normalize, rcond, np):
    # Make sure the matrix H has the right shape and type
    H = np.asarray(H).astype(dtype=np.float32, copy=True)
    if (H.ndim != 2) or (H.size == 0):
        raise RuntimeError("Invalid basis transformation matrix H. Must be "
                           "two-dimensional and have a non-zero size.")

    # Extract q and N
    q, N = H.shape

    # Make sure rcond is valid; when using the inverse mode, compute the
    # pseudo-inverse of H
    rcond = float(rcond)
    if rcond <= 0.0:
        raise RuntimeError("rcond must be strictly positive")

    # Make sure units is positive
    units = int(units)
    if units <= 0:
        raise RuntimeError("Number of units must be strictly positive.")

    # Make sure "pad" is a boolean.
    pad = bool(pad)

    # Coerce the "collapse" parameter
    collapse = coerce_collapse(collapse)

    # Make sure "mode" is valid
    if not ((mode is Forward) or (mode is Inverse)):
        raise RuntimeError("Mode must either be either "
                           "temporal_basis_transformation_network.Forward or "
                           "temporal_basis_transformation_network.Inverse")

    # If normalize is true, make sure that the rows of H have unit length
    if normalize:
        H /= np.linalg.norm(H, axis=1)[:, None]

    # Compute the pseudo-inverse of H if we're running in inverse mode. The
    # pseudo-inverse is exactly equal to the transpose if H is orthogonal; the
    # pseudo-inverse is needed for non-orthogonal bases such as the LDN basis.
    if mode is Inverse:
        H = np.linalg.pinv(H, rcond=rcond)

    return q, N, H, units, pad, collapse, mode


def check_input_shape(S, units, q, collapse, mode):
    """
    Checks a given input shape for compatibility. The input shape S is a
    tuple or list. Wildcard dimensions may either be indicated by using "-1"
    or "None". Raises a "RuntimeError" exception with a descriptive error
    message if the shape is not valid. Returns a normalized shape tuple in
    which all wildcard dimensions are replaced by -1.
    """
    # Make sure the direction argument is valid
    if not ((mode is Forward) or (mode is Inverse)):
        raise RuntimeError(
            "Invalid value for mode, must be Forward or Inverse")

    # Normalize wildcard entries, make sure there is at most one wildcard
    # entry
    S = tuple(-1 if s is None else int(s) for s in S)
    if not all(((s == -1) or (s > 0)) for s in S):
        raise RuntimeError("Shape entries most be strictly positive or -1")
    if sum(s == -1 for s in S) > 1:
        raise RuntimeError("More than one wildcard entry in the shape")

    # The expected input shape depends on whether we're in forward or
    # inverse mode; in inverse mode the input shape is actually the output
    # shape of a corresponding forward network, hence we have to take
    # "collapse" into account as well.
    pre_collapse, _ = coerce_collapse(collapse)
    if mode is Forward:
        S_expected = (-1, units)
    elif pre_collapse:  # and (mode is Inverse)
        S_expected = (-1, units * q)
    else:  # mode is Inverse
        S_expected = (-1, units, q)

    # Check whether the last dimensions are as expected
    ok = len(S) >= len(S_expected)
    for i in range(1, len(S_expected) + 1):
        ok = ok and ((S[-i] == S_expected[-i]) or (S_expected[-i] == -1))

    # Raise an exception with a nice error message if something went wrong
    if not ok:
        fmt = "TemporalBasisTrafo: Invalid shape. " \
              "Got ({S}) but expected (..., {S_expected})"
        raise RuntimeError(
            fmt.format(S=", ".join(map(str, S)),
                       S_expected=", ".join("*" if s == -1 else str(s)
                                            for s in S_expected)))

    # Return the normalized shape
    return S


def compute_shapes_and_permutations(S, units, q, N, pad, collapse, mode):
    """
    For a given network configuration computes the shapes and permutations. The
    order these are returned in corresponds to the order in which these
    operations are applied in the final network. In case a shape or permuation
    is a no-op, "None" is returned instead.

    input_shape_pre:
        Shape the input is being reshaped into before the input permutation is
        applied.

    input_perm:
        Permutation applied to the input.

    input_shape_post:
        Shape the input is reshaped into after the permutation was applied.

    output_shape_pre:
        Output shape the convolution output should be re-shaped into before
        the output permutation is applied.

    output_perm:
        Permutation applied to the output.

    output_shape_post:
        Reshape operation applied to the output after the permutation is
        applied.

    M_in, M_out, M_pad:
        The number of input and output samples, as well as the number of
        samples that need to be padded with zeros.
    """

    # Make sure the given shape is valid, canonicalize wildcard entries
    S = check_input_shape(S, units, q, collapse, mode)

    # Fetch some convenient variables
    l, (pre_collapse, post_collapse) = len(S), coerce_collapse(collapse)

    # Compute the dimension M_in is stored in. Fetch M_in and compute the
    # number of output dimensions taking padding into account.
    M_in_dim = l - 2 if ((mode is Forward) or (pre_collapse)) else l - 3
    M_in = S[M_in_dim]
    M_out = M_in if (pad or (mode is Inverse)) else max(1, M_in - N + 1)
    M_pad = 0 if (mode is Inverse) else (M_out - M_in + N - 1)

    # Compute the intermediate shape. Multiply all input dimensions
    # except for the dimension containing M_in_dim. Since there is at most
    # one wildcard dimension (checked above), negative numbers indicate
    # that there is one wildcard present
    n_batch = 1
    for i, s in enumerate(S):
        n_batch *= (1 if i == M_in_dim else s)
    n_batch = max(-1, n_batch)  # negative values should be -1 (wildcard)

    # I we're in inverse mode, we incorrectly included q in the above
    # computation. Undo this by dividing by q.
    if (n_batch > 0) and (mode is Inverse):
        assert n_batch % q == 0  # Code errored out above if this is not true
        n_batch //= q

    # Per default, do nothing
    input_shape_pre, input_perm, input_shape_post = None, None, None
    output_shape_pre, output_perm, output_shape_post = None, None, None

    if mode is Forward:
        # Move the dimensions containing the units to the left of the samples
        input_perm = tuple(i if i < l - 2 else (2 * l - 3 - i)
                           for i in range(l))

        # If there is only one output dimension and no padding is required, we
        # use a matrix multiplication and do not need the following reshape
        if (M_out > 1) or (M_pad > 0):
            # Collapse all dimensions to the left of the input samples
            input_shape_post = (n_batch, M_in, 1)

        # Reshape such that the dimensions corresponding to the individual
        # units are separated out
        output_shape_pre = tuple((*S[:-2], units, M_out, q))

        # Move the units to the right of the samples
        output_perm = tuple((*range(0, M_in_dim), M_in_dim + 1, M_in_dim, l))

        # Depending on whether or not a post-collapse is required, collapse
        # units and q.
        if post_collapse:
            output_shape_post = tuple((*S[:M_in_dim], M_out, units * q))
    elif mode is Inverse:
        # If pre_collapse is true, we first need to un-collapse the input.
        if pre_collapse:
            input_shape_pre = tuple((*S[:M_in_dim], M_in, units, q))

        # If required, collapse the output dimensions
        if post_collapse:
            output_shape_post = tuple((*S[:M_in_dim], M_out, units * N))

    return input_shape_pre, input_perm, input_shape_post, \
           output_shape_pre, output_perm, output_shape_post, \
           M_in, M_out, M_pad

