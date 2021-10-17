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

import itertools
import numpy as np
import dlop_ldn_function_bases as bases

from temporal_basis_transformation_network import Forward, Inverse
from temporal_basis_transformation_network.keras import TemporalBasisTrafo
from temporal_basis_transformation_network.reference import trafo as trafo_ref


def _iterative_reference_implementation(q, xs):
    """
    Reference implementation feeding an input signal "xs" through a Legendre
    Delay Network.
    """
    assert xs.ndim == 1
    N, = xs.shape
    A, B = bases.discretize_lti(1.0 / N, *bases.mk_ldn_lti(q))
    res, m = np.zeros((N, q)), np.zeros(q)
    for i in range(N):
        m = A @ m + B * xs[i]
        res[i] = m
    return np.asarray(res, dtype=np.float32)


def _test_impulse_response_generic(q, N, dims_pre=tuple(), dims_post=(1, )):
    """
    Generates some test data of the shape (*dims_pre, N, *dims_post) and
    feeds it into a TemporalBasisTrafo instances with a Legendre Delay Network
    basis. Computes the reference Delay Network output and compares the output
    to the output returned by the TemporalBasisTrafo.
    """

    # Generate some test data
    q, N = 10, 100
    rng = np.random.RandomState(49818)
    xs = rng.randn(*dims_pre, N, *dims_post)

    # Generate the reference output; iterate over all batch dimensions
    H = bases.mk_ldn_basis(q, N, normalize=False)
    ys_ref = np.zeros((*dims_pre, N, *dims_post, q))
    dims_cat = tuple((*dims_pre, *dims_post))
    for idcs in itertools.product(*map(range, dims_cat)):
        # Split the indices into the pre and post indices
        idcs_pre = idcs[:len(dims_pre)]
        idcs_post = idcs[len(dims_pre):]

        # Assemble the source/target slice
        sel = tuple((*idcs_pre, slice(None), *idcs_post))
        ys_ref[sel] = _iterative_reference_implementation(q, xs[sel])

    # Create the tensorflow network with the LDN basis
    ys = TemporalBasisTrafo(H,
                            dims_post[0],
                            collapse=False,
                            pad=True,
                            normalize=False)(xs).numpy()
    assert ys.shape == ys_ref.shape

    # Perform the same computation using the numpy reference implementation
    ys_ref_np = trafo_ref(xs,
                          H,
                          dims_post[0],
                          collapse=False,
                          pad=True,
                          normalize=False)

    # Make sure that all the implementations tested here produce approximately
    # the same output
    np.testing.assert_allclose(ys, ys_ref, atol=1e-6)
    np.testing.assert_allclose(ys, ys_ref_np, atol=1e-6)
    np.testing.assert_allclose(ys_ref, ys_ref_np, atol=1e-6)


def test_impulse_response_single_batch_single_unit():
    # Most simple case. Single unit, 100 input samples
    _test_impulse_response_generic(10, 100)


def test_impulse_response_multiple_batches_single_unit():
    # Same as above, but add some arbitrary batch dimensions
    _test_impulse_response_generic(10, 100, (1, 1))
    _test_impulse_response_generic(10, 100, (5, 1))
    _test_impulse_response_generic(10, 100, (5, 3, 1))
    _test_impulse_response_generic(10, 100, (5, 2, 3, 1))


def test_impulse_response_single_batch_multiple_units():
    # Single batch dimension; seven individual units
    _test_impulse_response_generic(10, 100, tuple(), (7, ))


def test_impulse_response_multiple_batches_multiple_unit():
    # Arbitrary batch dimensions but seven individual units
    _test_impulse_response_generic(10, 100, (1, ), (7, ))
    _test_impulse_response_generic(10, 100, (5, ), (7, ))
    _test_impulse_response_generic(10, 100, (5, 3), (7, ))
    _test_impulse_response_generic(10, 100, (5, 2, 3), (7, ))


def test_inverse_compression():
    # Generate some test data
    q, N = 10, 100
    rng = np.random.RandomState(49818)
    xs = rng.randn(N, 1)

    H = bases.mk_dlop_basis(q, N)

    ys = TemporalBasisTrafo(H, pad=False)(xs).numpy()
    xs_inv = TemporalBasisTrafo(H, 1, mode=Inverse)(ys).numpy()

    ys_ref = trafo_ref(xs, H, pad=False)
    xs_inv_ref = trafo_ref(ys_ref, H, mode=Inverse)

    np.testing.assert_allclose(ys, ys_ref, atol=1e-6)
    np.testing.assert_allclose(xs_inv, xs_inv_ref, atol=1e-6)


def test_inverse_full_reconstruction():
    # Generate some test data
    q, N = 100, 100
    rng = np.random.RandomState(49818)
    xs = rng.randn(N, 1)

    H = bases.mk_dlop_basis(q, N)

    ys = TemporalBasisTrafo(H, pad=False)(xs).numpy()
    xs_inv = TemporalBasisTrafo(H, 1, mode=Inverse)(ys).numpy()

    ys_ref = trafo_ref(xs, H, pad=False)
    xs_inv_ref = trafo_ref(ys_ref, H, 1, mode=Inverse)

    np.testing.assert_allclose(ys, ys_ref, atol=1e-6)
    np.testing.assert_allclose(xs_inv, xs.T, atol=1e-6)
    np.testing.assert_allclose(xs_inv_ref, xs.T, atol=1e-6)


def test_kernel_normalisation():
    q, N = 16, 100

    H = bases.mk_ldn_basis(q, N, normalize=False)
    H = H.astype(np.float32)

    H_norm = H / np.linalg.norm(H, axis=1)[:, None]
    H_inv = np.linalg.pinv(H, rcond=1e-6)
    H_norm_inv = np.linalg.pinv(H_norm, rcond=1e-6)

    # Test trainable vs not trainable (these are slightly different code-paths)
    for trainable in [False, True]:
        # No normalisation, forward mode
        layer = TemporalBasisTrafo(H, normalize=False, trainable=trainable)
        np.testing.assert_allclose(H, layer.kernel(), atol=1e-6)
        layer.build((None, N, 1))
        np.testing.assert_allclose(H, layer.kernel(), atol=1e-6)

        # Normalisation, forward mode
        layer = TemporalBasisTrafo(H, normalize=True, trainable=trainable)
        np.testing.assert_allclose(H_norm, layer.kernel(), atol=1e-6)
        layer.build((None, N, 1))
        np.testing.assert_allclose(H_norm, layer.kernel(), atol=1e-6)

        # No normalisation, inverse mode
        layer = TemporalBasisTrafo(H,
                                   normalize=False,
                                   mode=Inverse,
                                   trainable=trainable)
        np.testing.assert_allclose(H_inv, layer.kernel(), atol=1e-6)
        layer.build((None, 1, q))
        np.testing.assert_allclose(H_inv, layer.kernel(), atol=1e-6)

        # Normalisation, inverse mode
        layer = TemporalBasisTrafo(H,
                                   normalize=False,
                                   mode=Inverse,
                                   trainable=trainable)
        np.testing.assert_allclose(H_inv, layer.kernel(), atol=1e-6)
        layer.build((None, 1, q))
        np.testing.assert_allclose(H_inv, layer.kernel(), atol=1e-6)


def test_trainable():
    q, N = 10, 20
    trafo = TemporalBasisTrafo((q, N), trainable=False)
    assert trafo.trainable is False

    trafo = TemporalBasisTrafo((q, N), trainable=True)
    assert trafo.trainable is True

