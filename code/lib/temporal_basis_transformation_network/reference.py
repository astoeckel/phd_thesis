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

import numpy as np

from .common import *


def trafo(xs,
          H,
          units=1,
          pad=True,
          collapse=True,
          mode=Forward,
          normalize=True,
          rcond=1e-6,
          np=np):
    """
    Reference implementation of the temporal basis transformation network. This
    implementation does not use TensorFlow, but only numpy. The backing numpy
    implementation can be changed by switching from 
    """
    # Coerce the parameters
    q, N, H, units, pad, collapse, mode = coerce_params(
        H, units, pad, collapse, mode, normalize, rcond, np)

    # Make sure the input data is a float32 array
    xs = np.asarray(xs, dtype=np.float32)

    # Compute the re-shapes that should be applied to the input and output
    input_shape_pre, input_perm, input_shape_post, \
    output_shape_pre, output_perm, output_shape_post, \
    M_in, M_out, M_pad = \
        compute_shapes_and_permutations(xs.shape, units, q, N, pad, collapse, mode)

    # Re-arrange the input signal
    if not input_shape_pre is None:
        xs = xs.reshape(input_shape_pre)
    if not input_perm is None:
        xs = xs.transpose(input_perm)
    if not input_shape_post is None:
        xs = xs.reshape(input_shape_post)

    # Pad the input signal if there is padding to be done
    if (M_pad > 0):
        assert xs.ndim == 3
        s0, _, s2 = xs.shape
        xs = np.concatenate((np.zeros((s0, M_pad, s2)), xs), axis=1)

    # Compute the convolution
    if (mode is Forward) and ((M_out > 1) or (M_pad > 0)):
        N_conv = input_shape_post[0]
        ys = np.zeros((N_conv, M_out, q))
        for i in range(N_conv):
            for j in range(q):
                ys[i, :, j] = np.convolve(xs[i, :, 0], H[j, ::-1], 'valid')
    else:
        ys = xs @ H.T

    # Re-arrange the output signal
    if not output_shape_pre is None:
        ys = ys.reshape(output_shape_pre)
    if not output_perm is None:
        ys = ys.transpose(output_perm)
    if not output_shape_post is None:
        ys = ys.reshape(output_shape_post)

    return ys

