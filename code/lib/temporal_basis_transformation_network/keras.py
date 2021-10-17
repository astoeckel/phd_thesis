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
import tensorflow as tf

from .common import *


class TemporalBasisTrafo(tf.keras.layers.Layer):
    """
    DESCRIPTION
    ===========

    The TemporalBasisTrafo is a TensorFlow Keras layer that implements a fixed
    convolution with a function basis described by a matrix H of size q x N.
    This matrix can be interpreted as a set of q FIR filters of length N. These
    filters can form a discrete temporal basis. Weighting the momentary
    filtered values thus can be used to compute functions over time in a purely
    feed-forward fashion.

    This code is a generalised version of the feed-forward Legendre Memory Unit
    (ff-LMU) model proposed by Narsimha R. Chilkuri in 2020/21. Instead of
    computing the Legendre Delay Network (LND) basis H, this network will work
    witha any temporal basis matrix H. You can for example use the
    "dlop_ldn_function_bases" package [1] to generate basis transformations.

    The network can furthermore be operated in an inverse mode. In this mode,
    if the output of another TemporalBasisTrafo network is provided, the
    network will reconstruct the original input using the Moore-Penrose
    pseudo-inverse of H. The pseudo inverse is equal to the transpose if H is
    orthogonal.

    For example, if H is the DFT matrix then a network with
    inverse = False will compute the forward discrete Fourier transformation,
    whereas a network with inverse = True will compute the inverse discrete
    Fourier transformation.

    Per default this Layer does not have any trainable parameters. If
    "trainable" is set to true, then the layer will act as a temporal
    convolution network and the given H is the initial temporal convolution.

    [1] https://github.com/astoeckel/dlop_ldn_function_bases

    INPUT AND OUTPUT DIMENSIONS
    ===========================

    The input and output dimensionalities depend on whether the network is
    operating in forward or inverse mode.

    Forward mode
    ------------

    In forward mode (default), this layer takes an input with dimensions

        [n_1, ..., n_i, M, units]         (Input; Mode: Forward)

    The input dimensions n_1, ..., n_i are optional and are interpreted as
    independent batches. M is the number of input samples.

    Depending on the value of "collapse", the output is either an array
    of shape (collapse = True, default)

        [n_1, ..., n_i, M', units * q]    (Output; Mode: Forward, Collapse)

    or (collapse = False)

        [n_1, ..., n_i, M', units, q]     (Output; Mode: Forward)

    where M' is the number of output samples. If padding is enabled (default)
    then M' = M. Otherwise M' = max(1, M - N + 1).

    Inverse mode
    ------------

    In inverse mode, if pre_collapse = True (default), this layer takes an input
    with dimensions

        [n_1, ..., n_i, M, units * q]     (Input; Mode: Inverse, Pre-Collapse)

    Instead, if pre_collapse = False

        [n_1, ..., n_i, M, units, q]      (Input; Mode: Inverse)

    where M is the number of input samples.

    If post_collapse = True (default), the output dimensions are

        [n_1, ..., n_i, M, units * N]     (Output; Mode: Inverse, Post-Collapse)

    or, if post_collapse = False is not set,

        [n_1, ..., n_i, M, units,  N]     (Output; Mode: Inverse)

    """
    def __init__(
        self,
        H,
        units=1,
        pad=False,
        collapse=True,
        mode=Forward,
        normalize=True,
        rcond=1e-6,
        trainable=False,
        initializer=None,
    ):
        """
        Creates a new instance of the BasisTransformationNetwork using the
        basis transformation in the matrix H.

        Parameters
        ==========

        H:  Basis transformation matrix to use. This must be a 2D array of
            size q x N, where q is the number of dimensions an input
            series is being transformed into, and N is the length of one
            input sequence, or, in other words, the number of timesteps.

            H may also be a tuple (q, N). In this case a random TensorFlow
            initializer will be used. The initializer can be selected using
            the `initializer` parameter; it will default to `he_uniform`.

        units: Number of times the basis transformation unit should be
            repeated.

        pad: In Forward mode, if True, pads the input with zeros such that the
            output size and nput size are exactly the same. That is, if
            pad = True and M samples are fed into the network, M samples
            will be returned.

            Otherwise, if pad = False (default), if M samples are
            fed into the network then max(1, M - N + 1) samples will be
            returned, where N is the length of the basis transformation
            matrix. Appropriate padding will always be added if
            M - N + 1 < 0.

            This has no effect in Inverse mode, where the number of input and
            output samples will always be the same.

        collapse: Either a single boolean value, or a tuple (pre_collapse,
            post_collapse). A single boolean value b is translated to (b, b).
            The default value is True, applying both a pre- and post-collapse.

            pre_collapse is only relevant when operating in inverse mode, if set
            to true, reverts a post_collapse applied by a previous network.

        mode: Determines the mode the network operates in. Must be set to
            either the "Forward" or "Inverse" constants exported in the
            temporal_basis_transformation_network package.

        rcond: Regularisation constant used when computing the pseudo-inverse of
            H for the inverse mode.

        trainable: If True, learns the convolution. The convolution is shared
            between all units in the layer.

        normalize: If True (default), the rows of the basis transformation
            matrix H are normalized to length one (in the L2 norm).
            Normalization if the basis transformation matrix will also be
            maintained if `trainable` is true

        initializer: If `trainable` is True and only a shape tuple has been
            given for H, this initializer is used. The default is
            "he_uniform".
        """
        # Call the inherited constructor
        super().__init__(trainable=trainable)

        # Special handling if `H` is a tuple
        if (not initializer is None) and (not isinstance(H, tuple)):
            raise RuntimeError("H must be a tuple (q, N) if an "
                               "initializer is given")

        # Fetch the initializer if one is given
        if (isinstance(H, tuple) and (len(H) == 2) and isinstance(H[0], int)
                and isinstance(H[1], int)):
            # Fetch q, N from H
            q, N = H

            # Fetch an initializer
            self._initializer = "he_uniform" if (
                initializer is None) else initializer

            # Create a stub H to make coerce_params below happy
            H = np.eye(max(q, N))[:q, :N]
        else:
            self._initializer = None

        # Make sure the given parameters make sense
        self._q, self._N, self._H, self._units, self._pad, self._collapse, \
        self._mode = \
            coerce_params(H, units, pad, collapse, mode, normalize, rcond, np)
        self._normalize = normalize

        # In inverse mode, when training the basis transformation, and an
        # initial basis was given, the individual columns of the inverted H
        # may not have length one. We will preserve that length while training.
        self._target_norm = None
        if (self.trainable and self._normalize and (self._mode is Inverse)
                and (self._initializer is None)):
            self._target_norm = np.linalg.norm(self._H, axis=0)

        # Initialize the Tensorflow constants
        self._tf_H, self._tf_pad, self._tf_norm = None, None, None

        # Initialize all the shapes and variables that are being populated in
        # the "build" method
        self._input_shape_pre, self._input_perm, self._input_shape_post, \
        self._output_shape_pre, self._output_perm, self._output_shape_post, \
        self._M_in, self._M_out, self._M_pad = [None] * 9

    def get_config(self):
        return {
            "q": self._q,
            "N": self._N,
            "units": self._units,
            "pad": self._pad,
            "pre_collapse": self._collapse[0],
            "post_collapse": self._collapse[1],
            "mode": repr(self._mode),
            "trainable": self.trainable,
            "normalize": self._normalize,
        }

    def build(self, S):
        """
        This function is called before the first call to "call". Creates
        TensorFlow constants and computes the input and output
        shapes/permutations.
        """

        # Some useful variables
        q, N, forward = self._q, self._N, self._mode is Forward

        # Compute the input/output permutations
        self._input_shape_pre, self._input_perm, self._input_shape_post, \
        self._output_shape_pre, self._output_perm, self._output_shape_post, \
        self._M_in, self._M_out, self._M_pad = \
            compute_shapes_and_permutations(
                S, self._units, q, N, self._pad, self._collapse, self._mode)

        # Upload the basis transformation H into a tensorflow variable.
        conv = (self._M_out > 1) or (self._M_pad > 0)
        shape = (N, q) if forward else (q, N)
        if self.trainable or (not self._initializer is None):
            # Either use H as a constant initializer or use the initializer
            # that was given in the constructor
            if self._initializer is None:
                initializer = tf.constant_initializer(self._H.T)
            else:
                initializer = self._initializer

            # Add the convolution kernel as weights
            self._tf_H = self.add_weight("kernel",
                                         trainable=self.trainable,
                                         initializer=initializer,
                                         shape=shape,
                                         dtype='float32')
        else:
            self._tf_H = tf.constant(value=self._H.T,
                                     shape=shape,
                                     dtype='float32')

        # Padding used to add N - 1 zeros to the beginning of the input array.
        # This way the convolution operation will return exactly N output
        # samples.
        self._tf_pad = None
        if self._M_pad > 0:
            self._tf_pad = tf.constant([[0, 0], [self._M_pad, 0], [0, 0]],
                                       dtype='int32')

        # Load the target_norm vector into a TensorFlow variable.
        if not self._target_norm is None:
            self._tf_norm = tf.constant(self._target_norm, dtype='float32')

    def call(self, xs):
        """
        Implements the actual basis transformations. Reshapes the inputs,
        computes a convolution, and reshapes the output.
        """
        # Reshape and permute the input for convenient convolution
        if not self._input_shape_pre is None:
            xs = tf.reshape(xs, self._input_shape_pre)
        if not self._input_perm is None:
            xs = tf.transpose(xs, perm=self._input_perm)
        if not self._input_shape_post is None:
            xs = tf.reshape(xs, self._input_shape_post)

        # Apply padding to the intput
        if not self._tf_pad is None:
            xs = tf.pad(xs, self._tf_pad)

        # Fetch the convolution kernel. This applies normalisation if so
        # desired.
        H = self.kernel(numpy=False, transpose=False)

        # Apply the convolution
        if (self._mode is Forward) and ((self._M_out > 1) or
                                        (self._M_pad > 0)):
            ys = tf.nn.convolution(xs, tf.reshape(H, (self._N, 1, self._q)))
        else:
            ys = tf.matmul(xs, H)

        # Reshape and permute the output to the original shape
        if not self._output_shape_pre is None:
            ys = tf.reshape(ys, self._output_shape_pre)
        if not self._output_perm is None:
            ys = tf.transpose(ys, perm=self._output_perm)
        if not self._output_shape_post is None:
            ys = tf.reshape(ys, self._output_shape_post)
        return ys

    def kernel(self, numpy=True, transpose=True):
        # H has not been converted into a TensorFlow variable yet. Note that
        # the TensorFlow variable is a transpsed version of H, so the transpose
        # operation below seems inverted, but this is correct. No normalisation
        # has to be done, since _H is already transposed.
        if self._tf_H is None:
            if not self._initializer is None:
                raise RuntimeError("Kernel matrix not yet initialized")
            return self._H if transpose else self._H.T

        # Normalize the kernel. We don't have to do this if the layer is not
        # trainable, since normalisation happened at construction time.
        H = self._tf_H
        if self._normalize and self.trainable:
            if self._mode is Forward:
                norm = tf.norm(self._tf_H, axis=0)
            else:  # self._mode is Inverse:
                norm = tf.norm(self._tf_H, axis=1)
            if not self._tf_norm is None:
                norm = norm * self._tf_norm
            if self._mode is Forward:
                H = H / tf.reshape(norm, (1, self._q))
            else:  # self._mode is Inverse:
                H = H / tf.reshape(norm, (self._q, 1))

        # These options are mostly used internally to access the normalized
        # vector.
        if transpose:
            H = tf.transpose(H)
        if numpy:
            H = H.numpy()

        return H

