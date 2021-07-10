#    Code for the "Nonlinear Synaptic Interaction" Paper
#    Copyright (C) 2017-2020   Andreas Stöckel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import scipy.signal

def mk_2d_flt(sigma, res, cutoff=1e-3):
    """
    Computes a dynamically sized Gaussian filter for use with the gen_2d_fun
    function. Note that the returned filter is 1D only (since it is symmetric).
    """
    x_max = np.sqrt(-np.log(cutoff)) * sigma
    res_ext = int(x_max * res)

    # Make sure all resolutions are odd
    res = (2 * (res // 2)) + 1
    res_ext = (2 * (res_ext // 2)) + 1

    # Special case: if the frequency is too small, just return a single point
    # filter
    if res_ext == 0:
        xs_flt = np.zeros(res)
        xs_flt[res // 2] = 1.0
        return xs_flt

    # Extend x_max if the extended resolution is smaller than the desired
    # resolution
    if res_ext < res:
        x_max = x_max * res / res_ext
        res_ext = res

    # Compute the actual filter
    xs_flt = np.linspace(-x_max, x_max, res_ext)
    flt = np.exp(-(xs_flt ** 2) / (sigma ** 2))
    flt = flt / np.sum(flt)

    return flt

def mk_cropped_2d_flt(sigma, res):
    """
    Crops the filter returned by mk_2d_flt to the given resolution. This is
    for visulaisation purposes.
    """

    flt = mk_2d_flt(sigma, res)
    r0 = int(flt.size / 2 - res / 2)
    r1 = int(flt.size / 2 + res / 2)
    flt = flt[None, :] * flt[:, None]
    return flt[r0:r1, r0:r1]


def norm_2d_fun(zs):
    """
    Normalises a function to have an RMS of one.
    """
    zs -= np.mean(zs)
    zs /= np.sqrt(np.mean(zs ** 2))
    return zs

def _spiral(nrow, ncol):
    """
    Used internally to generate a set of spiral indices, i.e., calling
    _spiral(3, 3) returns:

        4 5 2
        5 0 1
        6 7 8

    This code is adapted from https://stackoverflow.com/a/36961324
    """
    def spiral_ccw(A):
        res = []
        while A.size > 0:
            res.append(A[0][::-1])
            A = A[1:][::-1].T
        return np.concatenate(res)

    def base_spiral(nrow, ncol):
        return spiral_ccw(np.arange(nrow * ncol).reshape(nrow, ncol))[::-1]

    A = np.arange(0, nrow * ncol).reshape(nrow, ncol)
    B = np.zeros_like(A)
    B.flat[base_spiral(*A.shape)] = A.flat
    return B

def gen_2d_fun(flt, res, rng=None, f=None, norm=True):
    """
    Generates a random 2D function with the given filter returned by mk_2d_flt.
    """

    if rng is None:
        rng = np.random

    # Make sure the resolution is odd
    res = (2 * (res // 2)) + 1

    # Determine the extended range over which we have to filter
    res_ext = max(res, flt.size)
    res_ext = res_ext + flt.size
    res_ext = (2 * (res_ext // 2)) + 1

    # Interval to cut out
    r0 = int(res_ext / 2 - res / 2)
    r1 = int(res_ext / 2 + res / 2)

    if f is None:
        # Reproducibly generate some random noise
        zs_flat = rng.normal(0, 1, res_ext * res_ext)

        # Reorder the noise such that the first indices are in the center.
        # This ensures that the pixels in the centre are the same for the
        # same random seed, even if res_ext differs.
        zs = zs_flat[_spiral(res_ext, res_ext)]
    else:
        x_ext = 1 + (res_ext / res - 1) * 0.5
        xs = np.linspace(-x_ext, x_ext, res_ext)
        xss, yss = np.meshgrid(xs, xs)
        zs = f(xss, yss)

    # Filter the input function
    zs_flt_horz = np.zeros((res_ext, res_ext))
    zs_flt = np.zeros((res, res))
    for i in range(res_ext):
        #zs_flt_horz[i, :] = np.convolve(zs[i, :], flt, 'same')
        zs_flt_horz[i, :] = scipy.signal.convolve(zs[i, :], flt, 'same', method='fft')
    for i in range(r0, r1):
        #zs_flt[:, i - r0] = np.convolve(zs_flt_horz[:, i], flt, 'same')[r0:r1]
        zs_flt[:, i - r0] = scipy.signal.convolve(zs_flt_horz[:, i], flt, 'same', method='fft')[r0:r1]

    return norm_2d_fun(zs_flt) if norm else zs_flt

