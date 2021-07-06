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

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'libbioneuronqp'))
import bioneuronqp

from nef_synaptic_computation.lif_utils import *
from nef_synaptic_computation.tuning_curves import *

from gen_2d_fun import *
from two_comp_parameters import *

###############################################################################
# Helper functions                                                            #
###############################################################################


def solve(f,
          A_xs,
          A_ys,
          ws,
          reg=None,
          iTh=None,
          nonneg=True,
          max_iter=MAX_ITER,
          tol=TOL,
          **kwargs):

    # Convert the arguments to the format expected by bioneuronqp.solve
    Apre = np.concatenate((A_xs, A_ys)).T
    Jpost = f.reshape(-1, 1)

    # Run the solver
    WE, WI = bioneuronqp.solve(Apre,
                               Jpost,
                               ws,
                               iTh=iTh,
                               nonneg=nonneg,
                               reg=reg,
                               tol=tol,
                               max_iter=max_iter,
                               progress_callback=None,
                               renormalise=True,
                               **kwargs)

    return np.concatenate((WE, WI)).reshape(4, -1)


def eval(A_xs, A_ys, ws, fws):
    a0, a1, a2, b0, b1, b2 = ws
    wf1, wg1, wf2, wg2 = fws
    f1, f2 = (wf1, wf2) @ A_xs
    g1, g2 = (wg1, wg2) @ A_ys
    f_eval = ((a0 + a1 * (f1 + g1) + a2 * (f2 + g2)) /
              (b0 + b1 * (f1 + g1) + b2 * (f2 + g2)))
    return f_eval


def RMSE(f, f_tar):
    f, f_tar = np.copy(f), np.copy(f_tar)
    f[np.logical_and(f_tar < 0, f < 0)] = 0
    f_tar[f_tar < 0] = 0
    E = f - f_tar
    return np.sqrt(np.mean(np.square(E))) / J_SCALE


def generate_tuning_curves(res,
                           N=N_NEURONS,
                           dir="x",
                           dim=1,
                           rng=None,
                           max_rates=MAX_RATES):
    # If dir is given, dim must be one
    assert (dir is None) != (dim == 1)

    if rng is None:
        rng = np.random

    # Generate the gain an bias parameters
    x_intercepts = rng.uniform(-0.95, 0.95, N) * ([-1, 1] * (N // 2))
    max_rates = rng.uniform(*max_rates)
    gain, bias = first_order_tuning_curve_parameters(max_rates, x_intercepts)

    # Sample the 2D space regularly
    xs = np.linspace(-1, 1, res)
    xss, yss = np.meshgrid(xs, xs)

    if dim == 1:
        # Special case: 50 positive and 50 negative encoders
        encoders = [1] * (N // 2) + [-1] * (N // 2)

        # Compute the activities
        A = lif_rate((xss if (dir == "x") else yss).flatten()[:, None] *
                     (encoders * gain)[None, :] + bias)
    else:
        # Compute random encoders on the surface of the unit hypersphere
        encoders = rng.normal(0, 1, (N, dim))
        encoders /= np.linalg.norm(encoders, axis=1)[:, None]

        # Generate the 2D input samples
        samples = np.array((xss.flatten(), yss.flatten())).T

        # Compute the output activities
        A = lif_rate((samples @ encoders.T) * gain + bias)

    return A


###############################################################################
# Actual experiment function                                                  #
###############################################################################


def approximate_function(ws,
                         sigma,
                         reg,
                         pre_config="split",
                         rng=None,
                         res=RES,
                         iTh=ITH,
                         noise_a_pre=NOISE_A_PRE,
                         n_smpls=N_SAMPLES,
                         n_noise_trials=N_NOISE_TRIALS,
                         n_neurons=N_NEURONS,
                         max_rates=MAX_RATES):

    # Make sure "mode" is one of "split" or "combined"
    assert pre_config in {"split", "combined"}

    # Fetch the default random number generator
    if rng is None:
        rng = np.random

    # Generate a few new random number generators based on this one
    rng1 = np.random.RandomState(rng.randint(1 << 31))
    rng2 = np.random.RandomState(rng.randint(1 << 31))
    rng3 = np.random.RandomState(rng.randint(1 << 31))

    # Compute the target function
    if callable(sigma):
        xs = np.linspace(-1, 1, res)
        xss, yss = np.meshgrid(xs, xs)
        f_tar = norm_2d_fun(sigma(xss, yss)).flatten() * J_SCALE
    else:
        flt = mk_2d_flt(sigma, res)
        f_tar = gen_2d_fun(flt, res, rng1).flatten() * J_SCALE

    # Determine the training set
    idx_train = rng.randint(0, f_tar.size, n_smpls)

    # Generate the pre-neuron tuning curves
    if pre_config == "combined":
        A2d = generate_tuning_curves(res,
                                     n_neurons * 2,
                                     None,
                                     dim=2,
                                     rng=rng2,
                                     max_rates=max_rates)
        Apre1, Apre2 = A2d, A2d
    else:
        Apre1 = generate_tuning_curves(res,
                                       n_neurons,
                                       "x",
                                       rng=rng2,
                                       max_rates=max_rates)
        Apre2 = generate_tuning_curves(res,
                                       n_neurons,
                                       "y",
                                       rng=rng2,
                                       max_rates=max_rates)

    f_tar_train = f_tar[idx_train]
    Apre1_train = Apre1[idx_train].T
    Apre2_train = Apre2[idx_train].T

    # Solve for weights WE, WI
    fws = solve(f_tar_train, Apre1_train, Apre2_train, ws, reg=reg, iTh=iTh)
    fws_no_mask_negative = solve(f_tar_train,
                                 Apre1_train,
                                 Apre2_train,
                                 ws,
                                 reg=reg)

    # Calculate and return the error
    errs, errs_no_mask_negative = np.zeros((2, n_noise_trials))
    for i_noise in range(n_noise_trials):

        def mknoise(A):
            return rng3.normal(0, noise_a_pre * max_rates[1], A.shape)

        # Generate some Gaussian noise
        Anoise1 = mknoise(Apre1)
        Anoise2 = Anoise1 if pre_config == "combined" else mknoise(Apre2)

        # Add the noise, make sure the pre-activities are not smaller than zero
        Apre1_noise = np.clip(Apre1 + Anoise1, 0, None).T
        Apre2_noise = np.clip(Apre2 + Anoise2, 0, None).T

        f = eval(Apre1_noise, Apre2_noise, ws, fws)
        f_no_mask_negative = eval(Apre1_noise, Apre2_noise, ws,
                                  fws_no_mask_negative)

        errs[i_noise] = RMSE(f, f_tar)
        errs_no_mask_negative[i_noise] = RMSE(f_no_mask_negative, f_tar)

    # Compute the median error under noise
    return {
        "err": np.median(errs),
        "err_no_mask_negative": np.median(errs_no_mask_negative),
        "f_tar": f_tar,
        "idx_train": idx_train,
        "Apre1": Apre1,
        "Apre2": Apre2,
        "fws": fws,
        "fws_no_mask_negative": fws_no_mask_negative,
        "ws": ws,
    }


def run_single_experiment_common(sigma, reg, p_key, i_repeat):
    # This function is shared between the "run_2d_frequency_sweep" and the
    # "run_2d_regularisation_sweep" scripts

    # Determine the "pre_config"
    pre_config = "combined" if p_key.endswith("_2d") else "split"

    # Setup the random number generator
    rng = np.random.RandomState(4917 * i_repeat + 373)

    # Run the actual experiment
    res = approximate_function(ws=mkws(p_key),
                               sigma=sigma,
                               reg=reg,
                               pre_config=pre_config,
                               rng=rng)
    err = res["err"]
    err_no_mask_negative = res["err_no_mask_negative"]

    # Return the indices and the computed errors
    return err, err_no_mask_negative

