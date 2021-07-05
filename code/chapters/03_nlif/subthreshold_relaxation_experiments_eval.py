#!/usr/bin/env python3

import itertools
import multiprocessing
import sys, os

import numpy as np

import random

import lif_utils
import env_guard

import bioneuronqp

import tqdm
import h5py

# Maximum firing rate (hard-coded in nonneg_common.mk_ensemble)
A_MAX = 100

# Number of noise parameters to use
N_SIGMAS = 21
SIGMAS = np.logspace(-3, 0, N_SIGMAS)

# Number of repetitions for each noise parameter
N_REPEAT = 11

F_DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'data',
                      'subthreshold_relaxation_experiment.h5')
F_DATA_EVAL = os.path.join(os.path.dirname(__file__), '..', '..', 'data',
                           'subthreshold_relaxation_experiment_eval.h5')


def evaluate(params):
    idx_fun, idx_reg, idx_th, idx_ratio, idx_net = params

    with h5py.File(F_DATA, 'r') as f:
        # Fetch the source datasets
        DS_D = f["D"]
        DS_AS_TEST_PRE = f["as_test_pre"]
        DS_JS_TEST_POST = f["js_test_post"]
        DS_YS = f["ys"]
        DS_WS = f["ws"]

        # Fetch the correct entries from the source dataset
        As_pre = DS_AS_TEST_PRE[idx_net][()]
        W = DS_WS[idx_fun, idx_reg, idx_th, idx_ratio, idx_net][()]
        ys_ref = DS_YS[:, idx_fun][()]
        ys_rms = np.sqrt(np.mean(np.square(ys_ref)))
        Js_post_tar = DS_JS_TEST_POST[idx_net, idx_fun]
        Js_rms = np.sqrt(np.mean(np.square(Js_post_tar)))
        D = DS_D[idx_net][()]

        # Compute the decoding without noise
        def compute_error(sigma=0.0, idx=0):
            # Add some noise to the pre-activities
            rng = np.random.RandomState(49957 * idx + 5813)
            if sigma == 0.0:
                As_pre_noise = As_pre
            else:
                As_pre_noise = (As_pre + rng.normal(0, sigma, As_pre.shape))

            # Compute the post-activities
            Js_post = As_pre_noise @ W
            As_post = lif_utils.lif_rate(Js_post)

            # Decode the function
            ys_dec = As_post @ D

            # Compute the decoding error
            E_dec = np.sqrt(np.mean(np.square(ys_dec - ys_ref)))  / ys_rms

            # Compute the current error
            iTh = 1.0
            is_sup = Js_post_tar > iTh
            is_invalid = np.logical_and(Js_post > iTh, ~is_sup)
            E_cur = np.sqrt(
                np.mean(
                    np.square(is_sup * (Js_post - Js_post_tar) + is_invalid *
                              (Js_post - iTh)))) / Js_rms

            return E_cur, E_dec

        # Compute the "zero" error
        E0_cur, E0 = compute_error()

        # Fill the error matrix by computing the error gain
        errs_cur = np.zeros((N_REPEAT, N_SIGMAS))
        errs = np.zeros((N_REPEAT, N_SIGMAS))
        gains = np.zeros((N_REPEAT, N_SIGMAS))
        for n in range(N_REPEAT):
            for o in range(N_SIGMAS):
                sigma = SIGMAS[o] * A_MAX
                errs_cur[n, o], errs[n, o] = compute_error(sigma, n)
                gains[n, o] = errs[n, o] / sigma

        err_reg = np.sqrt(np.mean(np.square(W)))

    return idx_fun, idx_reg, idx_th, idx_ratio, idx_net, E0_cur, E0, errs_cur, errs, err_reg, gains


def main():
    # Open the experiment data and write the evaluation data
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    with h5py.File(F_DATA, 'r') as f, h5py.File(F_DATA_EVAL, 'w') as f_eval:
        # Create the target datasets
        params_shape = f["ws"].shape[:-2]

        # Store the sigmas
        f_eval.create_dataset("sigmas", data=SIGMAS)

        # Copy the other parameters
        f_eval.create_dataset('regs', data=f["regs"])
        f_eval.create_dataset('ths', data=f["ths"])
        f_eval.create_dataset('ratios', data=f["ratios"])

        # Compute the errors and gains
        DS_ERRS_CUR = f_eval.create_dataset(
            "errs_cur", (*params_shape, N_REPEAT, N_SIGMAS))
        DS_ERRS = f_eval.create_dataset("errs",
                                        (*params_shape, N_REPEAT, N_SIGMAS))
        DS_GAINS = f_eval.create_dataset("gains",
                                         (*params_shape, N_REPEAT, N_SIGMAS))
        DS_ERR_REGS = f_eval.create_dataset("err_regs", (*params_shape, ))
        DS_E0S_CUR = f_eval.create_dataset("e0s_cur", (*params_shape, ))
        DS_E0S = f_eval.create_dataset("e0s", (*params_shape, ))

        params = list(itertools.product(*[range(N) for N in params_shape]))
        with env_guard.SingleThreadEnvGuard():
            with multiprocessing.get_context('spawn').Pool() as pool:
                for i, j, k, l, m, e0_cur, e0, errs_cur, errs, err_reg, gains in tqdm.tqdm(
                        pool.imap_unordered(evaluate,
                                            params), total=len(params)):
                    DS_E0S_CUR[i, j, k, l, m] = e0
                    DS_E0S[i, j, k, l, m] = e0
                    DS_ERR_REGS[i, j, k, l, m] = err_reg
                    DS_ERRS_CUR[i, j, k, l, m] = errs_cur
                    DS_ERRS[i, j, k, l, m] = errs
                    DS_GAINS[i, j, k, l, m] = gains


if __name__ == "__main__":
    main()

