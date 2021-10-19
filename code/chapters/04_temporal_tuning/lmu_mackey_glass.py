#!/usr/bin/env python3

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))

import numpy as np
import tempfile
import itertools
import multiprocessing
import tqdm
import random
import datetime

import dlop_ldn_function_bases as bases

import lmu_utils
from lmu_utils import mk_mackey_glass_dataset

BASES = [
    (lmu_utils.mk_ext_ldn_basis, "ldn"),  # 0
    (lmu_utils.mk_mod_fourier_basis, "mod_fourier"),  # 1
    (lmu_utils.mk_ext_dlop_basis, "dlop"),  # 2
    (lmu_utils.mk_ext_fourier_basis, "fourier"),  # 3
    (lmu_utils.mk_ext_cosine_basis, "cosine"),  # 4
    (lmu_utils.mk_ext_haar_basis, "haar"),  # 5
    (lmu_utils.mk_ext_eye_basis, "eye"),  # 6
    (None, "random"),  # 7
]

N_EPOCHS = 200


def run_single_experiment(params, verbose=False, return_model=False):
    import tensorflow as tf
    from temporal_basis_transformation_network.keras import TemporalBasisTrafo

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    # Destructure the parameters
    idcs, basis_idx, ext_flt, train, seed = params
    basis_ctor, basis_name = BASES[basis_idx]

    # Special handling if a random convolution should be used
    if basis_ctor is None:
        # Generate a random initial temporal convolution
        rng = np.random.RandomState(481 + 341 * seed)

        def basis_ctor(q, N, Nmul=1):
            return (int(q), int(N * Nmul))

    # Set the TF seed
    tf.random.set_seed(seed=131 + 513 * seed)

    # Generate the dataset
    Nm = 3 if ext_flt else 1
    N_wnd0, N_wnd1, N_wnd2, N_wnd3 = N_wnds = (17 * Nm, 9 * Nm, 9 * Nm, 5 * Nm)
    ds_train, ds_val, ds_test = mk_mackey_glass_dataset(N_wnds=N_wnds, seed=seed, verbose=verbose)
    N_wnd = ds_train.element_spec[0].shape[1]
    N_pred = ds_train.element_spec[1].shape[1]
    rms = 1.0

    # Run the experiment
    with tempfile.NamedTemporaryFile() as f:
        N_units0 = 1
        N_units1 = 10
        N_units2 = 10
        N_units3 = 10
        q0, q1, q2, q3 = N_wnd0 // Nm, N_wnd1 // Nm, N_wnd2 // Nm, N_wnd3 // Nm
        H0 = basis_ctor(q0, q0, Nm)
        H1 = basis_ctor(q1, q1, Nm)
        H2 = basis_ctor(q2, q2, Nm)
        H3 = basis_ctor(q3, q3, Nm)
        model = tf.keras.models.Sequential([
          tf.keras.layers.Reshape((N_wnd, 1)),                       # (N_wnd0 + N_wnd1 + N_wnd2 + N_wnd3, 1)
          TemporalBasisTrafo(H0, units=N_units0, trainable=train),   # (N_wnd1 + N_wnd2 + N_wnd3, q * N_units0)

          tf.keras.layers.Dense(N_units1, activation='relu'),        # (N_wnd1 + N_wnd2 + N_wnd3, N_units1)
          TemporalBasisTrafo(H1, units=N_units1, trainable=train),   # (N_wnd2 + N_wnd3, q * N_units1)

          tf.keras.layers.Dense(N_units2, activation='relu'),        # (N_wnd2 + N_wnd3, N_units2)
          TemporalBasisTrafo(H2, units=N_units2, trainable=train),   # (N_wnd3, q * N_units2)

          tf.keras.layers.Dense(N_units3, activation='relu'),        # (N_wnd3, N_units3)
          TemporalBasisTrafo(H3, units=N_units3, trainable=train),   # (1, q * N_units3)

          tf.keras.layers.Dense(N_pred, use_bias=False),             # (1, N_pred)
          tf.keras.layers.Reshape((N_pred,))                         # (N_pred)
        ])

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f.name,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        traj = np.zeros((N_EPOCHS, 2))

        class RecordingCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                traj[epoch][0] = logs["loss"]
                traj[epoch][1] = logs["val_loss"]

        model_recording_callback = RecordingCallback()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.mse,
        )

        model.fit(
            ds_train,
            epochs=N_EPOCHS,
            validation_data=ds_val,
            callbacks=[model_checkpoint_callback, model_recording_callback],
            verbose=verbose,
        )

        model.load_weights(f.name)

        nrmse = np.sqrt(model.evaluate(ds_test, verbose=verbose)) / rms
        if return_model:
            return idcs, nrmse, traj, model
        return idcs, nrmse, traj


if __name__ == '__main__':
    if len(sys.argv) > 1:
        n_partitions = int(sys.argv[1])
        partition_idx = int(sys.argv[2])
    else:
        n_partitions = 1
        partition_idx = 0

    assert n_partitions > 0
    assert partition_idx < n_partitions
    assert partition_idx >= 0

    if n_partitions == 1:
        fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
                          "lmu_mackey_glass.npz")
    else:
        fn = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'data',
            "lmu_mackey_glass_{}.npz".format(partition_idx))

    multiprocessing.freeze_support()

    basis_idcs = range(len(BASES))
    seeds = range(101)
    params = list([((i, j, k, l), basis_idx, ext_flt, train, seed)
                   for i, basis_idx in enumerate(basis_idcs)
                   for j, ext_flt in enumerate([False, True])
                   for k, train in enumerate([False, True])
                   for l, seed in enumerate(seeds)])

    random.seed(587232)
    random.shuffle(params)

    partitions = np.linspace(0, len(params), n_partitions + 1, dtype=int)
    i0 = partitions[partition_idx]
    i1 = partitions[partition_idx + 1]
    print(
        f"Partition {partition_idx} out of {n_partitions} (i0={i0}, i1={i1}); total={len(params)}"
    )

    errs = np.zeros((len(basis_idcs), 2, 2, len(seeds))) * np.nan
    trajs = np.zeros((len(basis_idcs), 2, 2, len(seeds), N_EPOCHS, 2)) * np.nan
    with multiprocessing.get_context('spawn').Pool() as pool:
        for (i, j, k, l), E, traj in tqdm.tqdm(pool.imap_unordered(
                run_single_experiment, params[i0:i1]),
                                               total=i1 - i0):
            errs[i, j, k, l] = E
            trajs[i, j, k, l] = traj

    np.savez(
        fn, **{
            "errs": errs,
            "trajs": trajs,
            "basis_names": [x[1] for x in BASES],
        })

