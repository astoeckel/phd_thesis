#!/usr/bin/env python3

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))

import numpy as np
import lmu_utils as utils
import tempfile
import itertools
import multiprocessing
import tqdm
import random
import datetime

import dlop_ldn_function_bases as bases


def dataset_file(fn):
    return os.path.join(os.path.dirname(__file__), '../../../data/datasets',
                        fn)


MNIST = {
    "train_imgs": utils.read_idxgz(dataset_file('train-images-idx3-ubyte.gz')),
    "train_lbls": utils.read_idxgz(dataset_file('train-labels-idx1-ubyte.gz')),
    "test_imgs": utils.read_idxgz(dataset_file('t10k-images-idx3-ubyte.gz')),
    "test_lbls": utils.read_idxgz(dataset_file('t10k-labels-idx1-ubyte.gz'))
}


def mk_psmnist_dataset(n_validate=10000, seed=103891, batch_size=100):
    import tensorflow as tf

    # Generate a random number generator for the given seed
    rng = np.random.RandomState(57503 + 15173 * seed)

    mnist_train_orig_imgs, mnist_train_orig_lbls, \
    mnist_test_imgs, mnist_test_lbls = \
        np.copy(MNIST["train_imgs"]), np.copy(MNIST["train_lbls"]), \
        np.copy(MNIST["test_imgs"]), np.copy(MNIST["test_lbls"])

    # Randomly split the validation dataset off the validation data
    idcs = rng.permutation(np.arange(mnist_train_orig_imgs.shape[0]))
    idcs_train = idcs[n_validate:]
    idcs_val = idcs[:n_validate]

    mnist_train_imgs = mnist_train_orig_imgs[idcs_train]
    mnist_train_lbls = mnist_train_orig_lbls[idcs_train]

    mnist_val_imgs = mnist_train_orig_imgs[idcs_val]
    mnist_val_lbls = mnist_train_orig_lbls[idcs_val]

    # Generate a random permutation of the pixels
    perm = rng.permutation(np.arange(28 * 28))

    def permute(imgs):
        res_imgs = np.zeros((imgs.shape[0], 28 * 28), dtype=np.float32)
        for i in range(imgs.shape[0]):
            res_imgs[i] = 2.0 * imgs[i].astype(
                np.float32).flatten()[perm] / 255.0 - 1.0
        return res_imgs

    ds_train = tf.data.Dataset.from_tensor_slices(
        (permute(mnist_train_imgs), mnist_train_lbls))
    ds_train = ds_train.shuffle(len(idcs_train))
    ds_train = ds_train.batch(batch_size)

    ds_val = tf.data.Dataset.from_tensor_slices(
        (permute(mnist_val_imgs), mnist_val_lbls))
    ds_val = ds_val.batch(batch_size)

    ds_test = tf.data.Dataset.from_tensor_slices(
        (permute(mnist_test_imgs), mnist_test_lbls))
    ds_test = ds_test.batch(batch_size)

    return ds_train, ds_val, ds_test


BASES = [
    (bases.mk_ldn_basis, "ldn"),  #0
    (utils.mk_mod_fourier_basis, "mod_fourier"),  #1
    (bases.mk_dlop_basis, "dlop"),  #2
    (bases.mk_fourier_basis, "fourier"),  #3
    (bases.mk_cosine_basis, "cosine"),  #4
    (bases.mk_haar_basis, "haar"),  #5
    (None, "random"),  #6
]

N_EPOCHS = 10


def run_single_experiment(params,
                          verbose=False,
                          return_model=False,
                          use_gpu=False,
                          use_single_cpu=True):
    import tensorflow as tf
    from temporal_basis_transformation_network.keras import TemporalBasisTrafo

    if use_single_cpu:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

    # Disable all devices but a single GPU
    if use_gpu:
        import os
        if "GPU" in os.environ:
            GPU = int(os.environ["GPU"])
            devs = []
            for dev in tf.config.list_physical_devices('GPU'):
                if dev.name.endswith(":{}".format(GPU)):
                    devs.append(dev)
                    break
            tf.config.set_visible_devices(devs, 'GPU')
    else:
        tf.config.set_visible_devices([], 'GPU')

    # Destructure the parameters
    idcs, basis_idx, q, train, seed = params
    basis_ctor, basis_name = BASES[basis_idx]

    # Special handling if a random convolution should be used
    if basis_ctor is None:
        # Generate a random initial temporal convolution
        rng = np.random.RandomState(481 + 341 * seed)

        def basis_ctor(q, N):
            return (int(q), int(N))
    elif basis_ctor is utils.mk_mod_fourier_basis:
        q = 2 * ((q - 1) // 2) + 1  # Use odd q for the modified Fourier basis

    # Set the TF seed
    tf.random.set_seed(seed=131 + 513 * seed)

    # Generate a dataset
    ds_train, ds_val, ds_test = mk_psmnist_dataset(seed=seed)

    # Run the experiment
    with tempfile.NamedTemporaryFile() as f:
        N = 28 * 28
        N_neurons = 346
        N_units = 1
        H = basis_ctor(q, N)
        model = tf.keras.models.Sequential([
            tf.keras.layers.Reshape((N, 1)),  # (N, 1)
            TemporalBasisTrafo(H, units=N_units, trainable=train),  # (1, q)
            tf.keras.layers.Dropout(0.5),  # (1, q)
            tf.keras.layers.Dense(N_neurons,
                                  activation='relu'),  # (1, N_neurons)
            tf.keras.layers.Dense(10, use_bias=False),  # (1, 10)
            tf.keras.layers.Reshape((10, ))  # (10)
        ])

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f.name,
            save_weights_only=True,
            monitor='val_sparse_categorical_accuracy',
            mode='max',
            save_best_only=True)

        traj = np.zeros((N_EPOCHS, 2))

        class RecordingCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                traj[epoch][0] = logs["loss"]
                traj[epoch][1] = logs["val_loss"]

        model_recording_callback = RecordingCallback()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

        model.fit(
            ds_train,
            epochs=N_EPOCHS,
            validation_data=ds_val,
            callbacks=[model_checkpoint_callback, model_recording_callback],
            verbose=verbose,
            use_multiprocessing=False,
        )

        model.load_weights(f.name)

        acc = model.evaluate(ds_test, verbose=verbose)[1]
        if return_model:
            return idcs, acc, traj, model
        return idcs, acc, traj  # Return the test accuracy


if __name__ == '__main__':
    fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
                      "lmu_psmnist.h5")

    multiprocessing.freeze_support()

    import sys
    if len(sys.argv) > 1:
        assert len(sys.argv) == 3
        n_slices = int(sys.argv[1])
        i_slice = int(sys.argv[2])
    else:
        n_slices = 1
        i_slice = 1

    #qs = [468]
    qs = [101]
    basis_idcs = range(len(BASES))
    seeds = range(11)
    params = list([((i, j, k, l), basis_idx, q, train, seed)
                   for i, basis_idx in enumerate(basis_idcs)
                   for j, q in enumerate(qs)
                   for k, train in enumerate([False, True])
                   for l, seed in enumerate(seeds)])
    random.seed(572492)
    random.shuffle(params)

    slice_idcs = np.linspace(0, len(params), n_slices + 1, dtype=np.int)
    i0_slice, i1_slice = slice_idcs[i_slice - 1], slice_idcs[i_slice]
    params = params[i0_slice:i1_slice]
    print("n_slices={}, i_slice={}, i0={}, i1={}".format(
        n_slices, i_slice, i0_slice, i1_slice))

    errs = np.zeros((len(basis_idcs), len(qs), 2, len(seeds)))
    trajs = np.zeros((len(basis_idcs), len(qs), 2, len(seeds), N_EPOCHS, 2))
    with multiprocessing.get_context('spawn').Pool() as pool:
        for (i, j, k, l), E, traj in tqdm.tqdm(pool.imap_unordered(
                run_single_experiment, params),
                                               total=len(params)):
            errs[i, j, k, l] = E
            trajs[i, j, k, l] = traj

    fn = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data',
                      "lmu_psmnist.npz")
    np.savez(
        fn, **{
            "errs": errs,
            "trajs": trajs,
            "basis_names": [x[1] for x in BASES],
            "qs": qs,
        })

