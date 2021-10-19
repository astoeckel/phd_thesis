import os
import lmu_utils


def dataset_file(fn):
    return os.path.join(os.path.dirname(__file__), '../../../../data/datasets',
                        fn)


def double_rows(A):
    res = np.zeros((A.shape[0] * 2, *A.shape[1:]))
    res[0::2] = A
    res[1::2] = A
    return res


def double(A):
    return double_rows(double_rows(A).T).T


def quadruple(A):
    return double(double(A))


imgs = lmu_utils.read_idxgz(dataset_file('train-images-idx3-ubyte.gz'))

fig = plt.figure(figsize=(7.5, 3.0))

ax1 = fig.add_axes([0.1, 0.7, 0.1, 0.2])
ax1.imshow(quadruple(imgs[0]), cmap='Greys', interpolation='none')
ax1.set_xticks([])
ax1.set_yticks([])
ax1.spines["right"].set_visible(True)
ax1.spines["top"].set_visible(True)

S = imgs.shape[1:]
N = np.prod(S)
np.random.seed(8789)
pi = np.random.choice(np.arange(0, N, dtype=int), N, replace=False)

ax2 = fig.add_axes([0.1, 0.1, 0.1, 0.2])
ax2.imshow(quadruple(imgs[0].flatten()[pi].reshape(*S)),
           cmap='Greys',
           interpolation='none')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.spines["right"].set_visible(True)
ax2.spines["top"].set_visible(True)

ax3 = fig.add_axes([0.25, 0.475, 0.2, 0.05])
ax3.imshow(quadruple(imgs[0].flatten()[pi][:20].reshape(1, 20)),
           extent=[0.5, 19.5, 0, 1],
           cmap='Greys',
           interpolation='none')
ax3.set_yticks([])
ax3.spines["left"].set_visible(False)
ax3.set_xlabel("Time $t$")

utils.save(fig)

