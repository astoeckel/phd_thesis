import numpy as np
import scipy.special

# See Chi, H., Mascagni, M., & Warnock, T. (2005).
# On the optimal Halton sequence. Mathematics and Computers in Simulation,
# 70(1), 9–21. https://doi.org/10.1016/j.matcom.2005.03.004


def halton_seq(dim_idx, smpl_idx):
    P = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67,
        71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139,
        149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
        227, 229
    ]
    W = [
        1, 2, 2, 5, 3, 7, 3, 10, 18, 11, 17, 5, 17, 26, 40, 14, 40, 44, 12, 31,
        45, 70, 8, 38, 82, 8, 12, 38, 47, 70, 29, 57, 97, 110, 32, 48, 84, 124,
        155, 26, 69, 83, 157, 171, 8, 32, 112, 205, 15, 31
    ]

    x, div = 0.0, float(P[dim_idx])
    # Represent the number j in the basis p[i] as j = b0 + b1 * p[i] +
    # b2 * p[i]^2 ...
    j = smpl_idx
    while j > 0:
        # Current digit
        b = j % P[dim_idx]

        # Apply the linear congruent permutation proposed in the Chi et
        # al. paper
        bp = (b * W[dim_idx]) % P[dim_idx]

        # Sum the digit divided by the current divisor
        x += float(bp) / div

        # Go to the next digit, increment the divisor
        j //= P[dim_idx]
        div *= float(P[dim_idx])

    return x


# This code is adapted from nengolib.stats.ntmdists.spherical_transform in
# Aaron Voelker's nengolib, which in turn is based on Section 1.5 and
# Appendix B.2 of
#
# K.-T. Fang and Y. Wang, Number-Theoretic Methods in Statistics.
#     Chapman & Hall, 1994.


def spherical_coordinates_inverse_cdf(m, y):
    # See Appendix B.2 p. 309-310.
    y_reflect = (1.0 - y) if (y > 0.5) else y
    z = np.sqrt(scipy.special.betaincinv(0.5 * m, 0.5, 2.0 * y_reflect))
    x = np.arcsin(z) / np.pi
    return (1.0 - x) if (y > 0.5) else x


def spherical_transform(X):
    # Number of input and output dimensions
    Din = len(X)
    Dout = len(X) + 1

    # Transform the hypercube coordinates into spherical coordinates according
    # to Section 1.5.3 and Appendix B.2.
    phis = np.zeros(Din)
    for i in range(Din):
        phis[i] = spherical_coordinates_inverse_cdf(Din - i, X[i])

    # Convert the spherical coordinates into cartesian coordinates. See eq.
    # (1.5.27) (note the implicit parentheses around ∏ Si)
    C, S = np.zeros(Din), np.zeros(Din)
    for i in range(Din):
        x = np.pi * (2.0 if (i == (Din - 1)) else 1.0) * phis[i]
        C[i] = np.cos(x)
        S[i] = np.sin(x)

    tar = np.zeros(Dout)
    for j in range(Dout):
        tar[j] = 1.0 if j == 0 else (tar[j - 1] * S[j - 1])
    for j in range(Din):
        tar[j] *= C[j]

    return tar


def spherical_halton(n_dims, smpl_idx):
    if n_dims == 1:
        return 1.0 if (int(smpl_idx) % 2 == 0) else -1.0
    X = np.array([halton_seq(i, smpl_idx) for i in range(n_dims - 1)])
    return spherical_transform(X)


def halton_ball(n_dims, smpl_idx):
    r = np.power(halton_seq(0, smpl_idx), 1.0 / n_dims)
    if n_dims == 1:
        return np.array((2.0 * r - 1.0, ))
    X = np.array([halton_seq(i + 1, smpl_idx) for i in range(n_dims - 1)])
    Y = r * spherical_transform(X)
    return Y

