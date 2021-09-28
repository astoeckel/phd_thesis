import numpy as np

def lstsq_cstr(A, b, C, d, reg=1e-3):
    A, b, C, d = np.array(A), np.array(b), np.array(C), np.array(d)

    assert A.ndim == 2
    assert b.ndim == 1
    assert C.ndim == 2
    assert d.ndim == 1

    assert A.shape[0] == b.shape[0]
    assert A.shape[1] == C.shape[1]
    assert C.shape[0] == d.shape[0]

    N_smpls = A.shape[0]
    N_vars = A.shape[1]
    N_cstr = C.shape[0]

    P = np.zeros((N_vars + N_cstr, N_vars + N_cstr))
    P[:N_vars, :N_vars] = 2 * (A.T @ A + N_smpls * np.square(reg))
    P[:N_vars, N_vars:] = -C.T
    P[N_vars:, :N_vars] = C

    q = np.zeros((N_vars + N_cstr))
    q[:N_vars] = 2 * A.T @ b
    q[N_vars:] = d

    return np.linalg.solve(P, q)[:N_vars]

