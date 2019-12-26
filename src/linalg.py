import numba as nb
import numpy as np
from scipy.linalg import eigvalsh
from sympy import Matrix

from src.util import np_amax, np_amin


@nb.njit
def get_lattice_pts_in_prism(mat):
    """
    :param mat: matrix of row vectors with integer entries.
    :return: matrix whose rows are all integer lattice points in the smallest rectangular prism
    containing the points of mat.
    """

    n = mat.shape[1]
    column_min = np_amin(mat, axis=0)
    size_vector = (np_amax(mat, axis=0) - column_min + 1).astype(np.int64)
    prod_vec = np.cumprod(size_vector)
    prod = prod_vec[-1]

    _lattice_pts = np.zeros((prod, n), dtype=np.int64)
    add_vec = np.zeros(n, dtype=np.int64)
    for i in range(prod):
        add_vec[0] = i % prod_vec[0]
        for j in range(1, n):
            add_vec[j] = np.int64(i / prod_vec[j - 1]) % size_vector[j]
        _lattice_pts[i] = column_min + add_vec
    return _lattice_pts


@nb.njit
def form_constraint_eq_matrices(mat, mat_other):
    """
    :param mat: matrix whose rows are integer vectors.
    :param mat_other: matrix whose rows are integer vectors.
    :return: list of sparse, symmetric matrices, one for each row of mat,
    with size the number of rows of mat_other,
    having a 1 in the entry indexed by (beta,beta') if beta + beta' = alpha,
    and 0 otherwise.
    """

    _mat = []
    m_other = mat_other.shape[0]
    for i in range(mat.shape[0]):
        mat_i = np.zeros((m_other, m_other))
        point_count = 0
        for j in range(m_other):
            for k in range(j, m_other):
                if np.all(mat[i] == (mat_other[j] + mat_other[k])):
                    mat_i[j, k] = 1
                    mat_i[k, j] = 1
                    point_count += 1
        if point_count > 0:
            _mat.append(mat_i)
    return _mat


def flatten(matrix_list):
    """
    :param matrix_list: list of matrices
    :return: list of matrices of matrix_list each reformatted to the format required by solvers.sdp function.
    """
    constr = np.array([_a.flatten() for _a in matrix_list])
    return constr


@nb.njit
def get_explicit_rep_objective(sym_matrix_list):
    """
    :param sym_matrix_list:
    :return:
    column objective vector d in the SDP written in explicit form,
    so that the objective is to minimize d^T y.
    The objective is chosen to correspond to the identity matrix in the implicit representation.
    """

    obj_vec = [np.trace(_m) for _m in sym_matrix_list]
    return obj_vec[1:]


def is_symmetric_and_positive_definite(sym_mat):
    if np.allclose(sym_mat, sym_mat.T):
        try:
            np.linalg.cholesky(sym_mat)
            return True
        except np.linalg.LinAlgError as err:
            if 'Matrix is not positive definite' in str(err):
                return False
            else:
                raise
    else:
        return False
