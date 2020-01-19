import numba as nb
import numpy as np
from cvxopt.base import spmatrix
from scipy.linalg import null_space, orth
from scipy.spatial.qhull import ConvexHull
from sympy import Matrix
from src.util import np_amax, np_amin, sym_coeff


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


def form_sos(gram_mat_q, monom_vec):
    """
    :param gram_mat_q: a rational symmetric PSD matrix
    :param monom_vec: basis vector of monomials corresponding to gram_mat_q in 1/2*ConvexHull(poly)
    :return: sos, an expression consisting of a sum-of-squares decomposition of the polynomial
    with Gram matrix gram_mat_q
    """
    L, U, p = Matrix(gram_mat_q).LUdecomposition()
    perm_mat = Matrix.eye(L.shape[0]).permuteFwd(p)
    perm_vec = perm_mat * monom_vec

    coeffs = np.array((U * perm_mat.transpose())).diagonal()
    n = len(monom_vec)
    factors = [L[:, i].transpose().dot(perm_vec) for i in range(n)]
    sos = np.dot(coeffs, [f ** 2 for f in factors]).as_expr()
    return sos


def get_pts_in_cvx_hull(mat, tolerance=1e-03):
    """
    :param mat: matrix whose rows are integer lattice points,
    :param tolerance:
    :return: matrix whose rows are the integer lattice points lying within the convex hull
    of these points to within a given tolerance. This includes the case in which the convex
    hull is less than full dimension.
    """

    m = mat.shape[0]
    mat_0 = mat - np.repeat([mat[0]], m, axis=0)  # Translating so that span goes through origin.
    _null_space = null_space(mat_0)
    _integer_pts = get_lattice_pts_in_prism(mat)
    if _null_space.shape[1]:
        _dot_prod = np.abs(
            _integer_pts.dot(_null_space) - np.repeat([mat.dot(_null_space)[0]], _integer_pts.shape[0],
                                                      axis=0))  # Calculate dot product with null vectors
        include = np.all(np.less(_dot_prod, tolerance), axis=1)  # Include only points in same subspace up to tolerance
        _integer_pts = _integer_pts[list(include)]

    _orth = orth(mat_0.T)
    if _orth.shape[1] > 1:
        _cvx_hull = ConvexHull(mat.dot(_orth))

        # Now check the points of _integer_pts against the inequalities that define the convex hull
        __cvx_hull = (_cvx_hull.equations[:, :-1].dot(_orth.T.dot(_integer_pts.T))
                      + _cvx_hull.equations[:, -1].reshape((_cvx_hull.equations.shape[0], 1)))
        include = np.all(np.less(__cvx_hull, tolerance), axis=0)
        _integer_pts = _integer_pts[list(include)]
    return _integer_pts


def form_coeffs_constraint_eq_sparse_upper(monoms, sqroot_monoms):
    """
    Forms the coefficients of the constraint equations given matrices monoms, sqroot_monoms
    whose rows correspond to the multi-indices in the convex hull and 1/2 the convex hull
    of the multi-indices of a polynomial.
    Constraint matrices are returned in spmatrix form; only upper triangular elements given.
    :param monoms:
    :param sqroot_monoms:
    :return:
    """

    num = sqroot_monoms.shape[0]
    constraints = []
    for i in range(monoms.shape[0]):
        constraint_i_rows = []
        constraint_i_cols = []
        count_nontriv = 0
        for j in range(num):
            for k in range(j, num):
                if np.all(monoms[i] == (sqroot_monoms[j] + sqroot_monoms[k])):
                    constraint_i_rows.append(j)
                    constraint_i_cols.append(k)
                    count_nontriv += 1
        if count_nontriv:
            constraints.append(spmatrix(1, constraint_i_rows, constraint_i_cols, (num, num)))
    return constraints


def get_explicit_form_basis(monoms, sqroot_monoms, coeffs):
    """
    :param monoms:
    :param sqroot_monoms:
    :param poly: sympy poly
    :return: tuple of symmetric |sqroot_monoms|*|sqroot_monoms| matrices (G_0,G_1,...,G_n),
    n = |sqroot_monoms|*(|sqroot_monoms|+1)/2 - (number of nontrivial constraints),
    where G(y) = G_0 + G_1 y_1 + ... + G_n y_n, y in R^n
    parametrizes the set of Gram matrices for the polynomial.
    """

    dim = sqroot_monoms.shape[0]
    param = int(dim * (dim + 1) / 2)
    constr = form_coeffs_constraint_eq_sparse_upper(monoms, sqroot_monoms)
    gram_mats_sym = [np.zeros((dim, dim)) for _ in range(param - len(constr) + 1)]

    num = 1
    for i, _c in enumerate(constr):
        count = len(_c.I)
        _t = (_c.I[-1], _c.J[-1])
        _st = sym_coeff(_t)
        for j in range(count - 1):
            _t_other = (_c.I[j], _c.J[j])
            gram_mats_sym[num + j][_t_other] = 1
            gram_mats_sym[num + j][_t] = -sym_coeff(_t_other) / _st
        gram_mats_sym[0][_t] = coeffs[i] / _st
        num += count - 1

    # make each matrix in gram_mats_sym symmetric by accessing only upper-triang elts, and copying them onto lower-triang elts:
    gram_mats_sym = [np.tril(g.T) + np.triu(g, 1) for g in gram_mats_sym]
    return gram_mats_sym
