from sympy import expand, Matrix, nan, degree_list
from cvxopt import matrix, solvers, spmatrix
from scipy.spatial import ConvexHull
import numpy as np
from scipy.linalg import null_space, orth, eigvalsh
from fractions import Fraction
import numba as nb

from src.linalg import get_lattice_pts_in_prism, form_constraint_eq_matrices
from src.poly import get_special_sos_multiplier, get_max_even_divisor
from src.util import get_rational_approximation, sym_coeff

DSDP_OPTIONS = {'show_progress': False, 'DSDP_Monitor': 5, 'DSDP_MaxIts': 1000, 'DSDP_GapTolerance': 1e-07,
                'abstol': 1e-07, 'reltol': 1e-06, 'feastol': 1e-07}


def get_pts_in_cvx_hull(mat, tolerance=1e-03):
    """
    :param mat: matrix whose rows are integer lattice points,
    :param tolerance:
    :return: matrix whose rows are the integer lattice points lying within the convex hull
    of these points to within a given tolerance. This includes the case in which the convex
    hull is less than full dimension.
    """

    nmons = mat.shape[0]
    a_0 = mat - np.repeat([mat[0]], nmons, axis=0)  # Translating so that span goes through origin.
    _null_space = null_space(a_0)
    _integer_pts = get_lattice_pts_in_prism(mat)
    if _null_space.shape[1] == 0:  # In this case, the convex hull has full dimension.
        __integer_pts = _integer_pts
    else:
        _dot_prod = np.abs(
            _integer_pts.dot(_null_space) - np.repeat([mat.dot(_null_space)[0]], _integer_pts.shape[0],
                                                      axis=0))  # Calculate dot product with null vectors
        include = np.all(np.less(_dot_prod, tolerance), axis=1)  # Include only points in same subspace up to tolerance
        __integer_pts = _integer_pts[list(include)]

    _orth = orth(a_0.T)
    if _orth.shape[1] > 1:
        _cvx_hull = ConvexHull(mat.dot(_orth))

        # Now check the points of __integer_pts against the inequalities that define the convex hull
        __cvx_hull = (_cvx_hull.equations[:, :-1].dot(_orth.T.dot(__integer_pts.T))
                      + _cvx_hull.equations[:, -1].reshape((_cvx_hull.equations.shape[0], 1)))
        include = np.all(np.less(__cvx_hull, tolerance), axis=0)
        ___integer_pts = __integer_pts[list(include)]
    else:
        """
        If the linear span of the points of mat is 1- or 0-dimensional, then there is no need to use inequalities given
        the construction of the get_lattice_pts_in_prism function.
        """
        ___integer_pts = __integer_pts
    return ___integer_pts


@nb.njit
def constr_eq_compat(poly_ind, sqroot_monoms):
    """
    :param poly_ind:
    :param sqroot_monoms:
    :return: Boolean value on whether the constraint equations admit a solution.
    The constraint equations are unsatisfiable only if there is a monomial term
    appearing in the polynomial that is not the sum of two points in sqroot_monoms.
    """

    compat = True
    for i in range(poly_ind.shape[0]):
        count = 0
        for j in range(sqroot_monoms.shape[0]):
            for k in range(j, sqroot_monoms.shape[0]):
                if np.all(poly_ind[i] == sqroot_monoms[j] + sqroot_monoms[k]):
                    count += 1
        if count == 0:
            compat = False
    return compat


@nb.njit
def form_sdp_constraint_dense(matrix_list):
    """
    :param matrix_list: list of matrices
    :return: list of matrices of matrix_list each reformatted to the format required by solvers.sdp function.
    """

    num_constr = len(matrix_list)
    row_size = matrix_list[0].shape[0] ** 2
    constr = np.zeros((num_constr, row_size))
    for i in range(num_constr):
        constr[i] = matrix_list[i].reshape((1, row_size))
    return constr


# jit doesn't support sparse matrices (spmatrix) used here
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


def get_explicit_form_basis(monoms, sqroot_monoms, poly):
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
    gram_mats_sym = []
    coeff = get_coeffs(poly)

    for i in range(param - len(constr) + 1):
        gram_mats_sym.append(np.zeros((dim, dim)))

    num = 1
    for i in range(len(constr)):
        count = len(constr[i].I)
        for j in range(count - 1):
            gram_mats_sym[num + j][constr[i].I[j], constr[i].J[j]] = 1
            gram_mats_sym[num + j][constr[i].I[-1], constr[i].J[-1]] = -sym_coeff(constr[i].I[j],
                                                                                  constr[i].J[j]) / sym_coeff(
                constr[i].I[-1], constr[i].J[-1])
        gram_mats_sym[0][constr[i].I[-1], constr[i].J[-1]] = coeff[i] / sym_coeff(constr[i].I[-1], constr[i].J[-1])
        num += count - 1
    for i in range(len(gram_mats_sym)):
        # make gram_mats_sym[i] symmetric by accessing only upper-triang elts, and copying them onto lower-triang elts:
        gram_mats_sym[i] = np.tril(gram_mats_sym[i].T) + np.triu(gram_mats_sym[i], 1)
    return gram_mats_sym


@nb.njit
def get_explicit_rep_objective(sym_matrix_list):
    """
    :param sym_matrix_list:
    :return:
    column objective vector d in the SDP written in explicit form,
    so that the objective is to minimize d^T y.
    The objective is chosen to correspond to the identity matrix in the implicit representation.
    """

    obj_vec = np.zeros((len(sym_matrix_list) - 1, 1))
    for i in range(1, len(sym_matrix_list)):
        obj_vec[i - 1, 0] = np.trace(sym_matrix_list[i])
    return obj_vec


def get_sqroot_monoms(poly):
    """
    :param poly:
    :return: column vector of monomials, the basis of the space of polynomials
    whose square is in the convex hull of the monomials of poly.
    """

    poly_indices = np.array(poly.as_poly().monoms())
    sqroot_monoms = get_pts_in_cvx_hull(1 / 2 * poly_indices)
    monom_vec = Matrix.ones(sqroot_monoms.shape[0], 1)
    for i in range(sqroot_monoms.shape[0]):
        for j in range(sqroot_monoms.shape[1]):
            monom_vec[i, 0] *= poly.as_poly().gens[j] ** sqroot_monoms[i, j]
    return monom_vec


def lu_to_sq_factors(lt, ut, perm, monom_vec):
    """
    :param lt: L in the LU decomposition of a rational PSD Gram matrix
    :param ut: U in the LU decomposition of a rational PSD Gram matrix
    :param perm: list of transpositions returned by LU decomposition
    :param monom_vec: vector of monomials in 1/2*ConvexHull(poly)
    :return: two lists corresponding to the SOS decomposition of poly,
    a list of positive factors, and a list of the polynomial factors to be squared.
    """

    perm_mat = Matrix.eye(lt.shape[0]).permuteFwd(perm)
    perm_vec = perm_mat * monom_vec
    pos_coeffs = []
    for i in range(ut.shape[0]):
        pos_coeffs.append((ut * perm_mat.transpose())[i, i])

    sq_factors = []
    for i in range(lt.shape[0]):
        sq_factors.append(lt[:, i].transpose().dot(perm_vec))
    return pos_coeffs, sq_factors


def sdp_expl_solve(basis_matrices, smallest_eig=0, objective='zero', dsdp_solver='dsdp', dsdp_options=DSDP_OPTIONS):
    """
    :param basis_matrices: list of symmetric matrices G_0, G_1, ..., G_n of same size
    :param smallest eig: parameter (default 0) may be set to small positive quantity to force non-degeneracy
    :param objective: string parameter, either 'zero', 'min_trace', or 'max_trace' (default 'zero'), determines
    the objective in the SDP solver
    :param dsdp_solver: string, default 'dsdp' to specify which solver sdp.solver uses
    :param dsdp_options:
    :return: solver_status, a string, either 'optimal', 'infeasible', or 'unknown', and sol_vec, a vector approximately
    optimizing the SDP problem if solver_status is 'optimal', and nan instead
    """

    sym_grams = matrix(form_sdp_constraint_dense(basis_matrices[1:])).T
    if objective == 'zero':
        obj_vec = matrix(np.zeros((len(basis_matrices) - 1, 1)))
    elif objective == 'min_trace':
        obj_vec = matrix(get_explicit_rep_objective(basis_matrices))
    else:
        # Maximize trace in nondegenerate case
        obj_vec = -matrix(get_explicit_rep_objective(basis_matrices))

    sol = solvers.sdp(c=obj_vec, Gs=[-sym_grams],
                      hs=[matrix(basis_matrices[0] - smallest_eig * np.eye(basis_matrices[0].shape[0]))],
                      solver=dsdp_solver, options=dsdp_options)

    if sol['status'] == 'optimal':
        solver_status = 'Optimal solution found'
        sol_vec = sol['x']
    elif sol['status'] == 'primal infeasible' or sol['status'] == 'dual infeasible':
        solver_status = 'infeasible'
        sol_vec = nan
    else:
        solver_status = 'unknown'
        sol_vec = nan

    return solver_status, sol_vec


def get_sos_helper(poly, eig_tol=-1e-07, epsilon=1e-07, max_denom_rat_approx=100):
    """
    :param poly: sympy polynomial
    :param eig_tol:
    :param epsilon:
    :param max_denom_rat_approx:
    :return: string with status whether poly is a sum of squares of polynomials, and a sympy expression that is
    the SOSRF decomposition of the poly
    """

    poly_indices = np.array(list(poly.as_poly().as_dict().keys()))
    monoms = get_pts_in_cvx_hull(poly_indices)
    sqroot_monoms = get_pts_in_cvx_hull(1 / 2 * poly_indices)
    # num_beta = sqroot_monoms.shape[0]
    sym_mat_list_gram = get_explicit_form_basis(monoms, sqroot_monoms, poly)
    if len(sym_mat_list_gram) > 1:
        solv_status, sol_vec = sdp_expl_solve(sym_mat_list_gram, smallest_eig=epsilon * 10 ** 4, objective='max_trace')
        if solv_status == 'Optimal solution found':
            gram_mat_q = form_rat_gram_mat(sym_mat_list_gram, sol_vec, max_denom=1000)
            psd_status, char_poly = check_psd_rational(gram_mat_q)
            if psd_status:
                monom_vec = get_sqroot_monoms(poly)
                if check_gram_exact(gram_mat_q, monom_vec, poly) == 'exact':
                    sos = form_sos(gram_mat_q, monom_vec)
                    msg = 'Exact SOS decomposition found.'
                    return msg, sos
                else:
                    msg = 'Not an exact Gram matrix.'
                    return msg, nan
            else:
                msg = 'Error. Solution not PSD'
                return msg, nan

        else:
            solv_status, sol_vec = sdp_expl_solve(sym_mat_list_gram, smallest_eig=eig_tol)
            if solv_status == 'Optimal solution found':
                gram_mat = form_num_gram_mat(sym_mat_list_gram, sol_vec)
                is_psd, eigs = check_psd_numerical(gram_mat, eig_tol=eig_tol)
                if is_psd == 'not PSD':
                    msg = 'No PSD Gram matrix found.'
                    return msg, nan

                gram_mat_q = form_rat_gram_mat(sym_mat_list_gram, sol_vec, max_denom=max_denom_rat_approx)
                psd_status, char_poly = check_psd_rational(gram_mat_q)
                if psd_status:
                    monom_vec = get_sqroot_monoms(poly)
                    if check_gram_exact(gram_mat_q, monom_vec, poly) == 'exact':
                        sos = form_sos(gram_mat_q, monom_vec)
                        msg = 'Exact SOS decomposition found.'
                        return msg, sos
                    else:
                        msg = 'Not an exact Gram matrix.'
                        return msg, nan
                else:
                    # Try again with larger denominator.
                    gram_mat_q = form_rat_gram_mat(sym_mat_list_gram, sol_vec, max_denom=10 ** 9 * max_denom_rat_approx)
                    psd_status, char_poly = check_psd_rational(gram_mat_q)
                    if psd_status:
                        monom_vec = get_sqroot_monoms(poly)
                        if check_gram_exact(gram_mat_q, monom_vec, poly) == 'exact':
                            sos = form_sos(gram_mat_q, monom_vec)
                            msg = 'Exact SOS decomposition found.'
                            return msg, sos
                        else:
                            msg = 'Not an exact Gram matrix.'
                            return msg, nan
                    else:
                        msg = 'Could not find exact PSD Gram matrix.'
                        return msg, nan

            else:
                msg = 'SDP solver could not find solution.'
                return msg, nan

    else:
        # Unique Gram matrix. No need for SDP.
        # The max denominator below should be changed to twice the largest denominator appearing as a coeff in poly.
        gram_mat_q = get_rational_approximation(sym_mat_list_gram[0], max_denom_rat_approx)
        psd_status, char_poly = check_psd_rational(gram_mat_q)
        if psd_status:
            monom_vec = get_sqroot_monoms(poly)
            if check_gram_exact(gram_mat_q, monom_vec, poly) == 'exact':
                sos = form_sos(gram_mat_q, monom_vec)
                msg = 'Exact SOS decomposition found.'
                return msg, sos
            else:
                msg = 'Not an exact Gram matrix.'
                return msg, nan
        else:
            msg = 'Unique Gram matrix not PSD. Not a sum of squares.'
            return msg, nan


def form_sos(gram_mat_q, monom_vec):
    """
    :param gram_mat_q: a rational symmetric PSD matrix
    :param monom_vec: basis vector of monomials corresponding to gram_mat_q
    :return: sos, an expression consisting of a sum-of-squares decomposition of the polynomial
    with Gram matrix gram_mat_q
    """
    lt, ut, perm = Matrix(gram_mat_q).LUdecomposition()
    # print(Matrix(gram_mat_q).LUdecomposition())

    coeffs, factors = lu_to_sq_factors(lt, ut, perm, monom_vec)
    # print(coeffs)
    # print(factors)
    # sos = np.sum([_c * factors[i] ** 2 for i, _c in enumerate(coeffs)]).as_poly()
    sos = np.sum([_c * factors[i] ** 2 for i, _c in enumerate(coeffs)]).as_expr()
    # msg = 'Exact SOS decomposition found.'
    return sos


def check_psd_rational(sym_rat_mat):
    """
    :param sym_rat_mat: symmetric rational matrix
    :return: not wrong_sign, a boolean expression, True if sym_rat_mat is PSD, False if not, and char_poly,
    a polynomial object, the characteristic polynomial of sym_rat_mat
    """
    char_poly = Matrix(sym_rat_mat).charpoly()
    char_coeffs = char_poly.all_coeffs()
    wrong_sign = False
    for i, _c in enumerate(char_coeffs):
        if (-1) ** i * _c < 0:
            wrong_sign = True

    return not wrong_sign, char_poly


def check_psd_numerical(sym_mat, eig_tol=-10 ** (-7)):
    """
    :param sym_mat: symmetric matrix of floats
    :param eig_tol: float, default -10**(-7)
    :return: string message, either 'PSD' or 'not PSD' according to whether the smallest eigenvalue
    computed by eigvalsh is greater than eig_tol
    """
    eigs = eigvalsh(sym_mat)
    if eigs[0] < eig_tol:
        status = 'not PSD'
    else:
        status = 'PSD'
    return status, eigs


def check_gram_exact(sym_rat_mat, monom_vec, poly):
    """
    :param sym_rat_mat: n*n symmetric matrix of rational numbers
    :param monom_vec: n*1 basis vector of monomials
    :param poly: polynomial
    :return: string, either 'exact' or 'not exact' according to whether monom_vec^T sym_rat_mat monom_vec = poly
    """
    # Check that v^T Q v = poly, where v is the monomial vector.
    check_poly = expand((Matrix(monom_vec).transpose() * Matrix(sym_rat_mat) * Matrix(monom_vec))[0, 0])
    # print(check_poly)
    if check_poly.as_poly() == poly.as_poly():
        status = 'exact'
    else:
        status = 'not exact'

    return status


def form_rat_gram_mat(basis_matrices, sol_vec_numerical, max_denom):
    """
    :param basis_matrices: list of k+1 n*n symmetric matrices
    :param sol_vec_numerical: k*1 vector
    :param max_denom: positive integer
    :return: finds best rational approximation rat_approx to sol_vec_numerical for which each entry has denominator
    bounded by max_denom, and returns symmetric matrix of rationals basis_matrices[0] + basis_matrices[1]*rat_approx[1]+...
    + basis_matrices[k]*rat_approx[k]
    """
    rat_approx = get_rational_approximation(sol_vec_numerical, max_denom)
    gram_mat_q = np.zeros_like(basis_matrices[0], dtype=Fraction)
    for i in range(len(basis_matrices) - 1):
        gram_mat_q += get_rational_approximation(basis_matrices[i + 1], 10) * rat_approx[i]
    gram_mat_q += get_rational_approximation(basis_matrices[0], 10)

    return gram_mat_q


def form_num_gram_mat(basis_matrices, sol_vec_numerical):
    """
    :param basis_matrices: list of k+1 n*n symmetric matrices
    :param sol_vec_numerical: k*1 vector
    :return: symmetric matrix  basis_matrices[0] + basis_matrices[1]*sol_vec_numerical[1]+...
    + basis_matrices[k]*sol_vec_numerical[k]
    """
    gram_mat = basis_matrices[0]
    for i in range(len(basis_matrices) - 1):
        gram_mat += basis_matrices[i + 1] * sol_vec_numerical[i]

    return gram_mat


def get_sos(poly, max_mult_power=3, dsdp_solver='dsdp', dsdp_options=DSDP_OPTIONS, eig_tol=-1e-07, epsilon=1e-07):
    """
    :param poly: sympy polynomial
    :param max_mult_power:
    :param dsdp_solver:
    :param dsdp_options:
    :param eig_tol:
    :param epsilon:
    :return: string with status whether poly is a sum of squares of polynomials, and a sympy expression that is
    the SOSRF decomposition of the poly
    """
    if poly == 0:
        _status = 'Zero polynomial.'
        return _status, nan

    # check polynomial is nonconstant
    if np.all([_d == 0 for _d in degree_list(poly)]):
        _status = 'Constant polynomial.'
        return _status, nan

    poly_indices = np.array(list(poly.as_poly().as_dict().keys()))
    monoms = get_pts_in_cvx_hull(poly_indices)
    num_alpha = monoms.shape[0]

    if not num_alpha:
        _status = 'Error in computing monomial indices.'
        return _status, nan

    degree = poly.as_poly().degree()
    if degree % 2:
        _status = 'Polynomial has odd degree. Not a sum of squares.'
        return _status, nan

    coeff_leading, max_even_divisor, remainder = get_max_even_divisor(poly)
    if remainder == 1:
        _status = 'Exact SOS decomposition found.'
        sos = coeff_leading * max_even_divisor
    else:
        _mult = get_special_sos_multiplier(remainder)
        for r in range(max_mult_power):
            print(f'Trying multiplier power: {r}')
            status_, sos_ = get_sos_helper(poly=(_mult ** r * remainder).as_poly(), eig_tol=eig_tol, epsilon=epsilon)
            if status_ == 'Exact SOS decomposition found.':
                _status = 'Exact SOS decomposition found.'
                sos = (1 / _mult ** r) * coeff_leading * max_even_divisor * sos_.as_expr()
                break
        else:
            _status = 'No exact SOS decomposition found.'
            sos = nan
    return _status, sos


def get_coeffs(poly):
    """
    :param poly: multivariable sympy poly
    :return: vector of coefficients, including zeros for all multi-indices
    in the convex hull of multi-indices appearing in poly.
    Includes case where multi-indices in poly have less than full-dimensional
    convex hull.
    """

    indices = np.array(list(poly.as_poly().as_dict().keys()))
    mat = get_pts_in_cvx_hull(indices)
    mat_other = get_pts_in_cvx_hull(1 / 2 * indices)
    num_nontriv_eq = len(form_constraint_eq_matrices(mat, mat_other))
    coeff_vec = np.zeros(num_nontriv_eq)
    for i in range(num_nontriv_eq):
        if tuple(mat[i]) in poly.as_poly().as_dict().keys():
            coeff_vec[i] = poly.as_poly().as_dict()[tuple(mat[i])]
    return coeff_vec