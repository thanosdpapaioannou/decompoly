from sympy import nan, degree_list, Matrix
from cvxopt import matrix, solvers
import numpy as np

from src.linalg import flatten, get_explicit_rep_objective, \
    is_symmetric_and_positive_definite, form_sos, get_pts_in_cvx_hull, get_explicit_form_basis
from src.poly import get_special_sos_multiplier, get_max_even_divisor, get_basis_repr, form_rat_gram_mat, get_coeffs
from src.util import get_rational_approximation

DSDP_OPTIONS = {'show_progress': False, 'DSDP_Monitor': 5, 'DSDP_MaxIts': 1000, 'DSDP_GapTolerance': 1e-07,
                'abstol': 1e-07, 'reltol': 1e-06, 'feastol': 1e-07}


def sdp_expl_solve(sym_mat_list, smallest_eig=0.001):
    """
    :param sym_mat_list: list of symmetric matrices G_0, G_1, ..., G_n of same size
    :param smallest eig: parameter (default 0) may be set to small positive quantity to force non-degeneracy
    :param objective: string parameter, either 'zero', or 'max_trace' (default 'zero'), determines
    the objective in the SDP solver
    :param dsdp_solver: string, default 'dsdp' to specify which solver sdp.solver uses
    :param dsdp_options:
    :return: solver_status, a string, either 'optimal', 'infeasible', or 'unknown', and sol_vec, a vector approximately
    optimizing the SDP problem if solver_status is 'optimal', and nan instead
    """
    obj_vec = -matrix(get_explicit_rep_objective(sym_mat_list))
    # print(obj_vec)
    _hs = sym_mat_list[0] - smallest_eig * np.eye(sym_mat_list[0].shape[0])
    sym_grams = matrix(flatten(sym_mat_list[1:])).T
    sol = solvers.sdp(c=obj_vec, Gs=[-sym_grams], hs=[matrix(_hs)], solver='dsdp', options=DSDP_OPTIONS)
    _status = sol['status']
    if _status == 'optimal':
        return 'Optimal solution found', sol['x']
    elif 'infeasible' in _status:
        return 'infeasible', nan
    else:
        return 'unknown', nan


def get_sos_helper(poly, epsilon=0.001, max_denom_rat_approx=100):
    """
    :param poly: sympy polynomial
    :param epsilon:
    :param max_denom_rat_approx:
    :return: string with status whether poly is a sum of squares of polynomials, and a sympy expression that is
    the SOSRF decomposition of the poly
    """
    indices = np.array(list(poly.as_dict().keys()))
    monoms = get_pts_in_cvx_hull(indices)
    sqroot_monoms = get_pts_in_cvx_hull(1 / 2 * indices)
    coeffs = get_coeffs(poly)
    sym_mat_list_gram = get_explicit_form_basis(monoms, sqroot_monoms, coeffs)
    if len(sym_mat_list_gram) == 1:
        # Unique Gram matrix. No need for SDP.
        gram_mat_q = get_rational_approximation(sym_mat_list_gram[0], max_denom_rat_approx)
        psd_status = is_symmetric_and_positive_definite(np.vectorize(float)(gram_mat_q))
        if not psd_status:
            status_ = 'Unique Gram matrix not PSD. Not a sum of squares.'
            return status_, nan
    else:
        _status, sol_vec = sdp_expl_solve(sym_mat_list_gram, smallest_eig=epsilon)
        if _status == 'Optimal solution found':
            gram_mat_q = form_rat_gram_mat(sym_mat_list_gram, sol_vec, max_denom=max_denom_rat_approx)
        else:
            status_ = 'Not an exact Gram matrix.'
            return status_, nan

    m, n = sqroot_monoms.shape
    monom_vec = Matrix.ones(m, 1)
    for i in range(m):
        for j in range(n):
            monom_vec[i, 0] *= poly.gens[j] ** sqroot_monoms[i, j]

    assert get_basis_repr(gram_mat_q, monom_vec).as_poly() == poly
    status_ = 'Exact SOS decomposition found.'
    sos = form_sos(gram_mat_q, monom_vec)
    return status_, sos


def get_sos(poly, max_mult_power=3, epsilon=0.001):
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

    # check polynomial is nonconstant and even-degreed in every variable
    if poly == 0:
        _status = 'Zero polynomial.'
        return _status, nan
    else:
        _degree_list = degree_list(poly)
        if np.all([_d == 0 for _d in _degree_list]):
            _status = 'Constant polynomial.'
            return _status, nan
        elif np.any([_d % 2 for _d in _degree_list]):
            _status = 'One of the variables in the polynomial has odd degree. Not a sum of squares.'
            return _status, nan

    indices = np.array(list(poly.as_dict().keys()))
    monoms = get_pts_in_cvx_hull(indices)
    num_alpha = monoms.shape[0]
    if not num_alpha:
        _status = 'Error in computing monomial indices.'
        return _status, nan

    coeff_leading, max_even_divisor, remainder = get_max_even_divisor(poly)
    if remainder == 1:
        _status = 'Exact SOS decomposition found.'
        sos = coeff_leading * max_even_divisor
    else:
        _mult = get_special_sos_multiplier(remainder)
        for r in range(max_mult_power):
            print(f'Trying multiplier power: {r}')
            status_, sos_ = get_sos_helper(poly=(_mult ** r * remainder).as_poly(), epsilon=epsilon)
            if status_ == 'Exact SOS decomposition found.':
                _status = status_
                sos = (1 / _mult ** r) * coeff_leading * max_even_divisor * sos_.as_expr()
                break
        else:
            _status = 'No exact SOS decomposition found.'
            sos = nan
    return _status, sos
