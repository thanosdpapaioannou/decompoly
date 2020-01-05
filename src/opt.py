from sympy import nan, degree_list
from cvxopt import matrix, solvers
import numpy as np

from src.linalg import flatten, get_explicit_rep_objective, \
    is_symmetric_and_positive_definite, form_sos, get_pts_in_cvx_hull, get_explicit_form_basis
from src.poly import get_special_sos_multiplier, get_max_even_divisor, get_basis_repr, form_rat_gram_mat, \
    form_num_gram_mat, get_coeffs, get_sqroot_monoms
from src.util import get_rational_approximation

DSDP_OPTIONS = {'show_progress': False, 'DSDP_Monitor': 5, 'DSDP_MaxIts': 1000, 'DSDP_GapTolerance': 1e-07,
                'abstol': 1e-07, 'reltol': 1e-06, 'feastol': 1e-07}


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

    sym_grams = matrix(flatten(basis_matrices[1:])).T
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

    indices = np.array(list(poly.as_dict().keys()))
    monoms = get_pts_in_cvx_hull(indices)
    sqroot_monoms = get_pts_in_cvx_hull(1 / 2 * indices)
    coeffs = get_coeffs(poly)
    sym_mat_list_gram = get_explicit_form_basis(monoms, sqroot_monoms, coeffs)
    if len(sym_mat_list_gram) > 1:
        solv_status, sol_vec = sdp_expl_solve(sym_mat_list_gram, smallest_eig=epsilon * 10 ** 4, objective='max_trace')
        if solv_status == 'Optimal solution found':
            gram_mat_q = form_rat_gram_mat(sym_mat_list_gram, sol_vec, max_denom=1000)
            monom_vec = get_sqroot_monoms(poly)
            if get_basis_repr(gram_mat_q, monom_vec).as_poly() == poly:
                sos = form_sos(gram_mat_q, monom_vec)
                msg = 'Exact SOS decomposition found.'
                return msg, sos
            else:
                msg = 'Not an exact Gram matrix.'
                return msg, nan

        else:
            solv_status, sol_vec = sdp_expl_solve(sym_mat_list_gram, smallest_eig=eig_tol)
            if solv_status == 'Optimal solution found':
                gram_mat = form_num_gram_mat(sym_mat_list_gram, sol_vec)
                psd_status = is_symmetric_and_positive_definite(gram_mat, eig_tol=eig_tol)
                if not psd_status:
                    msg = 'No PSD Gram matrix found.'
                    return msg, nan

                gram_mat_q = form_rat_gram_mat(sym_mat_list_gram, sol_vec, max_denom=max_denom_rat_approx)
                psd_status = is_symmetric_and_positive_definite(gram_mat_q)
                if psd_status:
                    monom_vec = get_sqroot_monoms(poly)
                    if get_basis_repr(gram_mat_q, monom_vec).as_poly() == poly:
                        sos = form_sos(gram_mat_q, monom_vec)
                        msg = 'Exact SOS decomposition found.'
                        return msg, sos
                    else:
                        msg = 'Not an exact Gram matrix.'
                        return msg, nan
                else:
                    # Try again with larger denominator.
                    gram_mat_q = form_rat_gram_mat(sym_mat_list_gram, sol_vec, max_denom=10 ** 9 * max_denom_rat_approx)
                    psd_status = is_symmetric_and_positive_definite(gram_mat_q)
                    if psd_status:
                        monom_vec = get_sqroot_monoms(poly)
                        # if get_basis_repr(gram_mat_q, monom_vec, poly):
                        if get_basis_repr(gram_mat_q, monom_vec).as_poly() == poly:
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
        psd_status = is_symmetric_and_positive_definite(np.vectorize(float)(gram_mat_q))
        if psd_status:
            monom_vec = get_sqroot_monoms(poly)
            # if get_basis_repr(gram_mat_q, monom_vec, poly):
            if get_basis_repr(gram_mat_q, monom_vec).as_poly() == poly:
                sos = form_sos(gram_mat_q, monom_vec)
                msg = 'Exact SOS decomposition found.'
                return msg, sos
            else:
                msg = 'Not an exact Gram matrix.'
                return msg, nan
        else:
            msg = 'Unique Gram matrix not PSD. Not a sum of squares.'
            return msg, nan


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

    poly_indices = np.array(list(poly.as_dict().keys()))
    monoms = get_pts_in_cvx_hull(poly_indices)
    num_alpha = monoms.shape[0]

    if not num_alpha:
        _status = 'Error in computing monomial indices.'
        return _status, nan

    degree = poly.degree()
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
