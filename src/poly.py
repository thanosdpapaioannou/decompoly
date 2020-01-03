import numpy as np
from sympy import factor_list, latex, expand, Matrix, poly

from src.linalg import get_pts_in_cvx_hull, form_constraint_eq_matrices
from src.util import get_rational_approximation


def get_special_sos_multiplier(poly):
    """
    :param poly: sympy polynomial
    :return: factorisation of poly
    """
    _symbols = [1] + list(poly.free_symbols)
    _mult = np.sum([_s ** 2 for _s in _symbols]).as_poly()
    return _mult


def get_max_even_divisor(poly):
    """
    :param poly: sympy polynomial
    :return: leading coefficient of poly, max polynomial divisor of poly that's even power, remainder of poly / max_divisor
    """
    _factors = factor_list(poly)
    _coeff_leading = _factors[0]
    _factors_non_constant = _factors[1]
    _factors_max_even_divisor = [(_p, 2 * (n // 2)) for (_p, n) in _factors_non_constant]
    _factors_remainder = [(_p, n - 2 * (n // 2)) for (_p, n) in _factors_non_constant]
    _max_even_divisor = np.prod([_p.as_expr() ** n for (_p, n) in _factors_max_even_divisor])
    _remainder = np.prod([_p.as_expr() ** n for (_p, n) in _factors_remainder])
    return _coeff_leading, _max_even_divisor, _remainder


def get_latex_from_poly(poly):
    _latex_string = latex(poly, mode='plain')
    _poly_str = _latex_string.split(',')[0].replace('\operatorname{Poly}{\left(', '').strip()
    return _poly_str


def get_basis_repr(sym_mat, monom_vec):
    """
    :param sym_mat: n*n symmetric matrix of rational numbers
    :param monom_vec: n*1 basis vector of monomials
    :return: monom_vec^T sym_mat monom_vec
    """
    # Check that v^T Q v = poly, where v is the monomial vector.
    _repr_poly = expand((Matrix(monom_vec).transpose() * Matrix(sym_mat) * Matrix(monom_vec))[0, 0])
    return _repr_poly


def form_num_gram_mat(basis_matrices, sol_vec):
    """
    :param basis_matrices: list of k+1 n*n symmetric matrices
    :param sol_vec: k*1 vector
    :return: symmetric matrix  basis_matrices[0] + basis_matrices[1]*sol_vec[1]+...
    + basis_matrices[k]*sol_vec[k]
    """
    gram_mat = basis_matrices[0]
    for i in range(len(basis_matrices) - 1):
        gram_mat += basis_matrices[i + 1] * sol_vec[i]
    return gram_mat


def form_rat_gram_mat(basis_matrices, sol_vec, max_denom):
    """
    :param basis_matrices: list of k+1 n*n symmetric matrices
    :param sol_vec: k*1 vector
    :param max_denom: positive integer
    :return: finds best rational approximation rat_sol_vec to sol_vec for which each entry has denominator
    bounded by max_denom, and returns symmetric matrix of rationals basis_matrices[0] + basis_matrices[1]*rat_sol_vec[1]+...
    + basis_matrices[k]*rat_sol_vec[k]
    """
    rat_sol_vec = get_rational_approximation(sol_vec, max_denom)
    rat_basis_matrices = [get_rational_approximation(b, max_denom) for b in basis_matrices]
    gram_mat_q = form_num_gram_mat(rat_basis_matrices, rat_sol_vec)
    return gram_mat_q


def is_polynomial(input):
    try:
        _ = poly(input)
    except:
        return False
    return True


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