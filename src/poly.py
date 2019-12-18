import numpy as np
from sympy import factor_list, latex


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