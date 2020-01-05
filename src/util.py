from fractions import Fraction

import numba as nb
import numpy as np


@nb.njit
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@nb.njit
def np_amax(arr, axis):
    return np_apply_along_axis(np.amax, axis, arr)


@nb.njit
def np_amin(arr, axis):
    return np_apply_along_axis(np.amin, axis, arr)


@nb.njit
def get_rational_approximation_one_0_to_1(x, max_denom):
    """
    :param x: float between 0 and 1
    :param max_denom: max denominator of approximation
    :return: numerator and denominator where denominator < max_denominator such that numerator / denominator is optimal
    rational approximation of x
    """
    a, b = 0, 1
    c, d = 1, 1
    while b <= max_denom and d <= max_denom:
        mediant = float(a + c) / (b + d)
        if x == mediant:
            if b + d <= max_denom:
                return a + c, b + d
            elif d > b:
                return c, d
            else:
                return a, b
        elif x > mediant:
            a, b = a + c, b + d
        else:
            c, d = a + c, b + d
    if b > max_denom:
        return c, d
    else:
        return a, b


@nb.njit
def get_rational_approximation_one(x, max_denom):
    """
    :param x: float
    :param max_denom: max denominator of approximation
    :return: numerator and denominator where denominator < max_denominator such that numerator / denominator is optimal
    rational approximation of x
    """
    x_floor = int(np.floor(x))
    x_frac = x - x_floor
    _num, _denom = get_rational_approximation_one_0_to_1(x_frac, max_denom)
    return _num + _denom * x_floor, _denom


def get_rational_approximation(mat, max_denom):
    """
    :param mat: matrix of floats
    :param max_denom: positive integer
    :return: np.ndarray of Fractions which are the best rational approximations to the entries of mat with denominator
    bounded by max_denom.
    """

    array = np.array(mat)
    rationals = np.zeros_like(mat, dtype=Fraction)
    for (i, j), a in np.ndenumerate(array):
        num, denom = get_rational_approximation_one(a, max_denom)
        rationals[i, j] = Fraction(num, denom)
    return rationals


@nb.njit
def sym_coeff(tuple):
    x, y = tuple
    if x == y:
        return 1
    else:
        return 2
