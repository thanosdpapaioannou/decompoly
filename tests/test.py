from unittest import TestCase
from sympy import poly, expand, nan, simplify
from fractions import Fraction
from src.opt import get_sos
from src.util import get_rational_approximation_one_0_to_1, get_rational_approximation_one
from src.linalg import is_symmetric_and_positive_definite
import numpy as np


class TestGet_sos(TestCase):

    def test_is_symmetric_and_positive_definite_0(self):
        sym_mat = np.array([[1, 0], [0, 1]])
        self.assertEqual(is_symmetric_and_positive_definite(sym_mat), True)

    def test_is_symmetric_and_positive_definite_1(self):
        sym_mat = np.array([[5, 3, 9], [4, 5, 7], [9, 7, 5]])
        self.assertEqual(is_symmetric_and_positive_definite(sym_mat), False)

    def test_is_symmetric_and_positive_definite_2(self):
        sym_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(is_symmetric_and_positive_definite(sym_mat), True)

    def test_get_rational_approximation_one_0_to_1_0(self):
        a = 0.5
        max_denom = 2
        num, denom = get_rational_approximation_one_0_to_1(a, max_denom=max_denom)
        self.assertEqual(Fraction(num, denom), Fraction.from_float(a).limit_denominator(max_denom))

    def test_get_rational_approximation_one_0_to_1_1(self):
        a = 0.44
        max_denom = 100
        num, denom = get_rational_approximation_one_0_to_1(a, max_denom=max_denom)
        self.assertEqual(Fraction(num, denom), Fraction.from_float(a).limit_denominator(max_denom))

    def test_get_rational_approximation_one_0_to_1_3(self):
        a = 0.36855050
        max_denom = 10000
        num, denom = get_rational_approximation_one_0_to_1(a, max_denom=max_denom)
        num_, denom_ = get_rational_approximation_one(a, max_denom=max_denom)
        self.assertAlmostEqual(num, num_)
        self.assertAlmostEqual(denom, denom_)

    def test_get_rational_approximation_one_4(self):
        a = 1.5
        max_denom = 3
        num, denom = get_rational_approximation_one(a, max_denom=max_denom)
        self.assertEqual(Fraction(num, denom), Fraction.from_float(a).limit_denominator(max_denom))

    def test_get_rational_approximation_one_0_to_1_5(self):
        a = 2.5
        max_denom = 5
        num, denom = get_rational_approximation_one(a, max_denom=max_denom)
        self.assertEqual(Fraction(num, denom), Fraction.from_float(a).limit_denominator(max_denom))

    def test_get_rational_approximation_one_0_to_1_6(self):
        a = 4.4
        max_denom = 100
        num, denom = get_rational_approximation_one(a, max_denom=max_denom)
        self.assertEqual(Fraction(num, denom), Fraction.from_float(a).limit_denominator(max_denom))

    def test_get_rational_approximation_one_0_to_1_7(self):
        a = 2424242.424242342409
        max_denom = 10000
        num, denom = get_rational_approximation_one(a, max_denom=max_denom)
        self.assertEqual(Fraction(num, denom), Fraction.from_float(a).limit_denominator(max_denom))

    def test_get_sos_odd_0(self):
        polynomial = poly('x')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'One of the variables in the polynomial has odd degree. Not a sum of squares.')
        self.assertEqual(expand(sos.as_poly()), nan)

    def test_get_sos_odd_1(self):
        polynomial = poly('abc')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'One of the variables in the polynomial has odd degree. Not a sum of squares.')
        self.assertEqual(expand(sos.as_poly()), nan)

    def test_get_sos_odd_2(self):
        polynomial = poly('x**3 + 1')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'One of the variables in the polynomial has odd degree. Not a sum of squares.')
        self.assertEqual(sos, nan)

    def test_get_sos_1(self):
        polynomial = poly('x**2 + 2*x + 1')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())

    def test_get_sos_am_gm_2(self):
        polynomial = poly('(x**2 + y**2)**2 - 2*x**2*y**2')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())

    def test_get_sos_am_gm_3(self):
        polynomial = poly('(x**2 + y**2 + z**2)**3 - 3*x**2*y**2*z**2')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())

    def test_get_sos_2(self):
        polynomial = poly('(x**2 + y**2 + z**2)**3 - 3*x**2*y**2*z**2')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())

    def test_get_sos_3(self):
        polynomial = poly('x**10 - x**6 - x**4 + 1')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())

    def test_get_sos_4(self):
        polynomial = poly('(x + 1)**2*(y-1)**2')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())

    def test_get_sos_5(self):
        polynomial = poly('(x**2 - 1)**2 * (x**2 + 1) * (x**4 + x**2 + 1)')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())

    def test_get_sos_le_van_barel_0(self):
        polynomial = poly('1 + x**2 + x**4 + x**6')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())

    def test_get_sos_motzkin(self):
        polynomial = poly('1 + x**2*y**2*(x**2 + y**2) - 3*x**2*y**2')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(simplify(sos).as_expr(), polynomial.as_expr())

    def test_get_sos_motzkin_2(self):
        polynomial = poly('x**4*y**2 + x**2*y**4 - 3*x**2*y**2*z**2 + z**6')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(simplify(sos).as_expr(), polynomial.as_expr())

    def test_get_sos_motzkin_with_denominator_multiplied_on_left(self):
        polynomial = poly('(1+x**2 + y**2)*(1 + x**2*y**2*(x**2 + y**2) - 3*x**2*y**2)')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())

    # def test_get_sos_robinson(self):
    #     polynomial = poly(
    #         'x**6 + y**6 + z**6 - (x**4*y**2 + x**2*y**4 + x**4*z**2 + x**2*z**4 + y**4*z**2 + y**2*z**4) + 3*x**2*y**2*z**2')
    #     status, sos = get_sos(polynomial)
    #     self.assertEqual(status, 'Exact SOS decomposition found.')
    #     self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())
    #
    # def test_get_sos_schmuedgen(self):
    #     polynomial = poly(
    #         '200*(x**3 - 4*x*z**2)**2 + 200*(y**3 - 4*y*z**2)**2 + (y**2 - x**2)*x*(x + 2*z)*(x**2 - 2*x*z + 2*y**2 - 8*z**2)')
    #     status, sos = get_sos(polynomial)
    #     self.assertEqual(status, 'Exact SOS decomposition found.')
    #     self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())
    #
    # def test_get_sos_scheiderer(self):
    #     polynomial = poly('x**4 + x*y**3 + y**4 - 3*x**2*y*z - 4*x*y**2*z + 2*x**2*z**2 + x*z**3 + y*z**3 + z**4')
    #     status, sos = get_sos(polynomial)
    #     self.assertEqual(status, 'Exact SOS decomposition found.')
    #     self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())
    #
    # def test_get_sos_harris(self):
    #     a, b, c, d, e = 16, -36, 20, 57, -38
    #     polynomial = poly(
    #         'a*( x**10 + y**10 + z**10)+ b*( x**8* y**2 + x**2* y**8 + x**8* z**2 + x**2* z**8 + y**8* z**2 + y**2* z**8 ) + c*( x**6* y**4 + x**4* y**6 + x**6* z**4 + x**4* z**6 + y**6* z**4 + y**4* z**6 ) + d*( x**6* y**2* z**2 + x**2* y**6* z**2 + x**2* y**2* z**6) + e*( x**4* y**4* z**2 + x**4* y**2* z**4 + x**2* y**4* z**4)')
    #     status, sos = get_sos(polynomial)
    #     self.assertEqual(status, 'Exact SOS decomposition found.')
    #     self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())
    #
    # def test_get_sos_lax_lax(self):
    #     polynomial = poly('(x-y)*(x-z)*(x-w)*x+(y-x)*(y-z)*(y-w)*y+(z-x)*(z-y)*(z-w)*z+(w-x)*(w-y)*(w-z)*w+x*y*z*w')
    #     status, sos = get_sos(polynomial)
    #     self.assertEqual(status, 'Exact SOS decomposition found.')
    #     self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())

    def test_get_sos_choi_lam(self):
        polynomial = poly('1 + y**2*z**2 - 4*x*y*z + x**2*z**2 + x**2*y**2')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(simplify(sos).as_expr(), expand(polynomial).as_expr())

    # def test_get_sos_choi_lam_2(self):
    #     polynomial = poly('x**2*y**2 + y**2*z**2 + x**2*z**2 + w**4 - 4*x*y*z*w')
    #     status, sos = get_sos(polynomial)
    #     self.assertEqual(status, 'Exact SOS decomposition found.')
    #     self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())

    def test_get_sos_choi_lam_2(self):
        polynomial = poly('y**4 + x**2 - 3*x**2*y**2 + x**4*y**2')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(simplify(sos).as_expr(), polynomial.as_expr())

    def test_get_sos_le_van_barel_1(self):
        polynomial = poly(
            '2*w**2 - 2*z*w + 8*z**2 - 2*y*z + y**2 - 2*y**2*w + y**4 - 4*x*y*z - 4*x**2*z + x**2*y**2 + 2*x**4')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())

    def mildorf_titu_andreescu(self):
        polynomial = poly('a**4*c**2 + b**4*a**2 + c**4*b**2 - a**3*b**2*c - b**3*c**2*a - c**3*a**2*b')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())

    # def mildorf_1(self):
    #     polynomial = poly(
    #         '(a**4*b**2 + b**4*c**2 + c**4*a**2)*(b**4*a**2 + c**4*b**2 + a**4*c**2) - 9*a**4*b**4*c**4')
    #     status, sos = get_sos(polynomial)
    #     self.assertEqual(status, 'Exact SOS decomposition found.')
    #     self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())
    #
    # def mildorf_2(self):
    #     polynomial = poly('a**4*b**2 + b**4*a**2 + 1 - (a**
    #     3*b**2 + b**3*a**2 + ab)')
    #     status, sos = get_sos(polynomial)
    #     self.assertEqual(status, 'Exact SOS decomposition found.')
    #     self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr()
    #     )
    #
    # def mildorf_5(self):
    #     polynomial = poly(
    #         '(a**2+b**2+c**2+d**2)*b**2*c**2*d**2 +(a**2+b**2+c**2+d**2)*a**2*c**2*d**2 + 4*(a**2+b**2+c**2+d**2)*a**2*b**2*d**2 + 16*(a**2+b**2+c**2+d**2)*a**2*b**2*c**2 - 64*a**2*b**2*c**2*d**2')
    #     status, sos = get_sos(polynomial)
    #     self.assertEqual(status, 'Exact SOS decomposition found.')
    #     self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())
    #
    # def mildorf_8(self):
    #     polynomial = poly('a**6 + b**6 + (1-a**2-b**2)**3 + 6*a**2*b**2*(1-a**2-b**2) - 1/4')
    #     status, sos = get_sos(polynomial)
    #     self.assertEqual(status, 'Exact SOS decomposition found.')
    #     self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())
    #
    # def test_get_sos_le_van_barel_2(self):
    #     polynomial = poly('x**4*y**2 + y**4*z**2 + x**2*z**4 -3*x**2*y**2*z**2 + z**8')
    #     status, sos = get_sos(polynomial)
    #     self.assertEqual(status, 'Exact SOS decomposition found.')
    #     self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())
    #
    # def test_get_sos_imo(self):
    #     # source: https://artofproblemsolving.com/wiki/index.php/2004_USAMO_Problems/Problem_5
    #     polynomial = poly('(a**10 - a**4 + 3)*(b**10 - b**4 + 3)*(c**10 - c**4 + 3) - (a**2 + b**2 + c**2)**3')
    #     status, sos = get_sos(polynomial)
    #     self.assertEqual(status, 'Exact SOS decomposition found.')
    #     self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())
    #
    # def test_get_sos_am_gm_4(self):
    #     polynomial = poly('(x**2 + y**2 + z**2+ t**2)**4 - 4*x**2*y**2*z**2*t**2')
    #     status, sos = get_sos(polynomial)
    #     self.assertEqual(status, 'Exact SOS decomposition found.')
    #     self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())
