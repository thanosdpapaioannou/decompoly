import numpy as np
from sympy import Matrix
import scipy.linalg

gram_mat_q = np.array([
    [4.997998, 0, -0.998999, 0, 0, 0, 0, -2.99700599],
    [0, 4.997998, 0, 0, 0, 0, -3, 0],
    [-0.998999, 0, 1, 0, 0, 0, 0, -0.998999],
    [0, 0, 0, 4.997998, 0, -3, 0, 0],
    [0, 0, 0, 0, 20.99401198, 0, 0, 0],
    [0, 0, 0, -3, 0, 4.997998, 0, 0],
    [0, -3, 0, 0, 0, 0, 4.997998, 0],
    [-2.99700599, 0, -0.998999, 0, 0, 0, 0, 4.997998]], dtype=np.float64)

# check symmetric and positive definite
np.allclose(gram_mat_q, gram_mat_q.T)
_chol = np.linalg.cholesky(gram_mat_q)

# sympy
L, U, p = Matrix(gram_mat_q).LUdecomposition()
# check LU decomposition
np.allclose(gram_mat_q, np.array((L * U).permuteBkwd(p), dtype=np.float64))

# scipy
p_sci, L_sci, U_sci = scipy.linalg.lu(gram_mat_q)
# check LU decomposition
np.allclose(gram_mat_q, p_sci.dot(L_sci.dot(U_sci)))

# SciPy and SymPy LU decompositions are distinct:
np.allclose(np.array(L, dtype=np.float64), np.array(L_sci, dtype=np.float64))
# ... because gram_mat_q is very near non-positive-definite, in fact, np.vectorize(round)(gram_mat_q) is not positive-definite!
gram_mat_q_round = np.vectorize(round)(gram_mat_q)
_chol = np.linalg.cholesky(gram_mat_q_round)
