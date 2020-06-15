"""
Probability stuff.
"""

import numpy as np
from scipy.special import comb, betainc, beta
from scipy.optimize import minimize
from functools import partial


def p(k, F):
    g = 2 * F * F + 1
    a = g / (1 + g)

    return np.power(g, -.5 * k) * np.power(a, k) / (k * beta(.5 * k, .5 * k + 1)) + 1 - betainc(.5 * k, .5 * k + 1, a)


def b(n, k, CR):
    return comb(n, k, exact=True) * np.power(CR, k) * np.power(1 - CR, n - k)


def p_cr(n, F, CR):
    aux = 0
    CR = CR[0]

    for k in range(1, n + 1):
        aux += p(k, F) * b(n, k, CR)

    return -aux


def get_opt_cr(n, F):
    return minimize(partial(p_cr, n, F), x0=np.array([1./n]), method='Nelder-Mead', options={'xatol': 1e-10}).x[0]