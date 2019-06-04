"""Miscellaneous utility functions and classes"""

from numpy import vectorize, float64


def kronecker_delta(i, j):
    return 1 if i == j else 0


def _round_by_tol(x, tol):
    return x if x > tol else 0


round_by_tol = vectorize(_round_by_tol, otypes=[float64])
