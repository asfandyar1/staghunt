"""Miscellaneous utility functions and classes"""

import gc
from numpy import vectorize, float64
from torch import is_tensor


def kronecker_delta(i, j):
    return 1 if i == j else 0


def _round_by_tol(x, tol):
    return x if x > tol else 0


round_by_tol = vectorize(_round_by_tol, otypes=[float64])


def prod(x):
    """
    Computes the product of the elements of an iterable
    :param x: iterable
    :type x: iterable
    :return: product of the elements of x
    """
    ret = 1
    for item in x:
        ret = item * ret
    return ret


def memory_report():
    for o in gc.get_objects():
        if is_tensor(o):
            print(o.device, o.dtype, tuple(o.shape))


def new_var(var_name, t, agent):
    """
    Naming of agent variables xij, where i in {1,...,T} and j in {1,...,M}
    :param var_name: variable name x, u, d,...
    :param t: time index
    :param agent: agent index
    :return: string with the name of the variable
    """
    return var_name + str(t) + str(agent)
