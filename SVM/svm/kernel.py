import numpy.linalg as la
import numpy as np


class Kernel(object):
    """Check kernels here https://en.wikipedia.org/wiki/Support_vector_machine"""
    @staticmethod
    def linear():
        return lambda x, y: np.inner(x, y)

    @staticmethod
    def gaussian(gamma):
        return lambda x, y: \
            np.exp(-np.sqrt(np.linalg.norm(x - y) ** 2 / (2 * gamma ** 2)))

    def gaussian(gamma):
        def f(x, y):
            exp = -np.sqrt(la.norm(x - y) ** 2 / (2 * gamma ** 2))
            return np.exp(exp)
        return f

    def gaussian(x, y):
        gamma = 1.0
        return np.exp(-np.sum((x - y) ** 2) / (2 * gamma ** 2))

