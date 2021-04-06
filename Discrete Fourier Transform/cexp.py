# ESE 224 --- Signal and Information Processing
#
# Spring 2021
#
# Discrete sines, cosines and complex exponentials
#
# Question 1.1: Creates complex exponential object


import numpy as np
import math
import cmath
import warnings

class cexp(object):
    """
    Creates a discrete complex exponential of discrete frequency k and duration N.
    Arguments:
        k: discrete frequency
        N: duration of the complex exponential
    """

    def __init__(self, k, N):
        assert N > 0, "N should be a nonnegative scalar"
        self.k = k #frequency
        self.N = N #length

        # Vector containing elements of time indexes
        self.n = np.arange(N)

        # Vector containing elements of the complex exponential
        self.exp_kN = np.exp(2j*cmath.pi*self.k*self.n / self.N)
        self.exp_kN *= 1 / (np.sqrt(N))

        # Vector containing real elements of the complex exponential
        self.exp_kN_real = self.exp_kN.real

        # Vector containing imaginary elements of the complex exponential
        self.exp_kN_imag = self.exp_kN.imag


def cexpt(f, T, fs):
    """
    This function generates a (sampled) continuous-time complex exponential.
    Arguments:
        f: frequency of the complex exponential
        T: duration
        fs: sampling frequency
    Returns
        t: vector of time indexes
        x: vector of samples of a complex exponential of frequency f and duration T
        N: number of samples
    """
    assert T > 0, "Duration of the signal cannot be negative."
    assert fs != 0, "Sampling frequency cannot be zero"

    if fs < 0:
        warnings.warn("Sampling frequency is negative. Using absolute value instead.")
        fs = - fs

    if f < 0:
        warnings.warn("Complex exponential frequency is negative. Using absolute value instead.")
        f = -f

    # Duration of the discrete signal
    N = math.floor(T * fs)
    t = np.linspace(0, (N - 1) / fs, N)
    # Discrete frequency
    k = N * f / fs
    # Complex exponential
    cpx_exp = cexp(k, N)
    x = cpx_exp.exp_kN
    x = np.sqrt(N) * x

    return t, x, N
