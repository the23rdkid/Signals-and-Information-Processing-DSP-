# ESE 224 --- Signal and Information Processing
#
# Spring 2021
#
# Discrete sines, cosines and complex exponentials
#
# Question 1.1: Defining the complex exponential class

import numpy as np
import math
import cmath

class sqpulse():
    """
    sqpulse Generates a square spulse
    solve() generates a square pulse vector x of length N
    """
    def __init__(self, T, T0, fs):
        """
        :param T: the duration
        :param T0: nonzero length
        :param fs: the sampling frequency
        """
        self.T = T
        self.T0 = T0
        self.fs = fs
                
    def solve(self):
        """
        :return: [x,N]
        :x: the square pulse vector x
        :N: The sample duration
        """
        Ts = 1/self.fs
        N = math.floor(self.T/Ts)
        M = math.floor(self.T0/Ts)
        x = np.zeros(N)
        
        for i in range(M):
            x[i] = 1/np.sqrt(M)
        
        return x, N

class tripulse():
    """
    sqpulse Generates a triangular spulse
    solve() generates a triangular pulse vector x of length N
    """
    def __init__(self, T, T0, fs):
        """
        :param T: the duration
        :param T0: nonzero length
        :param fs: the sampling frequency
        """
        self.T = T
        self.T0 = T0
        self.fs = fs
                
    def solve(self):
        """
        :return: [x,N]
        :x: the triangular pulse vector x
        :N: The sample duration
        """
        Ts = 1/self.fs
        N = math.floor(self.T/Ts)
        M = math.floor(self.T0/Ts)
        x = np.zeros(N)
        
        for i in range(np.int(M/2)):
            x[i] = i
            x[M-i-1] = i
            
        energy = np.linalg.norm(x)
            
        for i in range(M):
            x[i] = x[i]/energy        
        
        return x, N
    
    
class sqwave():
    """
    sqpulse Generates a square spulse
    solve() generates a square pulse vector x of length N
    """
    def __init__(self, T, f0, fs):
        """
        :param T: the duration
        :param T0: nonzero length
        :param fs: the sampling frequency
        """
        self.T = T
        self.f0 = f0
        self.fs = fs
        self.N = T*fs
                
    def solve(self):
        """
        :return: [x,N]
        :x: the square pulse vector x
        :N: The sample duration
        """
        n = np.arange(self.N)
        x = np.sign(np.cos(2*cmath.pi*self.f0/self.fs*n))
        
        return x, self.N
