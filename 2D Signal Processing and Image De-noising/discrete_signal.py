# ESE 224 --- Signal and Information Processing
#
# Spring 2021

import numpy as np
import cmath
from scipy import signal
    
    
class inner_prod_2D():
    """
    2—D inner-product
    """
    def __init__(self, x, y):
        """
        x,y: two 2-D signals
        """
        self.x=x
        self.y=y
        self.N=np.shape(x)[0]

    def solve(self):
        """
        \\\\\ METHOD: Compute the inner product
        """
        prod = 0        
        for i in range(self.N):
            for j in range(self.N):
                prod = prod + self.x[i,j] * np.conj(self.y[i,j])
            
        return prod
    

class complex_exp_2D():
    """
    Discrete Complex Exponentials.
    """
    def __init__(self, k, l, N):

        self.k=k
        self.l=l
        self.N=N

    def solve(self):
        """
        \\\\\ METHOD: output discrete complex exponentials
        """
        e_kl=np.zeros([self.N, self.N], dtype=np.complex)
        for i in range(self.N):
            for j in range(self.N):
                e_kl[i,j] = 1/np.sqrt(self.N*self.N)*np.exp(-1j*2*cmath.pi*(self.k*i/self.N+self.l*j/self.N))
            
        return e_kl, np.real(e_kl), np.imag(e_kl)
    
    

class sq_pulse_2D():
    """
    Unit Energy 2-D Square Pulse.
    """
    def __init__(self, N, L):

        self.N=N
        self.L=L
        self.samples=N*N

    def solve(self):
        """
        \\\\\ METHOD: Output 2-D Square Pulse
        """
        sq_pulse=np.zeros([self.N, self.N], dtype=np.float)
        for i in range(self.L):
            for j in range(self.L):
                sq_pulse[i,j] = 1/self.L/self.L
            
        return sq_pulse, self.samples
    
    
    

class Gaussian_2D():
    """
    Two-Dimensional Gaussian Signals.
    """
    def __init__(self, N, mu, sigma):
        
        self.N=N
        self.mu=mu
        self.samples=N*N
        self.sigma=sigma

    def solve(self):
        """
        \\\\\ METHOD: output Two-Dimensional Gaussian Signals
        """
        gaussian=np.zeros([self.N, self.N], dtype=np.float)
        for i in range(self.N):
            for j in range(self.N):
                gaussian[i,j] = np.exp(-((i-self.mu)*(i-self.mu)+(j-self.mu)*(j-self.mu))/2/self.sigma/self.sigma)
            
        return gaussian, self.samples
    
    

class DFT_2D():
    """
    2-D DFT
    """
    def __init__(self, x):
        """
        input time-domain signal x
        """
        self.x=x
        self.M=np.shape(x)[0]
        self.N=np.shape(x)[1]

    def solve(self):
        """
        \\\\\ METHOD: Compute DFT of x
        """
        X=np.zeros([self.M, self.N], dtype=np.complex)
        for m in range(self.M):
            for n in range(self.N):
                for i in range(self.M):
                    for j in range(self.N):
                        X[m,n] = X[m,n] + self.x[i,j]/np.sqrt(self.M*self.N)*np.exp(-1j*2*cmath.pi*(m*i/self.M+n*j/self.N))
            
        return X
    
    
    
class iDFT_2D():
    """
    2-D iDFT
    """
    def __init__(self, X):
        """
        Input DFT X
        """
        self.X=X
        self.M=np.shape(X)[0]
        self.N=np.shape(X)[1]

    def solve1(self):
        """
        \\\\\ METHOD: Compute the iDFT of X with N^2 coefficients
        """
        x=np.zeros([self.M, self.N], dtype=np.complex)
        for m in range(self.M):
            for n in range(self.N):
                for i in range(self.M):
                    for j in range(self.N):
                        x[m,n] = x[m,n] + self.X[i,j]/np.sqrt(self.M*self.N)*np.exp(1j*2*cmath.pi*(m*i/self.M+n*j/self.N))
            
        return x
    
    def solve2(self):
        """
        \\\\\ METHOD: Compute the iDFT of X with N^2/2 coefficients
        """
        x=np.zeros([self.M, self.N], dtype=np.complex)
        for m in range(self.M):
            for n in range(self.N):              
                for i in range(int(self.M/2)+1):
                    for j in range(self.N):
                        x[m,n] = x[m,n] + self.X[i,j]/np.sqrt(self.M*self.N)*np.exp(1j*2*cmath.pi*(m*i/self.M+n*j/self.N))
                        if i != 0:
                            x[m,n] = x[m,n] + np.conj(self.X[i,j])/np.sqrt(self.M*self.N)*np.exp(1j*2*cmath.pi*(-m*i/self.M-n*j/self.N))

        return x
    
    

    

class Convolution_2D():
    """
    2-D convolution
    """
    def __init__(self, x, y):
        """
        input signal x
        filter y
        """
        self.x=x
        self.y=y

    def solve(self):
        """
        \\\\\ METHOD: Compute 2-D convolution
        """
        filtered_signal = signal.convolve2d(self.x, self.y, boundary='symm', mode='same')
            
        return filtered_signal
