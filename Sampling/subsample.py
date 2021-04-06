
import numpy as np
from dft import dft
from idft import idft

class subsample(object):
    
    def __init__(self, x, T_s, tau):
        
        self.x = x
        self.T_s = T_s
        self.f_s = np.int(1/T_s)
        self.tau = tau
        self.f_ss = np.int(1/tau)
        self.N = len(x)
        
    # Without prefiltering
    def solve(self):

        step = np.int(self.tau/self.T_s)
        x_s = self.x[0::step]
        x_delta = np.zeros(self.N)
        x_delta[0::step] = x_s
        
        return x_s, x_delta
        
    # With prefiltering
    def solve2(self):
        
        # Low-pass filtering
        fmax = self.f_ss/2
        DFT = dft(self.x, self.f_s)
        [_, _, f_c, X_c] = DFT.solve3()
        index_min  = np.min(np.where(f_c >= -fmax)[0])
        index_max = np.max(np.where(f_c <= fmax)[0])
        X_band = np.zeros(self.N)
        X_band[index_min:index_max] = X_c[index_min:index_max]
        X_band = np.roll(X_band, np.int(np.floor(self.N/2+1)))
        iDFT = idft(X_band, self.f_s, self.N)
        x_band, t = iDFT.solve_ifft()
        
        # Subsample
        step = np.int(self.tau/self.T_s)
        x_s = x_band[0::step]
        x_delta = np.zeros(self.N)
        x_delta[0::step] = x_s
        
        return x_s, x_delta
        
        
        