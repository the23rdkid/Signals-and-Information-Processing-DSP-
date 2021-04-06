
import numpy as np
from dft import dft
from idft import idft

class reconstruct():
    
    def __init__(self, x_s, T_s, tau):
        
        self.x_s = x_s
        self.T_s = T_s
        self.f_s = np.int(1/T_s)
        self.tau = tau
        self.nu = np.int(1/tau)
        self.N = len(x_s)*np.int(tau/T_s)
        
    def solve(self):
        
        x = np.zeros(self.N)
        step = np.int(self.tau/self.T_s)
        x[0::step] = self.x_s
        DFT_obj = dft(x,self.f_s)
        [_,_,f_c,X_c] = DFT_obj.solve3()
        fmax = self.nu/2
        index_min  = np.min(np.where(f_c >= -fmax)[0])
        index_max = np.max(np.where(f_c <= fmax)[0])
        X_band = np.zeros(self.N)
        X_band[index_min:index_max] = step*X_c[index_min:index_max]
        X_band = np.roll(X_band, np.int(np.floor(self.N/2+1)))
        iDFT = idft(X_band, self.f_s, self.N)
        x_band, t = iDFT.solve_ifft()
        
        return x_band