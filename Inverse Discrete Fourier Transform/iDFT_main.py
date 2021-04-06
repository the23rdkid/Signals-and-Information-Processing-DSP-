# ESE 224 --- Signal and Information Processing
#
# Spring 2021
#
# Inverse Discrete Fourier Transform

###############################################################################
############################# I M P O R T I N G ###############################
###############################################################################

# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
import cmath
from scipy.io.wavfile import write
import sounddevice as sd

# Local files
import discrete_signal

###############################################################################
############################ CLASSES ##########################################
###############################################################################    

class dft_K_q16():
    """
    idft Inverse Discrete Fourier transform.
    """
    def __init__(self, x, fs, K):
        """
        :param X: Input DFT X
        :param fs: Input integer fs contains the sample frequency
        :param K: Input positive integer that determines the number of coeffients
        used to calculate the iDFT.
        """
        self.x=x
        self.fs=fs
        self.N=len(self.x)
        self.K=K

    def solve(self):
        """
        \\\\\ METHOD: Compute the iDFT with truncated K coefficients of largest energy
        :fk the real frequencies
        """
        X=np.zeros(self.N, dtype=np.complex)
        E=np.zeros(self.N)
        X_K=np.zeros(self.K, dtype=np.complex)
        index=np.zeros(self.K)
        
        for k in range(self.N):
            for n in range(self.N):
                X[k] = X[k]+1/np.sqrt(self.N)*self.x[n]*np.exp(-1j*2*cmath.pi*k*n/self.N)
        fk= np.arange(self.N)/self.N*self.fs
                
        for k in range(self.N):
            E[k]=abs(X[k])
        index_temp=np.argsort(-E)        
        index=index_temp[0:self.K]
        index = index[:,np.newaxis]
        X_K=X[index]
        X_K=np.concatenate((X_K,index),axis=1)
            
        return X_K, fk
    
    

class dft():
    """
    dft Discrete Fourier transform.
    solve1,solve2,solve3 are the discrete Fourier transform (DFT) of
    vector x. There are three different ways of obtaining the DFT.
    The most efficient way of obtaining the dft is method 3, but 1
    and 2 are included for educational purposes.
    X is a vector that contains the DFT coefficients and f is a
    vector that contains the real frequencies determined by sampling
    frequency fs. f are the frequencies starting at f=0 and X are the
    corresponding frequency components. f_c is a vector containing the
    frequencies such that f_c=0 is at the center, and X_c contains the
    frequency components corresponding to f_c. In essence, f_c and X_c
    are the centered counterparts of f and X. Frequency 0 is always
    present.
    If a parameter K is given, solve1,solve2, solve3 computes the DFT with
    only K coefficients. Recall that this periodizes signal x with period K.
    If the length of the signal x is less than K, then the signal will be
    padded with zeros. If the length of the signal x is greater than K,
    then there will be aliasing occured from periodizing singal x with
    period K. f_c and X_c are the centered counterparts of f and X
    """
    def __init__(self, x, fs, K=None):
        """
        :param x: Input vector x contains the discrete signal
        :param fs: Input integer fs contains the sample frequency
        :param K: Input positive integer that determines the number of coeffients
        used to calculate the DFT. If K is not provided, K=length(x).
        """
    # START: exceotion handling
        if (type(fs) != int) or (fs<=0):
            raise NameError('The frequency fs should be a positive integer.')
        if not isinstance(x, np. ndarray):
            raise NameError('The input signal x must be a numpy array.')
        if isinstance(x, np. ndarray):
            if x.ndim!=1:
                raise NameError('The input signal x must be a numpy vector array.')
        self.x=x
        self.fs=fs
        self.N=len(x)
        if K == None:
            K = len(self.x)
        # START: exception handling
        if (type(K) != int) or (K <= 0) or (K < 0):
            raise NameError('K should be a positive integer.')
        self.K=K
        self.f=np.arange(self.N)*self.fs/self.N # (0:K-1) just creates a vector from 0 to K by steps of 1.
        self.f_c=np.arange(-np.ceil(self.N/2)+1,np.floor(self.N/2)+1)*self.fs/self.N
        # This accounts for the frequencies
        # centered at zero. I want to be guaranteed that k=0 is always a
        # possible k. Then, I also have to account for both even and odd choices
        # of K, and that's why the floor() function appears to round down the
        # numbers.
    def changeK(self,K):
        """
        :param K: Input positive integer that determines the number of coeffients
        used to calculate the DFT. This function changes the attribute K of the class.
        """
        if (type(K) != int) or (K <= 0) or (K <  0):
            raise NameError('K should be a positive integer.')
        old_K=self.K
        self.K=K
        self.f=np.arange(self.K)*self.fs/self.K # (0:K-1) just creates a vector from 0 to K by steps of 1.
        self.f_c=np.arange(-np.ceil(K/2)+1,np.floor(self.K/2)+1)*self.fs/self.K
        # This accounts for the frequencies
        # centered at zero. I want to be guaranteed that k=0 is always a
        # possible k. Then, I also have to account for both even and odd choices
        # of K, and that's why the floor() function appears to round down the
        # numbers.
        print('The value of K was succefully change from %d to %d'%(old_K,self.K))
        pass

    def solve1(self):
        """
        \\\\\ METHOD 1: For loops
        By definition of DFT (eq. 1, lab 2) we have that
        X(k) = 1/sqrt(N) * sum_{n=0 to N-1} x(n) exp(-j 2 pi k n / N)
        where n is the discrete time index, k is the discrete time frequency and N is
        the length of the time signal x. Observe that this gives the DFT coefficient
        for a single coefficient k. We are trying to look for all coefficients
        k=0,1,...,K.
        This means that, for each value of k=0,1,...,K we will need to compute N
        multiplications x(n)*exp(-j 2 pi k n / N), for n=0,1,...,N-1, and sum the
        result.
        First thing we need is to create the variable X that we will output. This will
        be a vector of length K.
        """

        X=np.zeros(self.K,dtype=np.complex_)
        for k in range (self.K) :# For each time index k=0,1,...,K;
            for n in range (self.N):  # For each frequency n=0,1,...,N-1:
                X[k]=X[k]+1/np.sqrt(self.N)*self.x[n]*np.exp(-1j*2*cmath.pi*k*n/self.K)

        # Obs: in the case we have K different from N, then the
   		# signal will be periodized with period K. That is why the
        # exponential is divided by K instead of N.
        X_c=np.roll(X,np.int(np.ceil(self.K/2)-1)) # Circularly shift X to get it centered in f_c==0
        return [self.f,X,self.f_c,X_c]

    def solve2(self):
        """
        \\\\ METHOD 2: Matrix form
        Using for loops in python is rather expensive. Numpy is already optimized to
        work with vectors and matrices, so it's a good idea to take advantage of this,
        together with the built-in elementwise operators.
        Observe that we want a vector X=[X(0) X(1) ... X(K-1)]'=[X(k), k=0,...,K-1]'
        and that each element of the vector is computed as before:
        X(k) = 1/sqrt(N) * sum_{n=0 to N-1} x(n) exp(-j 2 pi k n / N)
        But, as mentioned in (eq. 2, lab. 2) this is nothing more than the inner
        product of x=[x(0) ... x(N-1)]'=[x(n), n=0,...,N-1]' with the complex
        exponential of frequency K and length N ekN=[ekN(0) ... ekN(N-1)]'=[ekN(n),
        n=0,...,N-1]', so that X(0)=<x,e0N>=e0N'*x, X(1)=<x,e1N>=e1N*x', ...,
        X(K-1)=<x,e(K-1)N>=e(K-1)N'*x. So we see that we have a bunch of vectors
        {ekN, k=0,...,K-1} that always multiply the same vector x. We can achieve
        this by creating a matrix where each of the ekN is a row, and then get
        the full vector X by multiplying this matrix by x. We will denote the
        matrix as WKN for future reference:
        WKN=[e0N'; e1N'; ... ; e(K-1)N'];
        Finally, observe that as we move from row to row, k grows from 0 to K-1.
        And as we move from column to column, it is n that grows from 0 to N-1.
        Also, observe that ekN is a function of (k,n) given by exp(-j 2 pi k n /K).
        So if we create a matrices of indices (k,n) then we can directly apply
        the exponential by making use of the elementwise nature of this operation
        in Numpy.
        """
        matrix_k=np.transpose(np.tile(np.arange(self.K),(self.N,1)))
        matrix_n=np.tile(np.transpose(np.arange(self.N)),(self.K,1))
        indices=np.multiply(matrix_k,matrix_n)
        WKN=1/np.sqrt(self.N)*np.exp(-1j*2*np.pi*indices/self.K)
        X=WKN@self.x

        X_c=np.roll(X,np.int(np.ceil(self.K/2)-1)) # Circularly shift X to get it centered in f_c==0
        return [self.f,X,self.f_c,X_c]

    def solve3(self):
        """
        \\\\\ METHOD 3: Built-in fft() function
        Even though the matrix form is fast, it is still not fast enough for large
        signals x. For that, it is better to use the built in fft() function which is
        the optimal way to compute a dft. Besides, it is really easy to code.
        """
        X=np.fft.fft(self.x,self.N)/np.sqrt(self.N);
        # \\\\\ CENTER FFT.
        X_c=np.roll(X,np.int(np.ceil(self.N/2-1))) # Circularly shift X to get it centered in f_c==0
        return [self.f,X,self.f_c,X_c]
    
    def solve4(self):
        """
        \\\\\ METHOD 3: Built-in fft() function
        Even though the matrix form is fast, it is still not fast enough for large
        signals x. For that, it is better to use the built in fft() function which is
        the optimal way to compute a dft. Besides, it is really easy to code.
        """
        X=np.fft.fft(self.x,self.N)/np.sqrt(self.N);
        # \\\\\ CENTER FFT.
        X_c=np.roll(X,np.int(np.ceil(self.N/2-1))) # Circularly shift X to get it centered in f_c==0
        
        X_K = X[0:(self.K+1)]
        
        return [self.f,X_K,self.f_c,X_c]

class idft_p11():
    """
    idft Inverse Discrete Fourier transform.
    """
    def __init__(self, X, fs):
        """
        :param X: Input DFT X
        :param fs: Input integer fs contains the sample frequency
        """
        self.X=X
        self.fs=fs
        self.N=2*(len(self.X)-1)

    def solve(self):
        """
        \\\\\ METHOD: Compute the iDFT with truncated N/2+1 coefficients
        :return iDFT x of duration N from partial DFT X, i.e., X[0], ..., X[N/2]
        :t the real time vector of size N
        """
        x=np.zeros(self.N)
        
        for n in range(self.N):
            x[n] = 1/np.sqrt(self.N)*self.X[0]*np.exp(1j*2*cmath.pi*0*n/self.N)
            for k in range(1,int(self.N/2)): 
                x[n] = x[n] + 1/np.sqrt(self.N)*self.X[k]*np.exp(1j*2*cmath.pi*k*n/self.N)
                x[n] = x[n] + 1/np.sqrt(self.N)*np.conj(self.X[k])*np.exp(-1j*2*cmath.pi*k*n/self.N)
            x[n] = x[n] + 1/np.sqrt(self.N)*self.X[int(self.N/2)]*np.exp(1j*2*cmath.pi*(int(self.N/2))*n/self.N)
                
        Ts= 1/self.fs
        Treal= np.arange(self.N)*Ts

        return x, Treal 

class idft():
    """
    idft Inverse Discrete Fourier transform.
    """
    def __init__(self, X, fs, N, K=None):
        """
        :param X: Input DFT X
        :param fs: Input integer fs contains the sample frequency
        :param N: The number of total signal samples N
        :param K: Input positive integer that determines the number of coeffients
        used to calculate the iDFT.
        """
        self.X=X
        self.fs=fs
        self.N=N 
        self.K=K
        if self.K==None:
            self.K=int(len(X)/2)-1

    def solve_K(self):
        """
        \\\\\ METHOD: Compute the iDFT with truncated K coefficients (Due to computation complexity, we will not use it in Section 2)
        :return iDFT x of duration N from partial DFT X, i.e., X[0], ..., X[K] with K < N/2
        :Treal the realt time vector of size N
        """
        x=np.zeros(self.N)
        
        for n in range(self.N):
            x[n] = 1/np.sqrt(self.N)*self.X[0]*np.exp(1j*2*cmath.pi*0*n/self.N)
            for k in range(1,self.K+1): 
                x[n] = x[n] + 1/np.sqrt(self.N)*self.X[k]*np.exp(1j*2*cmath.pi*k*n/self.N)
                x[n] = x[n] + 1/np.sqrt(self.N)*np.conj(self.X[k])*np.exp(-1j*2*cmath.pi*k*n/self.N)
                
        Ts= 1/self.fs
        Treal= np.arange(self.N)*Ts

        return x, Treal
    
    def solve_ifft(self):
        """
        \\\\\ METHOD: Compute the iDFT with provided function np.fft.ifft (Computationally efficient)
        :Treal the realt time vector of size N
        """
        x=np.fft.ifft(self.X,self.N)*np.sqrt(self.N)
                
        Ts= 1/self.fs
        Treal= np.arange(self.N)*Ts

        return x, Treal    
    
    
class Sginal_Recon_K_q18():
    """
    Signal reconstruction Question 1.7
    """
    def __init__(self, X_k, fk):
        """
        :param X: Input truncated DFT X_k
        :param fs: Input integer fs contains the sample frequency
        """
        self.X=X_k
        self.fk=fk
        self.N=len(fk)
        self.K=self.X.shape[0]

    def solve(self):
        """
        \\\\\ METHOD: Compute the iDFT with truncated K coefficients with largest energy
        """
        x=np.zeros(self.N)
        
        for n in range(self.N):
            for k in range(self.K):
                x[n] = x[n]+1/np.sqrt(self.N)*self.X[k,0]*np.exp(1j*2*cmath.pi*self.X[k,1]*n/self.N)
            
        return x   
    

###############################################################################
############################ QUESTIONS ########################################
############################################################################### 
        
    
def q_13(T, fs, T0, K):

    #Question 1.3
    
    sp = discrete_signal.sqpulse(T, T0, fs)    #generate square pulse signal
    x, N = sp.solve()
    DFT = dft(x,fs)      #compute the DFT
    [f,X,f_c,X_c] = DFT.solve3()    

    iDFT = idft(X, fs, N, K)      #compute the iDFT
    xhat_K, Treal = iDFT.solve_K()
    
    x_diff = x - xhat_K    #compute the energe difference
    energy_diff = np.linalg.norm(x_diff)*np.linalg.norm(x_diff)   
    print(energy_diff)
    
    # Plot
    fig, axs = plt.subplots(2)
    axs[0].grid()
    axs[1].grid()
    fig.suptitle('Original Signal and its DFT' )
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    axs[0].plot(Treal, x)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Signal')
    axs[1].plot(f_c, abs(X_c))
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('DFT')
    plt.show()    
    fig.savefig("square_DFT.png", bbox_inches='tight')
    
    plt.figure()
    plt.grid(True)
    plt.title('Reconstructed Signal')
    plt.plot(Treal, xhat_K)
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.savefig('Square_reconstructed.png')
    plt.show()
#    
def q_14(T, fs, T0, K):
    """
    Question 1.4
    
    """
    
    tp = discrete_signal.tripulse(T, T0, fs)    
    x, N = tp.solve()
    DFT = dft(x,fs)
    [f,X,f_c,X_c] = DFT.solve3()    

    iDFT = idft(X, fs, N, K)
    xhat_K, Treal = iDFT.solve_K()
    
    x_diff = x - xhat_K
    energy_diff = np.linalg.norm(x_diff)*np.linalg.norm(x_diff)
    print(energy_diff)
    
    # Plot
    fig, axs = plt.subplots(2)
    axs[0].grid()
    axs[1].grid()
    fig.suptitle('Original Signal and its DFT' )
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    axs[0].plot(Treal, x)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Signal')
    axs[1].plot(f_c, abs(X_c))
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('DFT')
    plt.show() 
    fig.savefig("triangular_DFT.png", bbox_inches='tight')
    
    plt.figure()
    plt.grid(True)
    plt.title('Reconstructed Signal')
    plt.plot(Treal, xhat_K)
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.savefig('Triangular_reconstructed.png')
    plt.show()


def q_18_example(T, fs, T0, K):
    """
    Question 1.8
    
    """
    
    sp = discrete_signal.sqpulse(T, T0, fs)    
    x, N = sp.solve()
    DFT_K = dft_K_q16(x,fs,K)
    X_K, fk = DFT_K.solve()
    
    iDFT = Sginal_Recon_K_q18(X_K, fk)
    xhat_K = iDFT.solve()
    
    Ts= 1/fs
    Treal= np.arange(N)*Ts
    
    # Plot
    fig, axs = plt.subplots(2)
    axs[0].grid()
    axs[1].grid()
    fig.suptitle('Original Signal and Reconstructed Signal' )
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    axs[0].plot(Treal, x)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Signal')
    axs[1].plot(Treal, xhat_K)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Signal')
    plt.show() 

def q_19(T, fs, f0, K):
    """
    Question 1.9
    
    """
    
    sp = discrete_signal.sqwave(T, f0, fs)    
    x, N = sp.solve()
    DFT = dft(x,fs)
    [f,X,f_c,X_c] = DFT.solve3()      
    
    DFT_K = dft_K_q16(x,fs,K)
    X_K, fk = DFT_K.solve()
    
    iDFT = Sginal_Recon_K_q18(X_K, fk)
    xhat_K = iDFT.solve()
    
    Ts= 1/fs
    Treal= np.arange(N)*Ts
    
    # Plot
    fig, axs = plt.subplots(2)
    axs[0].grid()
    axs[1].grid()
    fig.suptitle('Original Signal and its DFT' )
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    axs[0].plot(Treal, x)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Signal')
    axs[1].plot(f_c, abs(X_c))
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('DFT')
    plt.show()
    fig.savefig("Square_Wave_DFT.png", bbox_inches='tight')
    
    plt.figure()
    plt.grid(True)
    plt.title('Reconstructed Signal')
    plt.plot(Treal, xhat_K)
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.savefig('Square_Wave_reconstructed.png')
    plt.show()
    
    
    
class recordsound():
    """
    recordsound Record your voice for T time sampled at a frequency fs
    solve() generates a sampled signal of your voice and save it into a wav file
    """
    def __init__(self, T, fs):
        """
        :param T: the duration time
        :param fs: the sampling frequency
        """
        self.T = T
        self.fs = fs
                
    def solve(self):
        """
        :return: [x,N]
        :x: the triangular pulse vector x
        :N: The sample duration
        """
        print('start recording')
        voicerecording = sd.rec(int(self.T * self.fs), self.fs, 1)
        sd.wait()  # Wait until recording is finished
        print('end recording')
        write('myvoice.wav', self.fs, voicerecording)  # Save as WAV file 
        
        return voicerecording
    
def q_21(T, fs):
    """
    Question 2.2_first K DFT coefficients
    
    """

    myvoice = recordsound(T, fs)  
    x = myvoice.solve().reshape(T*fs)
    N = len(x)

    DFT = dft(x,fs)
    [freqs, X, f_c, X_c] = DFT.solve3()
    
    Ts= 1/fs
    Treal= np.arange(N)*Ts
    
    # Plot
    fig, axs = plt.subplots(2)
    axs[0].grid()
    axs[1].grid()
    fig.suptitle('Original Voice and its DFT' )
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    axs[0].plot(Treal, x)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Signal')
    axs[1].plot(f_c, X_c)
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('DFT')
    plt.show()     
    fig.savefig("Recorded_Voice_and_DFT.png", bbox_inches='tight')
    
def q_22_1(T, fs, gamma):
    """
    Question 2.2_first K DFT coefficients
    
    """

    myvoice = recordsound(T, fs)  
    x = myvoice.solve().reshape(T*fs)
    N = len(x)

    DFT = dft(x,fs)
    [freqs, X, f_c, X_c] = DFT.solve3()
    
    truncated_sample = int(N*gamma)
    X_truncated = np.zeros(N, dtype=np.complex)
    X_truncated[0:truncated_sample] = X[0:truncated_sample]
    
    iDFT = idft(X_truncated, fs, N)
    xhat_K, Treal = iDFT.solve_ifft()
    
    write('myvoice_truncated.wav', fs, xhat_K.real)
    
    # Plot
    fig, axs = plt.subplots(2)
    axs[0].grid()
    axs[1].grid()
    fig.suptitle('Original Voice and Reconstructed Voice' )
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    axs[0].plot(Treal, x)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Signal')
    axs[1].plot(Treal, xhat_K)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Signal')
    plt.show()     
    fig.savefig("Reconstructed_Voice.png", bbox_inches='tight')
    
def q_22_2(T, fs, gamma):
    """
    Question 2.2 _K DFT coefficients with largest energy
    """
    
    myvoice = recordsound(T, fs)  
    x = myvoice.solve().reshape(T*fs)
    N = len(x)

    DFT = dft(x,fs)
    [freqs, X, f_c, X_c] = DFT.solve3()
    
    truncated_sample = int(N*gamma/2)
    X_truncated = np.zeros(N, dtype=np.complex)
    E=np.zeros(N)
    for k in range(N):
        E[k]=abs(X[k])
    index_temp=np.argsort(-E)
    index=index_temp[0:truncated_sample]
    X_truncated[index]=X[index]
    
    iDFT = idft(X_truncated, fs, N)
    xhat_K, Treal = iDFT.solve_ifft()
    
    # Plot
    fig, axs = plt.subplots(2)
    axs[0].grid()
    axs[1].grid()
    fig.suptitle('Original Voice and Reconstructed Voice' )
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    axs[0].plot(Treal, x)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Signal')
    axs[1].plot(Treal, xhat_K)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Signal')
    plt.show()     
    fig.savefig("Reconstructed_Voice_Largest.png", bbox_inches='tight')
    
    
def q_23(T, fs, threshold):
    """
    Question 2.3
    
    """
    myvoice = recordsound(T, fs)  
    x = myvoice.solve().reshape(T*fs)
    N = len(x)

    DFT = dft(x,fs)
    [freqs, X, f_c, X_c] = DFT.solve3()    
    
    for k in range(N):
        E=abs(X[k])
        if E > threshold:
            X[k] = 0
    
    iDFT = idft(X, fs, N)
    xhat_K, Treal = iDFT.solve_ifft()
    
    # Plot
    fig, axs = plt.subplots(2)
    axs[0].grid()
    axs[1].grid()
    fig.suptitle('Original Signal and Reconstructed Signal' )
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    axs[0].plot(Treal, x)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Signal')
    axs[1].plot(Treal, xhat_K)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Signal')
    plt.show() 
    fig.savefig("Reconstructed_Voice_Masking.png", bbox_inches='tight')
    
    
def q_24(T, fs, gamma):
    """
    Question 2.4
    
    """
    
    myvoice = recordsound(T, fs)  
    x = myvoice.solve().reshape(T*fs)
    N = len(x)
    chunks = T*10
    chunk_sample = 0.1*fs
    x_recon = []
    
    for i in range(chunks):        
        current_chunk = x[int(i*chunk_sample):int(i*chunk_sample+chunk_sample)]
        n = len(current_chunk)
        DFT_chunk = dft(current_chunk, fs)
        [freqs_chunk, X_chunk, f_c_chunk, X_c_chunk] = DFT_chunk.solve3()
        K = int(n*gamma)
        X_chunk_truncated = np.zeros(n,dtype=np.complex)
        X_chunk_truncated[0:K] = X_chunk[0:K]
        iDFT = idft(X_chunk_truncated, fs, n)
        x_idft, Treal = iDFT.solve_ifft() 
        x_recon = np.concatenate([x_recon, x_idft])
    Ts= 1/fs
    Treal= np.arange(N)*Ts
        
    # Plot
    fig, axs = plt.subplots(2)
    axs[0].grid()
    axs[1].grid()
    fig.suptitle('Original Signal and Reconstructed Signal' )
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    axs[0].plot(Treal, x)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Signal')
    axs[1].plot(Treal, x_recon)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Signal')
    plt.show() 
    fig.savefig("MP3_compressor.png", bbox_inches='tight')

###############################################################################
################################## M A I N ####################################
###############################################################################

if __name__ == '__main__':
    
#    q_21(5, 20000)
    
    q_23(5, 20000, 0.25)
