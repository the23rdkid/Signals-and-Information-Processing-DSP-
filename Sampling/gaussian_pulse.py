
import numpy as np

class gaussian_pulse(object):
    """
    Generates a gaussian pulse signal
    Arguments:
        mu, sigma: mean and variance of the Gaussian pulse
        T: duration of the signal (int)
        fs: sampling frequncy (int)
    """

    def __init__(self, mu, sigma, T, fs):
        self.N = np.int(np.floor( T * fs))

        # Create the active part
        # self.sig = signal.gaussian(self.N, std=sigma)
        self.t = np.arange(0, T, 1 / fs)
        self.sig = np.exp(-(self.t-mu)**2 / (2 * sigma**2))

        # Create the time array
        self.t = np.arange(0, T, 1 / fs)