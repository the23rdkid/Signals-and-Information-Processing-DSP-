
import matplotlib.pyplot as plt

from gaussian_pulse import gaussian_pulse
from subsample import subsample
from dft import dft
from reconstruct import reconstruct

# Problem 2.2

mu = 1
sigma = 0.1
f_s = 40000
f_ss = 4000
T = 2
N = T*f_s

# Create Gaussian pulse
gaussian_obj = gaussian_pulse(mu, sigma, T, f_s)
x = gaussian_obj.sig
t = gaussian_obj.t

# Subsample
subsample_obj = subsample(x, 1/f_s, 1/f_ss)
x_s, x_delta = subsample_obj.solve()

# Plot
fig, axs = plt.subplots(2)
axs[0].grid()
axs[1].grid()
fig.suptitle('Original signal and subsampled signal' )
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
axs[0].plot(t, x)
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Signal')
axs[1].plot(t, x_delta)
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Signal')
plt.savefig('signal_and_subsampled_time_'+ str(sigma) + '.png')

# Problem 2.3

DFT_x = dft(x, f_s)
[_, _, freqs_c, X_c] = DFT_x.solve3()

DFT_x_delta = dft(x_delta, f_s)
[_, _, _, X_delta_c] = DFT_x_delta.solve3()

# Plot
fig, axs = plt.subplots(2)
axs[0].grid()
axs[1].grid()
fig.suptitle('DFT of original signal and subsampled signal' )
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
axs[0].plot(freqs_c, abs(X_c))
axs[0].set_xlabel('Frequency (Hz)')
axs[0].set_ylabel('DFT')
axs[1].plot(freqs_c, abs(X_delta_c))
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('DFT')
plt.savefig('signal_and_subsampled_freq_'+ str(sigma) + '.png')


# Problem 2.6 (first instance)

# Reconstruct

reconstruct_obj = reconstruct(x_s, 1/f_s, 1/f_ss)
x_r = reconstruct_obj.solve()

# Plot
fig, axs = plt.subplots(2)
axs[0].grid()
axs[1].grid()
fig.suptitle('Original signal and reconstructed signal' )
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
axs[0].plot(t, x)
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Signal')
axs[1].plot(t, x_r)
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Signal')
plt.savefig('signal_and_reconstructed_time_'+ str(sigma) + '.png')


# Problems 2.4--2.5

# Subsample
subsample_obj = subsample(x, 1/f_s, 1/f_ss)
x_s, x_delta = subsample_obj.solve2()

DFT_x = dft(x, f_s)
[_, _, freqs_c, X_c] = DFT_x.solve3()

DFT_x_delta = dft(x_delta, f_s)
[_, _, _, X_delta_c] = DFT_x_delta.solve3()

# Plot
fig, axs = plt.subplots(2)
axs[0].grid()
axs[1].grid()
fig.suptitle('DFT of original signal and prefiltered + subsampled signal' )
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
axs[0].plot(freqs_c, abs(X_c))
axs[0].set_xlabel('Frequency (Hz)')
axs[0].set_ylabel('DFT')
axs[1].plot(freqs_c, abs(X_delta_c))
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('DFT')
plt.savefig('signal_and_prefiltered_freq_'+ str(sigma) + '.png')

# Problem 2.6 (second instance)

# Reconstruct

reconstruct_obj = reconstruct(x_s, 1/f_s, 1/f_ss)
x_r = reconstruct_obj.solve()

# Plot
fig, axs = plt.subplots(2)
axs[0].grid()
axs[1].grid()
fig.suptitle('Original signal and reconstructed signal' )
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
axs[0].plot(t, x)
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Signal')
axs[1].plot(t, x_r)
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Signal')
plt.savefig('signal_and_reconstructed_no_loss_time_'+ str(sigma) + '.png')
