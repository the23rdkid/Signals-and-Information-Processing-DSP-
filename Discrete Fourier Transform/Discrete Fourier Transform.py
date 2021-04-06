# ESE 224 --- Signal and Information Processing
#
# Spring 2021
#
# Discrete Fourier Transforms
#
# Main function


###############################################################################
############################# I M P O R T I N G ###############################
###############################################################################

# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
import warnings
from scipy.io.wavfile import write

# Local files
import cexp
import dft
import discrete_signal


def energy(f, X,  interval):
    """
    energy computes the energy of a signal in a frequency interval
    X: DFT of signal (array)
    f: frequency array of signal DFT
    interval: Frequency interval (list of len()=2 of int), interval[0]=<interval[1]
    :return: Energy of signal in interval (float)
    """
    aux=0
    for i,freq in enumerate(f):
        if freq>=interval[0] and freq<=interval[1]:
            aux=aux+abs(X[i]*np.conjugate(X[i]))

    return aux

###############################################################################
############################ Q U E S T I O N 1.1 ##############################
###############################################################################


###############################################################################
############################ Q U E S T I O N 1.1 ##############################
###############################################################################

# In order to recover values in the canonical set [0,N-1], we can chop and shift.
# That is, take the DFT values associated with the set [-N/2,-1] and set them
# as the dft values for frequencies [N/2,N-1]

###############################################################################
############################ Q U E S T I O N 1.2 ##############################
###############################################################################

def q_12(T, fs, T0s):
    """
    Question 1.2: DFT of a pulse
    Arguments:
        T: duration of the signal (int)
        fs: sampling frequency (int)
        T0s: square_pulses_lengths (list of ints)
    """
    for T0 in T0s:

        # Obtain the signal
        sqpulse_signal = discrete_signal.sqpulse(T0, T, fs)

        # Obtain the DFT
        sqpulse_dft = dft.dft(sqpulse_signal.pulse, fs)
        [freqs,X,f_c,X_c]=sqpulse_dft.solve3()

        # Compute the enegy in interval
        Total_ener=np.sum(abs(X_c)**2)
        Partial_ener = abs(energy(f_c, X_c, [-1 / T0, 1 / T0]))
        print('Energy fraction of a square pulse of T0=%3.2f is %5.4f'%(T0, Partial_ener/Total_ener))

        # Plot
        fig, axs = plt.subplots(2)
        axs[0].grid()
        axs[1].grid()
        fig.suptitle('Square Pulse of Width %3.2fs '%(T0))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
        axs[0].plot(sqpulse_signal.t, sqpulse_signal.pulse)
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Signal')
        axs[1].plot(f_c,abs(X_c))
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylabel('DFT')
        plt.savefig('square_pulse_'+str(T0)+'.png')
        plt.show()

###############################################################################
############################ Q U E S T I O N 1.3 ##############################
###############################################################################

def q_13(T, fs, T0s):
    """
    Question 1.3: DFT of a triangular pulse
    Arguments:
        T: duration of the signal (int)
        fs: sampling frequency (int)
        T0s: square_pulses_lengths (list of ints)
    """
    for T0 in T0s:

        # Obtain the signal
        tripulse_signal = discrete_signal.tripulse(T0, T, fs)

        # Obtain the DFT
        tripulse_dft = dft.dft(tripulse_signal.pulse, fs)
        [freqs,X,f_c,X_c]=tripulse_dft.solve1()

        # Compute the enegy in interval
        Total_ener=np.sum(abs(X_c)**2)
        Partial_ener = abs(energy(f_c, X_c, [-1 / T0, 1 / T0]))
        print('Energy fraction of a triangular pulse of T0=%3.2f is %5.4f'%(T0, Partial_ener/Total_ener))

        # Plot
        fig, axs = plt.subplots(2)
        axs[0].grid()
        axs[1].grid()
        fig.suptitle('Triangular Pulse of Width %3.2fs '%(T0))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
        axs[0].plot(tripulse_signal.t, tripulse_signal.pulse)
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Signal')
        axs[1].plot(f_c,abs(X_c))
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylabel('DFT')
        plt.savefig('triangular_pulse_'+str(T0)+'.png')
        plt.show()

###############################################################################
############################ Q U E S T I O N 1.4 ##############################
###############################################################################

def q_14(T, fs, params):
    """
    Question 1.4: DFT of a other pulses
    Arguments:
        T: duration of the signal (int)
        fs: sampling frequency (int)
        params: parameters of other pulses
    """


    # Obtain the kaiser signal
    kaiser_signal = discrete_signal.kaiser_window(params[0], T, fs)

    # Obtain the DFT
    kaiser_dft = dft.dft(kaiser_signal.signal, fs)
    [freqs,X,f_c,X_c]=kaiser_dft.solve3()

    # Compute the enegy in interval

    # Plot
    fig, axs = plt.subplots(2)
    axs[0].grid()
    axs[1].grid()
    fig.suptitle('Kaiser Window of Beta %3.2f '%(params[0]))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    axs[0].plot(kaiser_signal.t, kaiser_signal.signal)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Signal')
    axs[1].plot(f_c,abs(X_c))
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('DFT')
    plt.savefig('kaiser_window_'+str(params[0])+'.png')

    plt.show()

###############################################################################
############################ Q U E S T I O N 3.1 ##############################
###############################################################################
def q_31(f0, T, fs):
    """
    Question 3.1: DFT of an A note
    Arguments:
        T: duration of the signal (int)
        fs: sampling frequency (int)
        f0: frequency of the note (int)
    """
    # Retrieves complex exponential
    t, cpxexp, num_samples = cexp.cexpt(f0, T, fs)
    cos = np.real(cpxexp)
    # Obtain the DFT
    Anote_dft = dft.dft(cos, fs)
    [freqs, X, f_c, X_c] = Anote_dft.solve3()
    # print(f_c[np.where(f_c == f0)], fs*np.where(freqs == f0)[0]/len(f_c),np.where(freqs == f0))

    # DFT Conjugate symmetric
    # Obtain the DFT
    print('The norm of the signal of an A note is',np.linalg.norm(abs(np.real(cpxexp)))**2)
    print('The norm of the DFT of an A note is',np.linalg.norm(abs(X_c))**2)

    # Plot
    plt.figure()
    plt.grid(True)
    plt.plot(f_c, abs(X_c))
    plt.xlim((-500,500))
    plt.title('DFT of an A note')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('DFT')
    plt.savefig('Anote_signal_dft.png')
    plt.show()
    # The problem remains in the fact that the total time of the signal should be an multiple
    # of the period of the A note.

###############################################################################
############################ Q U E S T I O N 3.2 ##############################
###############################################################################
def q_32(list_notes, list_times, fs):
    """
    Question 3.2: DFT of an song
    Arguments:
        list_notes: list of notes of the song (list)
        list_times: list of the times of each note (list)
        fs: sampling frequency (int)
    """
    assert len(list_notes) == len(list_times), "List of musical notes and musical times should have same length"
    song = []
    for note, note_time in zip(list_notes, list_times):
        fi = 2**((note - 49) / 12)*440
        _, x, N = cexp.cexpt(fi, note_time, fs)
        song = np.append(song, x.real)
        song = np.append(song, np.zeros(10))

    Anote_dft = dft.dft(song, fs)
    [freqs, X, f_c, X_c] = Anote_dft.solve3()


    plt.plot(f_c, abs(X_c))
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('DFT')
    plt.title('DFT of Minuet in G Major')

    plt.figure()
    plt.plot(f_c, abs(X_c))
    plt.xlim((-2000,2000))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('DFT')
    plt.title('DFT of Minuet in G Major')
    plt.savefig('song_dft.png')
    plt.show()
    return song,f_c, X_c

###############################################################################
############################ Q U E S T I O N 3.3 ##############################
###############################################################################
def q_33(song, f_c, X_c,notes_freq, notes_repetitions, notes):
    """
    Energy of different tones of a musical piece
    :list song: List that contains the song
    :list f_c: List of frequencies of the DFT of the song (centered in 0)
    :list X_c: List of values of the DFT of the song (centered in 0
    :list notes_freq: List of frequencies of the notes used for the song
    :list notes_repetitions: List of repetitions of each note i song
    :list notes:
    """
    fig, axs = plt.subplots(2)
    axs[0].grid()
    axs[1].grid()
    fig.suptitle('Energy per note of Minuet in G Major')
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

    ener_X_c=[abs(X_c[i])**2 for i in range(len(X_c))]

    axs[0].plot(f_c, ener_X_c)
    axs[0].set_xlim((0,2000))
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('DFT')
    y_max=max(ener_X_c)*1.1
    axs[0].set_ylim((0, y_max))
    axs[1].bar(notes_freq,notes_repetitions, width=15)
    # axs[1].set_xticks(notes_freq)
    y_max = max(notes_repetitions) * 1.1
    axs[1].set_ylim((0, y_max))
    axs[1].set_xlim((0,2000))
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Number of time')
    plt.savefig('EnergyPerNote.png')
    plt.show()

###############################################################################
############################ Q U E S T I O N 3.4 ##############################
###############################################################################
def q_34(f0, T, fs, instruments, instruments_name):
    """
    DFT of an A note of different musical instruments
    :int f0: Frequency of the note
    :int T: Duration of the signal
    :int fs: sampling frequency
    :list of lists of float instruments: list of harmonics of each instrument
    :list of str instruments_name: list of names of instruments
    :return:
    """
    # Obtain the A note for each instrument and the DFT
    for j,harmonic in enumerate(instruments):
        note=np.zeros(T*fs)
        for i,h in enumerate(harmonic):
            t, cpxexp, num_samples = cexp.cexpt(f0*(i+1), T, fs)
            note=note+h*np.real(cpxexp)
            # Cosine is the real part


        Anote_dft = dft.dft(note, fs)
        [freqs, X, f_c, X_c] = Anote_dft.solve3()

        total_energy=sum(abs(X_c)**2)
        energy_per_harmonic=[]
        for i,h in enumerate(harmonic):
            energy_per_harmonic=energy_per_harmonic+[energy(f_c, X_c, [-f0*(i+1),f0*(i+1)]) /total_energy]

        print(instruments_name[j],energy_per_harmonic)

        plt.figure()
        plt.grid(True)
        plt.title('A note in a '+instruments_name[j])

        plt.plot(f_c, abs(X_c))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('DFT')
        plt.savefig(instruments_name[j]+'_Anote_dft.png')
    plt.show()

###############################################################################
############################ Q U E S T I O N 3.5 ##############################
###############################################################################
def q_35(list_notes,list_times, fs, instrument, instrument_name):
    """
    DFT of your song on a musical instrument.
    :param list_notes: List of notes of song
    :param list_times: List of times per note of song
    :param fs: sampling frequency
    :param instrument: harmonics of intrument
    :param instrument_name: string with intrument name
    :return:
    """
    # Obtain the song played in your favourite intrument
    song = []
    for note, note_time in zip(list_notes, list_times):
        fi = 2**((note - 49) / 12)*440
        note = np.zeros(math.floor(note_time * fs))
        for i,h in enumerate(instrument):
            t, x, N = cexp.cexpt(fi*(i+1), note_time, fs)
            note=note+h*x

        song = np.append(song, x.real)

    write('song_in'+instrument_name+'.wav', fs, song.astype(np.float32))
    song_dft = dft.dft(song, fs)
    [freqs, X, f_c, X_c] = song_dft.solve3()

    plt.figure()
    plt.grid(True)
    plt.title('Minuet in G Major played in the '+instrument_name)
    plt.plot(f_c, abs(X_c))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('DFT')
    plt.savefig('Minuet_in_G'+instrument_name+'.png')
    plt.show()


if __name__ == '__main__':
    # # Problem 1.2
    # # Pulses widths
    # T0s = [0.5, 1, 4, 16]
    # duration_of_signal = 32
    # sampling_frequency = 8
    # q_12(duration_of_signal, sampling_frequency, T0s)
    #
    # # Problem 1.3
    # # Pulses widths
    # T0s = [0.5, 1, 4, 16]
    # duration_of_signal = 32
    # sampling_frequency = 8
    # q_13(duration_of_signal, sampling_frequency, T0s)

    # Problem 1.4
    # Pulses
    # beta = 16
    # params = [beta]
    # duration_of_signal = 32
    # sampling_frequency = 8
    # q_14(duration_of_signal, sampling_frequency, params)

    # Problem 3.2
    f0 = 440
    T = 2
    fs = 44100
    # fs = 8000
    q_31(f0, T, fs)

    # Problem 3.3
    # mapping notes
    C4=40
    D4=42
    D4s=43
    E4=44
    F4=45
    G4=47
    A4=49
    A4s=50
    B4=51
    C5=52
    D5=54
    D5s=55
    E5=56
    F5=57
    G5=59
    A5=61
    A5s=62
    B5=63
    C6=64
    C6s=65
    D6=66
    E6=68
    F6=69
    F6s=70
    G6=71
    A6=73
    B6=75
    # Notes

    # Song
    song_notes = [D6, G5, A5, B5, C6, D6, G5, G5, E6, C6, D6, E6, F6s, G6, G5, G5, C6, D6, C6, B5, A5, B5, C6, B5, A5, G5, F5, G5, A5, B5, G5, B5, A5,
                D6, G5, A5, B5, C6, D6, G5, G5, E6, C6, D6, E6, F6s, G6, G5, G5,
                C6, D6, C6, B5, A5, B5, C6, B5, A5, G5, A5, B5, A5, G5, F5, G5,
                B6, G6, A6, B6, G6, A6, D6, E6, F6s, D6, G6, E6, F6s, G6, D6, C6s, B5, C6, A5,
                A5, B5, C6s, D6, E6, F6, G6, F6, E6, F6, A5, C6s, D6,
                D6, G5, F5, G5, E6, G5, F5, G5, D6, C6, B5, A5, G6, F5, G5, A5,
                D5, E5, F5, G5, A5, B5, C6, B5, A5, B5, D6, G5, F5, G5]
    # song_notes = [D6, G5]
    rhythm=1
    b = 1*rhythm
    w = 2*rhythm
    h = 0.5*rhythm
    dw = 3*rhythm;
    # song_times=[b, h]

    song_times = [b, h, h, h, h, b, b, b, b, h, h, h, h, b, b, b,
                b, h, h, h, h, b, h, h, h, h, b, h, h, h, h, b, w,
                b, h, h, h, h, b, b, b, b, h, h, h, h, b, b, b,
                b, h, h, h, h, b, h, h, h, h, b, h, h, h, h, dw,
                b, h, h, h, h, b, h, h, h, h, b, h, h, h, h, b, h, h, b,
                h, h, h, h, h, h, b, b, b, b, b, b, dw,
                b, h, h, b, b, h, h, b, b, b, b, h, h, h, h, b,
                h, h, h, h, h, h, b, b, b, h, h, b, b, dw]


    song,f_c, X_c=q_32(song_notes, song_times, fs)

    notes = [C4, D4, D4s, E4, F4, G4, A4, A4s, B4, C5, D5, D5s, E5, F5, G5, A5, A5s, B5, C6, C6s, D6, E6, F6, F6s, G6,
             A6, B6]
    notes_freq = [2 ** ((note - 49) / 12) * 440 for note in notes]

    notes_repetitions=[0 for i in range(len(notes))]
    for i,note in enumerate(song_notes):
        notes_repetitions[notes.index(int(note))]=notes_repetitions[notes.index(int(note))]+song_times[i]

    q_33(song, f_c, X_c, notes_freq,notes_repetitions,notes)

    oboe=[1.386, 1.370, 0.360, 0.116, 0.106, 0.201, 0.037, 0.019]
    flute=[0.260, 0.118, 0.085, 0.017, 0.014]
    trumpet=[1.167, 1.178, 0.611, 0.591, 0.344, 0.139,
                0.090, 0.057, 0.035, 0.029, 0.022, 0.020, 0.014]
    clarinet=[0.061, 0.628, 0.231, 1.161, 0.201, 0.328, 0.154, 0.072, 0.186, 0.133,
                0.309, 0.071, 0.098, 0.114, 0.027, 0.057, 0.022, 0.042, 0.023]

    f0 = 440
    T = 2
    fs = 44100
    q_34(f0,T,fs,[oboe,flute,trumpet,clarinet],['oboe','flute','trumpet','clarinet'])

    q_35(song_notes, song_times,  fs, flute, 'flute')