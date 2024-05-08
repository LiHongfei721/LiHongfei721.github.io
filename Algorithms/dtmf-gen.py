import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt


key = '0'
output_file = "./dtmf.wav"
sample_rate = 48000   # Hz
valid_time = 0.2      # seconds


def harmonic_oscillator(x, w):
    a = np.sin(w)
    b = 2 * np.cos(w)

    y = np.zeros(x.size)
    y[0] = 0
    y[1] = a * x[0]
    for n in range(2, x.size):
        y[n] = b*y[n-1] - y[n-2]

    return y


dtmf_dict = {
    '1': (697, 1209),
    '2': (697, 1336),
    '3': (697, 1477),
    '4': (770, 1209),
    '5': (770, 1336),
    '6': (770, 1477),
    '7': (852, 1209),
    '8': (852, 1336),
    '9': (852, 1477),
    '*': (941, 1209),
    '0': (941, 1336),
    '#': (941, 1477),
}

fs = sample_rate
f1, f2 = dtmf_dict[key]
w1, w2 = 2 * np.pi * f1 / fs, 2 * np.pi * f2 / fs

x = np.zeros(int(valid_time * fs))
x[0] = 1
y = harmonic_oscillator(x, w1) + harmonic_oscillator(x, w2)
y /= max(abs(y))

plt.plot(y[0:y.size//10:])

wav.write(output_file, sample_rate, y)
