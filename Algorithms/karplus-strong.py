"""
1. http://www.music.mcgill.ca/~gary/courses/projects/618_2009/NickDonaldson/
2. https://theory.stanford.edu/~blynn/sound/karplusstrong.html
3. https://www.math.drexel.edu/~dp399/musicmath/Karplus-Strong.html#The-Karplus-Strong-Algorithm
"""

import pyaudio
import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# base frequency of a guitar's 6 strings
E4 = 329.6276
B3 = 246.9417
G3 = 195.9977
D3 = 146.8324
A2 = 110
E2 = 82.4069

fb = G3
fs = 44100
duration = 4

N = int(fs / fb - 0.5)   # f = fs / (N + 1/2)
data = np.zeros(fs * duration)
data[:N] = np.random.normal(0, 1, N)
data[N] = 0.5 * data[0]
for i in range(N+1, len(data)):
    data[i] = (data[i-N] + data[i-N-1]) / 2
data = data - np.mean(data)
data = data / max(abs(data)) * 0.5
data = data.astype(np.float32)
wav.write('./karplus-strong.wav', fs, data)

pa = pyaudio.PyAudio()
stream = pa.open(format=pa.get_format_from_width(4), channels=1, rate=fs, output=True)
stream.write(data)
stream.stop_stream()
stream.close()
pa.terminate()

HALF_FFT = slice(0, len(data) // 2)
y = fft(data)
y = 2 * np.abs(y[HALF_FFT]) / len(data)
x = fftfreq(len(data), 1 / fs)[HALF_FFT]

plt.figure(figsize=(6, 1.8), dpi=100)
plt.plot(x, y, 'b-')
xticks = fb * np.arange(10)
xlabel = [str(int(i)) for i in xticks]
plt.xticks(ticks=xticks, labels=xlabel)
plt.xlim([xticks[0], xticks[-1]])
plt.tight_layout()
plt.savefig("./karplus.png", transparent=True)
