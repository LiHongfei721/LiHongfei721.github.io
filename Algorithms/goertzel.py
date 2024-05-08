import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt


fs, x = wav.read('./dtmf.wav')
N = len(x)
multi = np.array([697, 770, 852, 941, 1209, 1336, 1477, 1633])
index = N * multi / fs + 0.5
index = index.astype(np.int32)

y = []
for k in index:
    cosine = np.cos(2 * np.pi * k / N)

    v1, v2 = 0, 0
    for n in range(N):
        v0 = x[n] + 2 * cosine * v1 - v2
        v2 = v1
        v1 = v0

    y.append(v0 - np.exp(-1j * 2 * np.pi / N * k) * v2)

y = np.abs(y) * 2 / N

plt.figure(figsize=(6, 2), dpi=100)
plt.stem(multi, y)
plt.xticks(ticks=multi, labels=[str(f) for f in multi])
# plt.xlabel("(Hz)")
plt.ylim(bottom=0.0)
plt.tight_layout()
plt.savefig("./goertzel-dtmf.png", transparent=True)
plt.show()

print(multi[y > y.mean()])
