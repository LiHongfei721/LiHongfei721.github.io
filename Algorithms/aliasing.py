import numpy as np
import matplotlib.pyplot as plt


"""
There are two signals:
    x1(t) = sin(w1 * t)
    x2(t) = sin(w2 * t)

Sampling both x1(t) and x2(t) with period T, then we have:
    x1[n] = sin(w1 * n * T)
    x2[n] = sin(w2 * n * T)

if x1[n] == x2[n]
then w1 * n * T == w2 * n * T + 2 * k * pi * n, for k = 0, 1, 2, ...
     w2 = w1 + 2 * k * pi / T
"""


k = 1
T = 0.099
w1 = 4 * np.pi
w2 = w1 + 2 * k * np.pi / T

seconds = 3 * 2 * np.pi / w1

t = np.arange(0, seconds, 1e-6)
x1t = np.sin(w1 * t)
x2t = np.sin(w2 * t)

n = np.arange(0, seconds/T)
x1n = np.sin(w1 * n * T)
x2n = np.sin(w2 * n * T)

plt.figure(num=None, figsize=(10, 3.6))
plt.plot(n * T, x1n, 'bo')
# plt.plot(n * T, x2n, 'gx', markersize=10)
plt.plot(t, x1t, 'r-')
plt.plot(t, x2t, 'g:')

ticks = np.arange(0, seconds + 1, T)
labels = [f"{i:.2f}" for i in ticks]
plt.xticks(ticks=ticks, labels=labels, rotation=90)

plt.xlim([0, seconds])

plt.grid(visible=True, axis='x', ls='--')
plt.tight_layout()
plt.show()
