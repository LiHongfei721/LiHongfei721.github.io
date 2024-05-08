from scipy.fft import fft
import numpy as np
import time


def dit_fft_recursion(x):
    n = len(x)

    if n == 1:
        return x

    y0 = dit_fft_recursion(x[0::2])
    y1 = dit_fft_recursion(x[1::2])

    w = np.exp(-1j * 2 * np.pi / n * np.arange(0, n // 2))

    y = np.zeros(n, dtype=complex)
    y[:n//2], y[n//2:] = y0 + w * y1, y0 - w * y1

    return y


def dif_fft_recursion(x):
    n = len(x)

    if n == 1:
        return x

    w = np.exp(-1j * 2 * np.pi / n * np.arange(0, n // 2))

    y0 = dif_fft_recursion(x[0:n//2] + x[n//2:])
    y1 = dif_fft_recursion(w * (x[0:n//2] - x[n//2:]))

    y = np.zeros(n, dtype=complex)
    y[0::2], y[1::2] = y0, y1

    return y


'''
def array_rearange(x):
    if len(x) == 2:
        return x

    x0 = array_rearange(x[0::2])
    x1 = array_rearange(x[1::2])

    return np.concatenate((x0, x1))
'''


def array_rearange(x):
    def bit_reverse(v, n=32):
        v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1)
        v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2)
        v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4)
        v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8)
        v = (v >> 16) | (v << 16)
        v = v >> (32 - n)
        return v

    N = len(x)

    n = int(np.log2(N))

    y = np.zeros(N, dtype=complex)
    for i in range(N):
        y[i] = x[bit_reverse(i, n)]

    return y


def dit_fft_iteration(x):
    x = np.array(array_rearange(x), dtype=complex)

    N = len(x)

    W = np.exp(-1j * 2 * np.pi / N * np.arange(N // 2))

    L = int(np.log2(N))
    for i in range(L):

        M = int(2 ** (i + 1))

        WN = W[::N//M]

        for k in range(0, N, M):
            x0, x1 = x[k: k + M//2], x[k + M//2: k+M]
            x[k: k + M//2], x[k + M//2: k + M] = x0 + x1 * WN, x0 - x1 * WN

    return x


def dif_fft_iteration(x):
    x = np.array(x, dtype=complex)

    N = len(x)

    W = np.exp(-1j * 2 * np.pi / N * np.arange(N // 2))

    L = int(np.log2(N))
    for i in range(L):

        M = int(2 ** (L - i))

        WN = W[::2**i]

        for k in range(0, N, M):
            x0, x1 = x[k:k+M//2], x[k+M//2:k+M]
            x[k:k+M//2], x[k+M//2:k+M] = x0 + x1, (x0 - x1) * WN

    return array_rearange(x)


c2 = np.random.rand(2 ** 16)

subset = slice(len(c2)//2 + 1, len(c2)//2 + 5)

test = [fft,
        dit_fft_recursion, dif_fft_recursion,
        dit_fft_iteration, dif_fft_iteration]

for func in test:
    print(func(c2)[subset])


for func in test:
    t0 = time.perf_counter()
    func(c2)
    t1 = time.perf_counter()
    print(t1 - t0)
