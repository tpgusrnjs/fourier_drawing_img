import numpy as np

def contour_to_fourier(contour):
    z = contour[:, 0] + 1j * contour[:, 1]
    z -= np.mean(z)

    coeffs = np.fft.fft(z) / len(z)
    N = len(z)

    freqs = np.arange(-N//2, N//2)
    coeffs = np.fft.fftshift(coeffs)

    order = np.argsort(np.abs(freqs))
    return coeffs[order], freqs[order]

def epicycle_position(coeffs, freqs, t, K):
    pos = 0j
    for i in range(K):
        pos += coeffs[i] * np.exp(2j * np.pi * freqs[i] * t)
    return pos