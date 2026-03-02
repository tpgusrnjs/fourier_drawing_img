import numpy as np

def contour_to_fourier(contour):
    center = contour.mean(axis=0)

    z = (contour[:, 0] - center[0]) + 1j * (contour[:, 1] - center[1])

    coeffs = np.fft.fft(z) / len(z)
    N = len(z)

    freqs = np.arange(-N//2, N//2)
    coeffs = np.fft.fftshift(coeffs)

    order = np.argsort(np.abs(freqs))
    return coeffs[order], freqs[order], center
