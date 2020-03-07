import numpy as np
from scipy.special import wofz


def normal(x):
    return 1 / np.sqrt(2.0 * np.pi) * np.exp(- 0.5 * x**2)


gauss = normal


def Gauss(x, A, x0, sigma, offset):
    """
    Standard Gaussian function with area A, centroid x0, 
    standard deviation sigma, and offset.
    """
    return normal((x - x0) / sigma) / sigma * A + offset


def _sigma_to_FWHM(sigma):
    """
    convert standard deviation to full-width-half-maximum
    """
    return 2.0 * np.sqrt(2.0 * np.log(2)) * sigma


def _FWHM_to_sigma(fwhm):
    """
    Convert full-width-half-maximum to standard deviation
    """
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2)))


Gauss.sigma_to_FWHM = _sigma_to_FWHM
Gauss.FWHM_to_sigma = _FWHM_to_sigma


def cauchy(x):
    return 1 / (np.pi * (1 + x**2))


def Lorentz(x, A, x0, gamma, offset):
    """
    Standard Lorentzian function
    """
    return cauchy((x - x0) / gamma) / gamma * A + offset


def voigt(x, sigma, gamma):
    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma / np.sqrt(2*np.pi)


def Voigt(x, A, x0, sigma, gamma, offset):
    """
    Voigt function, which is a convolution of Lorentzian and Gaussian
    """
    return voigt((x - x0), sigma, gamma) * A + offset


def voigt_fast(x, A, x0, sigma, gamma, offset):
    """
    Fast voigt function approximated by a linear combination of 
    two lorentzian and gaussian
    """
    raise NotImplementedError
