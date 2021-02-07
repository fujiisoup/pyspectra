import numpy as np
from scipy.special import wofz, gamma, gammaincc
from scipy import stats


def normal(x):
    return 1 / np.sqrt(2.0 * np.pi) * np.exp(-0.5 * x ** 2)


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
    return 1 / (np.pi * (1 + x ** 2))


def Lorentz(x, A, x0, gamma, offset):
    """
    Standard Lorentzian function
    """
    return cauchy((x - x0) / gamma) / gamma * A + offset


def _gamma_to_FWHM(gamma):
    """
    convert standard deviation to full-width-half-maximum
    """
    return 2.0 * gamma


def _FWHM_to_gamma(fwhm):
    """
    Convert full-width-half-maximum to standard deviation
    """
    return fwhm / 2.0


Lorentz.gamma_to_FWHM = _gamma_to_FWHM
Lorentz.FWHM_to_gamma = _FWHM_to_gamma


def voigt(x, sigma, gamma):
    return (
        np.real(wofz((x + 1j * gamma) / sigma / np.sqrt(2)))
        / sigma
        / np.sqrt(2 * np.pi)
    )


def Voigt(x, A, x0, sigma, gamma, offset):
    """
    Voigt function, which is a convolution of Lorentzian and Gaussian
    """
    return voigt((x - x0), sigma, gamma) * A + offset


def _sigma_gamma_to_FWHM(sigma, gamma):
    """
    Estimate FWHM for Voigt
    """
    fG = _sigma_to_FWHM(sigma)  # Gauss width
    fL = _sigma_to_FWHM(gamma)  # Lorentz width
    return 0.5346 * fL + np.sqrt(0.2166 * fL ** 2 + fG ** 2)


Voigt.sigma_gamma_to_FWHM = _sigma_gamma_to_FWHM


def voigt_fast(x, A, x0, sigma, gamma, offset):
    """
    Fast voigt function approximated by a linear combination of 
    two lorentzian and gaussian
    """
    raise NotImplementedError


def GeneralizedVoigt1(x, A, x0, sigma, gamma, df, offset):
    """
    Generalized voigt function with scale.
    This is defined as a convolution of a Gaussian distribution and t-distribution

    A: area
    x0: location parameter
    sigma: scale parameter for gaussian
    gamma: scale parameter for t-distribution
    df: degree of freedom for t-distribution
    offset: offset
    """
    return (
        generalized_voigt1((x - x0) / sigma, 1, gamma / sigma, df) / sigma * A + offset
    )


def generalized_voigt1(
    x, gamma, df, num_points=32, method="log_trapz", w_min=0.03, w_max=30.0,
):
    """
    Generalized voigt profile, where we convolute gaussian and student-t distribution with degree of 
    freedom df.

    gamma, df: scale parameter and degree of freedom for the t-distribution
    num_points: number of points to carry out the integration
    method: method for integration
    """
    x, gamma, df = np.broadcast_arrays(x, gamma, df)
    gamma2 = gamma ** 2
    dfhalf = df / 2
    # when w * gamma << 1
    pdf_lo = normal(x) * gammaincc(dfhalf, dfhalf * gamma2 / w_min)
    # when w * gamma >> 1
    pdf_hi = stats.t.pdf(x, loc=0, scale=gamma, df=df) * (
        1 - gammaincc(dfhalf + 0.5, (dfhalf * gamma2 + x ** 2 / 2) / w_max)
    )
    # when w * gamma ~ 1
    if method == "log_trapz":
        # w indicates w * gamma**2
        w = np.logspace(np.log10(w_min), np.log10(w_max), base=10, num=num_points)
        w = np.expand_dims(w, (-1,) * x.ndim)
        sqrt_w = np.sqrt(w + 1)
        pdf = np.trapz(
            normal(x / sqrt_w)
            / sqrt_w
            * stats.invgamma.pdf(w, a=dfhalf, loc=0, scale=dfhalf * gamma2),
            x=w,
            axis=0,
        )
    else:
        raise NotImplementedError

    return pdf + pdf_lo + pdf_hi
