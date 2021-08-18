import os
import numpy as np
from scipy.special import wofz, gamma, gammaincc
from scipy import stats

from .data import assure_directory, _default_cache_dir


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
    return generalized_voigt1((x - x0) / sigma, gamma / sigma, df) / sigma * A + offset


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
        
    elif method == "trapz":
        # w indicates w * gamma**2
        w = np.linspace(w_min, w_max, num=num_points)
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


def symmetric_stable(x, alpha, method='scipy', options=None):
    """
    Levy's alpha stable distribution with beta=0.

    Parameters
    ----------
    method:
        computation method. One of ['scipy' | 'mixture']
    """
    if options is None:
        options = {}
    if method.lower() == 'scipy':
        return stats.levy_stable.pdf(x, alpha, beta=0, **options)
    if method.lower() == 'mixture':
        return _symmetric_stable_mixture(x, alpha, options)


def positive_stable(x, alpha, method='scipy', options=None):
    """
    Levy's alpha stable distribution with beta=1.

    Parameters
    ----------
    method:
        computation method. One of ['scipy' | ]
    """
    if options is None:
        options = {}
    if method.lower() == 'scipy':
        return stats.levy_stable.pdf(x, alpha, beta=1, **options)


def _symmetric_stable_mixture(x, alpha, options):
    """
    Mixture approximations for symmetric stable distribution

    kuruoglu_approximation_1998,
	Approximation of alpha-stable probability densities using finite {Gaussian} mixtures
    """
    x = np.array(x)[..., np.newaxis]
    alpha = np.array(alpha)[..., np.newaxis]

    # default number of points
    num_points = getattr(options, 'num_points', 100)
    scale = np.cos(np.pi * alpha / 4)**(2 / alpha) * 2

    # integration based on the gaussian hermite quadrature rule
    # TODO optimize scale
    v = np.linspace(0, 30, num_points+1)[1:]**2
    w = np.gradient(v)

    # TODO enable to use custom method
    mixture = positive_stable(v / scale, alpha / 2, method='scipy') / scale
    gaussians = normal(x / np.sqrt(v)) / np.sqrt(v)
    return np.sum(gaussians * mixture * w, axis=-1)


# interpolation instance for the positive stable distribution
_PSTABLE_INTERP = None

def _positive_stable_interp(x, alpha, options):
    """
    Positive stable distribution based on interpolation
    """
    force_compute = getattr(options, 'force_compute', False)
    if _PSTABLE_INTERP is None or force_compute:
        # read the precomputed data
        cache_dir = os.sep.join(
            (_default_cache_dir, "stats"))
        assure_directory(cache_dir)
        filename = os.sep.join((cache_dir, "positive_stable.txt"))
        if os.path.exists(filename):
            data = np.loadtxt(filename)

    return _PSTABLE_INTERP(x, alpha)