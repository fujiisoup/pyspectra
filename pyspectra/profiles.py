import os
import numpy as np
from scipy.special import wofz, gamma, gammaincc
from scipy import stats

from .data import assure_directory, _default_cache_dir

try:
    import levy
    HAS_LEVY = True
except ImportError:
    HAS_LEVY = False


def _assert_has_levy(function):
    if not HAS_LEVY:
        raise ImportError(
            "levy should be installed to use {}.\n".format(function) + 
            "try `pip install pylevy`."
        )


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


def student_t(x, df):
    """
    Student's t-distribution with degree of freedom of df
    """
    return stats.t.pdf(x, df=df)


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


def symmetric_stable(x, alpha, method='pylevy', options=None):
    """
    Levy's alpha stable distribution with beta=0.

    Parameters
    ----------
    method:
        computation method. One of ['scipy' | 'pylevy' | 'mixture']
    In order to use `pylevy`, it should be installed by
    
    >>> pip install git+https://github.com/josemiotto/pylevy.git
    """
    if options is None:
        options = {}
    if method.lower() == 'scipy':
        return stats.levy_stable.pdf(x, alpha, beta=0, **options)
    if method.lower() == 'pylevy':
        _assert_has_levy('symmetric_stable')
        return levy.levy(x, alpha, beta=0, mu=0.0, sigma=1.0, cdf=False)
    if method.lower() == 'mixture':
        return _symmetric_stable_mixture(x, alpha, options)


def positive_stable(x, alpha, method='pylevy', options=None):
    """
    Levy's alpha stable distribution with beta=1.

    Parameters
    ----------
    method:
        computation method. One of ['scipy' | 'pylevy']
    """
    if options is None:
        options = {}
    if method.lower() == 'scipy':
        return stats.levy_stable.pdf(x, alpha, beta=1, **options)
    if method.lower() == 'pylevy':
        _assert_has_levy('positive_stable')
        return levy.levy(x, alpha, beta=1, mu=0.0, sigma=1.0, cdf=False)


def _symmetric_stable_mixture(x, alpha, options):
    """
    Mixture approximations for symmetric stable distribution

    kuruoglu_approximation_1998,
	Approximation of alpha-stable probability densities using finite {Gaussian} mixtures
    """
    x = np.array(x)[..., np.newaxis]
    if not np.isscalar(alpha):
        raise ValueError('alpha must be a scalar, not an array.')

    # default number of points
    num_points = options.get('num_points', 31)
    scale = np.cos(np.pi * alpha / 4)**(2 / alpha) * 2

    # integration based on the gaussian hermite quadrature rule
    # TODO optimize scale
    #v, w = np.polynomial.laguerre.laggauss(num_points)
    vmax = 1000
    log_vmin = ((alpha - 2) * 10 + 3) / 2
    sigma_max = np.sqrt(vmax) / 3
    v = np.logspace(log_vmin, np.log10(vmax), base=10, num=num_points+1)[1:]
    w = np.gradient(v)
    
    # TODO enable to use custom method
    mixture = positive_stable(v / scale, alpha / 2, method='scipy') / scale
    
    # gaussians
    gaussians = normal(x / np.sqrt(v)) / np.sqrt(v)
    gaussians = np.sum(gaussians * mixture * w, axis=-1)
    gaussian_edge = normal(sigma_max / np.sqrt(v)) / np.sqrt(v)
    gaussian_edge = np.sum(gaussian_edge * mixture * w, axis=-1)
    
    # use power from the largest v
    power = np.abs(x[:, 0])**(-alpha-1)
    power_edge = np.abs(sigma_max)**(-alpha-1)
    power = power / power_edge * gaussian_edge
    return np.where(np.abs(x[:, 0]) < sigma_max, gaussians, power)


def mittag_leffler(x, alpha, method='mixture', options=None):
    r"""
    i.e., the symmetric geometric syable distribution defined on x in [0, \infty]
    https://en.wikipedia.org/wiki/Geometric_stable_distribution

    Its laplace distribution is 
    1 / (1 + s^\alpha)
    """
    if options is None:
        option = {}
    if method in ['mixture', 'exponential_mixture']:
        return _mittag_leffler_exponential_mixture(x, alpha, options)
    if method == 'gammma_mixture':
        return _generalized_mittag_leffler_gamma_mixture(x, delta, 1, options)


def generalized_gamma(x, r, alpha):
    """
    generalized gamma distribution
    """
    return np.abs(alpha) / gamma(r) * x**(alpha * r - 1) * np.exp(-x**alpha)


def generalized_mittag_leffler(x, alpha, nu, method='mixture', options=None):
    r"""
    A generalization of mittag_leffler distribution.

    Its laplace distribution is 
    1 / (1 + s^\alpha)^\nu
    """
    if options is None:
        option = {}
    if method == 'mixture':
        return _generalized_mittag_leffler_gamma_mixture(
            x, delta=alpha, nu=nu, options=options)


def _mittag_leffler_exponential_mixture(x, alpha, options):
    """
    Mixture representation of Linnik distribution revisited
    by Kozubowski
    """
    x = np.array(x)[..., np.newaxis]
    # TODO     
    # # default number of points
    num_points = options.get('num_points', 31)

    # TODO optimize scale
    # [hint] 
    # the mixture distribution has a sharp peak around 1 if alpha ~ 1,
    # the best digitization method may vary depending on x
    log_vmin = -5
    log_vmax = 5
    y = np.concatenate([
        np.logspace(log_vmin, 0, base=10, num=num_points // 2, endpoint=False),
        np.logspace(0, log_vmax, base=10, num=num_points // 2, endpoint=False)
    ])
    w = np.gradient(y)
    
    # TODO enable to use custom method
    yalpha = y**alpha
    mixture = yalpha / (yalpha**2 + 1 + 2 * yalpha * np.cos(np.pi * alpha))
    
    # gaussians
    exponentials = np.exp(-x * y)
    exponentials = np.sum(exponentials * mixture * w, axis=-1) * np.sin(np.pi * alpha) / np.pi
    return exponentials


def _generalized_mittag_leffler_gamma_mixture(x, delta, nu, options):
    r"""
    Mixture representation of generalized mittag_leffler distribution by 
    generalized-gamma mixture 

    On Mixture Representations for the Generalized Linnik Distribution and Their Applications in Limit Theorems
    Korolev et al.
    Theorem 4
    """
    x = np.array(x)[..., np.newaxis]
    # TODO     
    # # default number of points
    num_points = options.get('num_points', 31)
    levy_method = options.get('levy_method', 'scipy')

    # TODO optimize scale
    # [hint] 
    # the mixture distribution has a sharp peak around 1 if alpha ~ 1,
    # the best digitization method may vary depending on x
    log_vmin = -3
    log_vmax = 4
    y = np.logspace(log_vmin, log_vmax, base=10, num=num_points)
    w = np.gradient(y)
    
    scale = 1 - delta
    return np.sum(
        positive_stable(y, alpha=delta, method=levy_method) * 
        generalized_gamma(x / (y * scale), nu, delta) / (y * scale) * 
        w, axis=-1)
