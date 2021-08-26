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
