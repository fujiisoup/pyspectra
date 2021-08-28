import os
import numpy as np
from scipy import interpolate, stats, special

from .data import assure_directory, _default_cache_dir
import pkg_resources

try:
    import levy
    HAS_LEVY = True
except ImportError:
    HAS_LEVY = False


FILE_SYMMETRIC_LEVY = pkg_resources.resource_filename(__name__, "data/symmetric_levy.npz")
FILE_POSITIVE_LEVY = pkg_resources.resource_filename(__name__, "data/positive_levy.npz")


def _assert_has_levy(function):
    if not HAS_LEVY:
        raise ImportError(
            "levy should be installed to use {}.\n".format(function) + 
            "try `pip install pylevy`."
        )


class Interpolator:
    def __init__(self, filename):
        self.filename = filename
        self._interpolator = None

    def __call__(self, *args):
        if self._interpolator is None:
            self._load()
        return self._call(*args)
    
    def _load(self):
        raise NotImplementedError
    
    def _call(self, *args):
        raise NotImplementedError


class LevyInterpolator(Interpolator):
    def _load(self):
        npzfile = np.load(self.filename)
        data = npzfile['data']
        x = npzfile['x']
        alpha = npzfile['alpha']
        x_asymp = npzfile['x_asymp']
        self._interp_xasymp = interpolate.RegularGridInterpolator(
            (alpha, ), x_asymp, method='linear', bounds_error=False, fill_value=None
        )
        self._interpolator = interpolate.RegularGridInterpolator(
            (alpha, x), data, method='linear', bounds_error=False, fill_value=None
        )

class SymmetricLevyInterpolator(LevyInterpolator):
    def __init__(self):
        super().__init__(FILE_SYMMETRIC_LEVY)
    
    def _call(self, x, alpha):
        x = np.abs(x)
        alphax = np.stack(np.broadcast_arrays(alpha, x), axis=-1)
        value = self._interpolator(alphax)
        x_asymp = self._interp_xasymp(x)
        return np.where(x < x_asymp, value, self._asymp(x, alpha))

    def _asymp(self, x, alpha):
        # asymptotic expansions
        return np.sin(np.pi * alpha / 2) * special.gamma(alpha + 1) / np.pi / x**(1 + alpha)

symmetricLevyInterpolator = SymmetricLevyInterpolator()


class PositiveLevyInterpolator(LevyInterpolator):
    def __init__(self):
        super().__init__(FILE_POSITIVE_LEVY)
    
    def _call(self, x, alpha):
        alphax = np.stack(np.broadcast_arrays(alpha, x), axis=-1)
        value = self._interpolator(alphax)
        x_asymp = self._interp_xasymp(x)
        return np.where(x < x_asymp, value, _positive_levy_asymptotic(x, alpha, n=3))


positiveLevyInterpolator = PositiveLevyInterpolator()


def build_levy():
    """
    Build a database for levy pdf
    """
    _build_symmetric_levy(n_alpha=301, x_min=1e-8, x_max=1e8, n_x=1001, rtol=1e-4)
    _build_positive_levy(n_alpha=301, x_min=1e-8, x_max=1e8, n_x=1001, rtol=1e-4)


def _build_symmetric_levy(n_alpha, x_min, x_max, n_x, rtol):
    """
    Precompute the pdf of symmetric levy distribution.
    """
    alphas = np.linspace(0, 2, n_alpha+1)[1:]
    x = np.concatenate([[0], np.logspace(np.log10(x_min), np.log10(x_max), n_x)])

    data = []
    x_asymp = []
    for alpha in alphas:
        computed = stats.levy_stable.pdf(x, alpha, beta=0)
        asymp = np.sin(np.pi * alpha / 2) * special.gamma(alpha + 1) / np.pi / x**(1 + alpha)
        merged = np.array(computed)
        
        # to avoid the instability, we replace the instable part by asymptotic expansions
        idx = np.abs(np.diff(np.log(computed)) / np.diff(np.log(x))) > 3
        idx = np.concatenate([idx[1:], [False, False]])
        idx[x < 10] = False

        # if approaching to the asymptotic, replace it
        idx += (x > 10) * (np.abs(np.log(computed / asymp)) < rtol)
        if np.sum(idx) > 0:
            min_idx = np.min(np.arange(n_x + 1)[idx])
            merged[min_idx:] = asymp[min_idx:]
            x_asymp.append(x[min_idx])
        else:
            x_asymp.append(x[-1])

        idx = (x < 0.3) * (np.abs(np.log(computed / computed[0])) < rtol)
        if np.sum(idx) > 0:
            max_idx = np.max(np.arange(n_x + 1)[idx])
            merged[:max_idx] = computed[0]
        data.append(merged)    

    np.savez(FILE_SYMMETRIC_LEVY, data=data, x=x, alpha=alphas, x_asymp=x_asymp)


def _positive_levy_asymptotic(x, alpha, n=None):
    """
    Asymptotic of positive levy
    """
    if n is None:
        return np.sin(alpha * np.pi) * special.gamma(alpha + 1) * x**(-alpha - 1) / np.pi
    else:
        x = x[..., np.newaxis]
        alpha = np.array(alpha)[..., np.newaxis]
        n = np.arange(1, n+1)
        mat = (-1)**n * np.sin(n * alpha * np.pi) / special.gamma(n + 1) * special.gamma(alpha * n + 1) * x**(-alpha * n - 1)
        return -np.nansum(mat, axis=-1) / np.pi


def _build_positive_levy(n_alpha, x_min, x_max, n_x, rtol):
    """
    Precompute the pdf of symmetric levy distribution.
    """
    alphas = 1 - np.linspace(0, 1, n_alpha + 1, endpoint=False)[1:]**2
    alphas = np.sort(alphas)
    x = np.concatenate([
        np.logspace(np.log10(x_min), np.log10(0.3), n_x // 3, endpoint=False),
        np.logspace(np.log10(0.3), np.log10(3), n_x // 3, endpoint=False),
        np.logspace(np.log10(3), np.log10(x_max), n_x // 3)
    ])
    n_x = x.size

    data = []
    x_asymp = []
    for alpha in alphas:
        scale = np.cos(0.5 * np.pi * alpha)**(1 / alpha)
        computed = stats.levy_stable.pdf(x / scale, alpha, beta=1) / scale
        asymp = _positive_levy_asymptotic(x, alpha, n=3000)
        computed = np.where(np.isfinite(computed), computed, asymp)
        # asymp = _positive_levy_asymptotic(x, alpha, n=3)
        merged = np.array(computed)
        
        # to avoid the instability, we replace the instable part by asymptotic expansions
        idx = ~(np.abs(np.diff(np.log(computed)) / np.diff(np.log(x))) < 5)
        idx = np.concatenate([idx[1:], [False, False]])
        #idx += ~(computed > 0)
        idx[x < 2] = False

        # if approaching to the asymptotic, replace it
        idx += (x > 10) * (np.abs(np.log(computed / asymp)) < rtol)
        if np.sum(idx) > 0:
            min_idx = np.min(np.arange(n_x)[idx])
            merged[min_idx:] = asymp[min_idx:]
            x_asymp.append(x[min_idx])
        else:
            x_asymp.append(x[-1])
        data.append(merged) 

    np.savez(FILE_POSITIVE_LEVY, data=data, x=x, alpha=alphas, x_asymp=x_asymp)


def symmetric_stable(x, alpha, method='interpolate', options=None):
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
    if method.lower() == 'interpolate':
        return symmetricLevyInterpolator(x, alpha)
    if method.lower() == 'pylevy':
        _assert_has_levy('symmetric_stable')
        return levy.levy(x, alpha, beta=0, mu=0.0, sigma=1.0, cdf=False)
    if method.lower() == 'mixture':
        return _symmetric_stable_mixture(x, alpha, options)


def positive_stable(x, alpha, method='interpolate', options=None):
    """
    Levy's alpha stable distribution with beta=1.

    Parameters
    ----------
    method:
        computation method. One of ['scipy' | 'interpolate' | 'pylevy']
    """
    if options is None:
        options = {}
    if method.lower() == 'scipy':
        scale = np.cos(0.5 * np.pi * alpha)**(1 / alpha)
        return stats.levy_stable.pdf(x / scale, alpha, beta=1, **options) / scale
    if method.lower() == 'interpolate':
        return positiveLevyInterpolator(x, alpha)
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
    return np.abs(alpha) / special.gamma(r) * x**(alpha * r - 1) * np.exp(-x**alpha)


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
