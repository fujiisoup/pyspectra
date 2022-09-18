from multiprocessing import Pool
import numpy as np
from scipy import integrate, interpolate, stats, special

import pkg_resources

try:
    import levy
    HAS_LEVY = True
except ImportError:
    HAS_LEVY = False


# the below is only be used for multiprocessing
def _worker_init(func):
    global _func
    _func = func
  
def _worker(x):
    return _func(x)

def _xmap(func, iterable, processes=None):
    with Pool(processes, initializer=_worker_init, initargs=(func,)) as p:
        return p.map(_worker, iterable)


FILE_SYMMETRIC_LEVY = pkg_resources.resource_filename(__name__, "data/symmetric_levy.npz")
FILE_POSITIVE_LEVY = pkg_resources.resource_filename(__name__, "data/positive_levy.npz")

def _assert_has_levy(function):
    if not HAS_LEVY:
        raise ImportError(
            "levy should be installed to use {}.\n".format(function) + 
            "try `pip install pylevy`."
        )


def normal(x):
    return 1 / np.sqrt(2.0 * np.pi) * np.exp(-0.5 * x ** 2)


class Interpolator:
    def __init__(self, filename):
        self.filename = filename
        self._interpolator = None

    def __call__(self, *args, **kwargs):
        if self._interpolator is None:
            self._load()
        return self._call(*args, **kwargs)
    
    def _load(self):
        raise NotImplementedError
    
    def _call(self, *args, **kwargs):
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


def _build_interp():
    """
    Build a database for some special distributions
    """
    #_build_symmetric_levy(n_alpha=301, x_min=1e-8, x_max=1e8, n_x=1001, rtol=1e-4)
    #_build_positive_levy(n_alpha=301, x_min=1e-8, x_max=1e8, n_x=1001, rtol=1e-4)
    #_build_generalized_mittagleffler(n_alpha=81, n_nu=140, x_min=1e-5, x_max=1e3, n_x=51)
    _build_generalized_mittagleffler_vdf(n_alpha=81, n_nu=140, x_min=1e-5, x_max=1e3, n_x=51)
    

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


FILE_GENERALIZED_MITTAG_LEFFLER = pkg_resources.resource_filename(__name__, "data/generalized_mittagleffler.npz")
FILE_GENERALIZED_MITTAG_LEFFLER_VDF = pkg_resources.resource_filename(__name__, "data/generalized_mittagleffler_vdf.npz")


def _build_generalized_mittagleffler(n_alpha, n_nu, x_min, x_max, n_x):
    alpha = 1 - np.logspace(-3, 0, base=10, num=n_alpha)[:-1][::-1]
    alpha = np.sort(np.concatenate([alpha, [1]]))
    nu = np.sort(1 / np.logspace(0, np.log10(30), base=10, num=n_nu))
    n_xhalf = (n_x - 1) // 2
    n_xquad = (n_x - n_xhalf) // 2
    x = np.concatenate([
        np.logspace(np.log10(x_min), -1, num=n_xquad, base=10, endpoint=False),
        np.logspace(-1, 1.3, num=n_xhalf, base=10, endpoint=False),
        np.logspace(1.3, np.log10(x_max), num=n_xquad+1, base=10, endpoint=True),
    ])
    # actual computation
    def build(alp):
        func = np.zeros((len(nu), len(x)))
        for j in range(len(nu)):
            n = nu[j]
            if alp == 1:
                func[j] = x**(1/n - 1) * np.exp(-x) / special.gamma(1 / n)
            else:
                func[j] = generalizedMittagLeffler_ExponentialMixture.quad(
                    x, gamma=alp, delta=n)
        return func

    func = np.stack(_xmap(build, alpha), axis=0)
    np.savez(FILE_GENERALIZED_MITTAG_LEFFLER, data=func, x=x, gamma=alpha, delta=nu)


def _build_generalized_mittagleffler_vdf(n_alpha, n_nu, x_min, x_max, n_x):
    alpha = 1 - np.logspace(-3, 0, base=10, num=n_alpha)[:-1][::-1]
    alpha = np.sort(np.concatenate([alpha, [1]]))
    powers = np.linspace(0, 1/2, 7)
    n_xhalf = (n_x - 1) // 2
    n_xquad = (n_x - n_xhalf) // 2
    x = np.concatenate([
        np.logspace(np.log10(x_min), -1, num=n_xquad, base=10, endpoint=False),
        np.logspace(-1, 1.3, num=n_xhalf, base=10, endpoint=False),
        np.logspace(1.3, np.log10(x_max), num=n_xquad+1, base=10, endpoint=True),
    ])
    # actual computation
    def build(alp):
        func = np.zeros((len(powers), len(x)))
        for j in range(len(powers)):
            power = powers[j]
            func[j] = generalizedMittagLefflerVDF_ExponentialMixture.quad(
                x, gamma=alp, delta=3/2/alp, power=power)
        return func

    func = np.stack(_xmap(build, alpha), axis=0)
    np.savez(FILE_GENERALIZED_MITTAG_LEFFLER_VDF, data=func, x=x, gamma=alpha, power=powers)


class ScaleMixture:
    r"""
    A class to make a discrete approximation of the scale mixture.
    Scale mixture of the two distribution
    \int f(x/s) /s g(s) ds

    Parameters
    ----------
    x: nd-array
        destination at which the mixture distribution is computed
    scales: 1d-array or nd-array
    src_dist: callable 
        source distribution f(x / s)
    weight_dist: callable   
        weight distribution g(s)
    x_asymp_min, x_asymp_max: 
        Asymptotic form is used for the smaller / larger x than these values.
        For None, no asymptotics are used.
    asymp_min, asymp_max: callable
        Asymptotic form is used for the smaller / larger x than these values.
    """
    def __call__(self, x, n_points, *args, **kwargs):
        x = np.asarray(x)
        scales = self.scale_func(x, n_points, *args, **kwargs)
        values = integrate.trapz(
            self.src_dist(x[..., np.newaxis], scales, *args, **kwargs) * 
            self.weight_dist(scales, *args, **kwargs), 
            x=scales,
        axis=-1)
        if getattr(self, 'x_asymp_min', None) is not None:
            values = np.where(
                x < self.x_asymp_min(*args, **kwargs), 
                self.asymp_min(x, *args, **kwargs), values)
        if getattr(self, 'x_asymp_max', None) is not None:
            values = np.where(
                x > self.x_asymp_max(*args, **kwargs), 
                self.asymp_max(x, *args, **kwargs), values)
        return values

    def src_dist(self, x, scales, *args, **kwargs):
        raise NotImplementedError

    def weight_dist(self, x, *args, **kwargs):
        raise NotImplementedError

    scale_range = (0, np.infty)  # integration range of the mixture 

    def quad(self, x, *args, **kwargs):
        """More accurate calculation based on the numerical integration"""
        x = np.asarray(x)
        shape = x.shape
        x = x.ravel()
        values = np.zeros(x.shape)
        errors = np.zeros(x.shape)
        for i, x1 in enumerate(x):
            res = integrate.quad(
                lambda s: self.src_dist(x1, s, *args, **kwargs) * self.weight_dist(s, *args, **kwargs),
                self.scale_range[0], self.scale_range[1],
            )
            values[i] = res[0]
            errors[i] = np.abs(res[1])

        if getattr(self, 'x_asymp_min', None) is not None:
            values_min = self.asymp_min(x, *args, **kwargs)
            values = np.where(
                (x < self.x_asymp_min(*args, **kwargs)) * 
                (values_min < errors) * np.isfinite(values_min), 
                values_min,
                values
            )
        if getattr(self, 'x_asymp_max', None) is not None:
            values_max = self.asymp_max(x, *args, **kwargs)
            values = np.where(
                (x > self.x_asymp_max(*args, **kwargs)) * 
                (values_max < errors) * np.isfinite(values_max), 
                values_max, 
                values
            )
        return values.reshape(shape)


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


class SymmetricStable_ExponentialMixture(ScaleMixture):
    def scale_func(self, x, n_points, alpha):
        # TODO optimize scale
        vmax = 1000
        log_vmin = ((alpha - 2) * 10 + 3) / 2
        sigma_max = np.sqrt(vmax) / 3
        return np.logspace(log_vmin, np.log10(vmax), base=10, num=n_points+1)[1:]

        
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


def generalized_mittag_leffler(
    x, alpha, nu, method='interp', options=None
):
    r"""
    A generalization of mittag_leffler distribution.

    Its laplace distribution is 
    1 / (1 + s^\alpha)^\nu
    """
    if options is None:
        options = {}

    if method == 'interp':
        return _generalized_mittag_leffler_interp(x, gamma=alpha, delta=1 / nu)

    if method == 'exponential_mixture':
        n_points = options.get('num_points', 31)
        return generalizedMittagLeffler_ExponentialMixture(
            x, n_points, gamma=alpha, delta=1/nu
        )
    if method == 'quad':

        return generalizedMittagLeffler_ExponentialMixture.quad(
            x, gamma=alpha, delta=1/nu
        )


def generalized_mittag_leffler_vdf(
    x, alpha, nu, power=0, method='interp', options=None
):
    r"""
    Velocity distribution for the generalized mittag-leffler
    """
    if options is None:
        options = {}

    if method == 'interp':
        return _generalized_mittag_leffler_interp(x, gamma=alpha, delta=1 / nu, power=power)

    if method == 'quad':
        return generalizedMittagLeffler_ExponentialMixture.quad(
            x, gamma=alpha, delta=1/nu, power=power
        )

def arccot(x):
    return np.pi / 2 - np.arctan(x)

class GeneralizedMittagLeffler_ExponentialMixture(ScaleMixture):
    r"""
    Generalized Mittag Leffler distribution.
    The laplace transform is

    (1 + delta s^gamma)^(-1/delta)

    Parameterization presented in 
    
    "A new family of tempered distributions"
    Barabesi et al
    
    is used.
    """
    def __call__(self, x, *args, **kwargs):
        raise NotImplementedError("The optimum is not set yet. Wait for the implementation")

    def src_dist(self, x, scales, delta, gamma, *args, **kwargs):
        return np.exp(-x / scales) / scales

    def weight_dist(self, x, delta, gamma, *args, **kwargs):
        y = 1 / x

        pi_g = np.pi * gamma
        y_g = y**gamma
        Fg = 1 - 1 / pi_g * arccot(1 / np.tan(pi_g) + y**gamma / np.sin(pi_g))

        value = np.sin(pi_g * Fg / delta) / (y_g**2 + 2 * y_g * np.cos(pi_g) + 1)**(0.5 / delta)
        return value * y / np.pi
    
    def scale_func(self, x, n_points, delta, gamma):
        a = self._interp_a((delta, gamma))
        b = self._interp_b((delta, gamma))
        c = self._interp_c((delta, gamma))
        p = np.linspace(-1, 1, n_points)
        return np.exp(a + b * p + c * p**3)

    def x_asymp_min(self, delta, gamma, *args, **kwargs):
        return 0.2 / delta

    def asymp_min(self, x, delta, gamma, *args, **kwargs):
        # Taylor expansion
        order = 80
        res = np.zeros(x.shape)
        coef = 1
        for k in range(order):
            an = gamma / delta + k * gamma
            res += coef * x**(an - 1) / special.gamma(an)
            coef *= -(1 / delta + k) / (k + 1)
        return res
        
    def x_asymp_max(self, delta, gamma, *args, **kwargs):
        return 3 / delta

    def asymp_max(self, x, delta, gamma, *args, **kwargs):
        return x**(-1-gamma) / special.gamma(1-gamma) * gamma / delta

'''
    def __init__(self):
        npzfile = np.load(FILE_GMITTAGLEFFLER)
        gamma = npzfile['alpha'][::-1]
        delta = npzfile['delta'][::-1]
        a = npzfile['a'][::-1, ::-1]
        b = npzfile['b'][::-1, ::-1]
        c = npzfile['c'][::-1, ::-1]
        xmin = npzfile['xmin'][::-1, ::-1]
        xmax = npzfile['xmax'][::-1, ::-1]
        self._interp_a = interpolate.RegularGridInterpolator(
            (delta, gamma), a, method='linear', bounds_error=False, fill_value=None
        )
        self._interp_b = interpolate.RegularGridInterpolator(
            (delta, gamma), b, method='linear', bounds_error=False, fill_value=None
        )
        self._interp_c = interpolate.RegularGridInterpolator(
            (delta, gamma), c, method='linear', bounds_error=False, fill_value=None
        )
        self._interp_xmin = interpolate.RegularGridInterpolator(
            (delta, gamma), xmin, method='linear', bounds_error=False, fill_value=None
        )
        self._interp_xmax = interpolate.RegularGridInterpolator(
            (delta, gamma), xmax, method='linear', bounds_error=False, fill_value=None
        )
'''


generalizedMittagLeffler_ExponentialMixture = GeneralizedMittagLeffler_ExponentialMixture()


class GeneralizedMittagLefflerVDF_ExponentialMixture(GeneralizedMittagLeffler_ExponentialMixture):
    def __call__(self, x, *args, **kwargs):
        raise NotImplementedError("The optimum is not set yet. Wait for the implementation")

    def src_dist(self, x, scales, delta, gamma, power=0.0, **kwargs):
        """
        power: correction in the power
        """
        xscale = np.sqrt(2 * scales)
        return (
            special.gammaincc(1 / 2 + power, (x / xscale)**2) * special.gamma(1 / 2 + power)
            / xscale / special.gamma(1 + power)
        ) * scales**power

generalizedMittagLefflerVDF_ExponentialMixture = GeneralizedMittagLefflerVDF_ExponentialMixture()


class Gamma_ExponentialMixture(ScaleMixture):
    def scale_func(self, x, n_points, gamma, *args, **kwargs):
        # TODO optimize more
        # return np.linspace(0, 1, num=n_points + 1, endpoint=False)[1:]**(0.5 / gamma)
        pts = np.linspace(0, 1, num=n_points + 1, endpoint=False)[1:]
        return pts**gamma 
        
    def src_dist(self, x, scales, *args, **kwargs):
        return np.exp(-x / scales) / scales

    def weight_dist(self, x, gamma, *args, **kwargs):
        coef = np.sin(np.pi * gamma) / np.pi
        return coef * x**(gamma - 1) * (1 - x)**(-gamma)

    scale_range = (0, 1)
    

gamma_ExponentialMixture = Gamma_ExponentialMixture()


def generalized_gamma(x, r, alpha):
    """
    generalized gamma distribution
    """
    return np.abs(alpha) / special.gamma(r) * x**(alpha * r - 1) * np.exp(-x**alpha)


class GeneralizedMittagLefflerInterp(Interpolator):
    def _load(self):
        npzfile = np.load(self.filename)
        data = np.log(npzfile['data'])
        x = np.log(npzfile['x'])
        gamma = npzfile['gamma']
        delta = npzfile['delta']
        self._interpolator = interpolate.RegularGridInterpolator(
            (gamma, delta, x), data, method='linear', bounds_error=False, fill_value=None
        )

    def _call(self, x, gamma, delta):
        x = np.log(np.abs(x))
        gammax = np.stack(np.broadcast_arrays(gamma, delta, x), axis=-1)
        return np.exp(self._interpolator(gammax))


_generalized_mittag_leffler_interp = GeneralizedMittagLefflerInterp(FILE_GENERALIZED_MITTAG_LEFFLER)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'build':
        _build_interp()