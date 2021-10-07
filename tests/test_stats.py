import numpy as np
from scipy import special, stats, integrate
import pytest
import pyspectra

def laplace_transform(s, func, a=0, b=np.infty):
    if hasattr(s, '__len__'):
        return np.array([
            laplace_transform(s1, func, a, b) for s1 in s
        ])

    def f(x, s):
        return func(x) * np.exp(-s * x)
    return integrate.quad(f, a, b, args=(s))[0]


@pytest.mark.parametrize("alpha", [1.2, 1.5, 1.9, 1.95])
def _test_symmetric_stable(alpha):
    x = np.logspace(-1, 1, num=100)
    expected = pyspectra.stats.symmetric_stable(x, alpha=alpha, method='scipy')
    actual = pyspectra.stats.symmetric_stable(x, alpha=alpha, method='interpolate')
    assert np.allclose(expected, actual, atol=1e-5, rtol=1e-2)


@pytest.mark.parametrize("alpha", [0.4, 0.8, 0.85, 0.9, 0.99])
def _test_positive_stable(alpha):
    x = np.logspace(-1, 1, num=100)
    expected = pyspectra.stats.positive_stable(x, alpha=alpha, method='scipy')
    actual = pyspectra.stats.positive_stable(x, alpha=alpha, method='interpolate')

    idx = slice(None)
    if alpha > 0.95:
        pass
    elif 0.85 < alpha:
        idx = (0.8 < x) * (x < 3)
        assert np.allclose(expected[idx], actual[idx], atol=1e-5, rtol=0.1)
    elif 0.7 < alpha:
        idx = (0.8 < x) * (x < 3)
        assert np.allclose(expected[idx], actual[idx], atol=1e-5, rtol=0.03)
    else:
        assert np.allclose(expected[idx], actual[idx], atol=1e-5, rtol=0.01)


@pytest.mark.parametrize("alpha", [0.4, 0.9, 0.99])
@pytest.mark.parametrize("nu", [1.4, 1.1])
def test_generalized_mittag_leffler_laplace(alpha, nu):
    s = np.logspace(-2, 2, num=31)
    actual_laplace = laplace_transform(
        s, lambda x: pyspectra.stats.generalized_mittag_leffler(
            x, alpha=alpha, nu=nu, method='interp'
        )
    )
    expected_laplace = 1 / (1 + s**alpha)**nu

    x = np.logspace(-4.5, 3, num=31)
    actual = pyspectra.stats.generalized_mittag_leffler(
            x, alpha=alpha, nu=nu, method='interp')
    integ = pyspectra.stats.generalizedMittagLeffler_ExponentialMixture.quad(
            x, gamma=alpha, delta=1/nu)
    
    '''
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.loglog(x, actual, label='actual')
    plt.loglog(x, integ, label='integ')

    plt.subplot(1, 2, 2)
    plt.loglog(s, actual_laplace, label='actual')
    plt.loglog(s, expected_laplace, '--', label='expected')
    plt.show()
    
    assert np.allclose(actual_laplace, expected_laplace, rtol=1e-2)
    assert np.allclose(actual, integ, rtol=1e-2)
    '''


@pytest.mark.parametrize("alpha", [0.4, 0.9, 0.99])
@pytest.mark.parametrize("nu", [1.4, 1.1])
def _test_generalized_mittag_leffler_vdf(alpha, nu):
    s = np.logspace(-2, 2, num=31)
    x = np.logspace(-4.5, 3, num=31)
    actual = pyspectra.stats.generalized_mittag_leffler_vdf(
            x, alpha=alpha, nu=nu, method='interp')
    integ = pyspectra.stats.generalizedMittagLefflerVDF_ExponentialMixture.quad(
            x, gamma=alpha, delta=1/nu)
    

@pytest.mark.parametrize("alpha", [0.4, 0.9])
def _test_generalized_mittag_leffler_compare_ggamma_mixture(alpha):
    nu = 1.5
    # make sure the distribution with nu = 1 is same with the usual mittag_leffler
    x = np.logspace(-2, 2, num=101)
    actual = pyspectra.stats.generalized_mittag_leffler(
        x, alpha=alpha, nu=nu, method='exponential_mixture', 
        options={'num_points': 301}
    )
    expected = generalized_mittag(x, alpha=alpha, nu=1.5)

    s = np.logspace(-3, 3, num=31)
    actual_laplace = laplace_transform(
        s, lambda x: pyspectra.stats.generalized_mittag_leffler(
            x, alpha=alpha, nu=nu, method='exponential_mixture',
            options={'num_points': 301}
        )
    )
    expected_laplace = 1 / (1 + s**alpha)**nu
    #assert np.allclose(actual_laplace, expected_laplace, rtol=1e-2)  # TODO increase the accuracy

    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.loglog(x, actual, label='actual')
    plt.loglog(x, expected, '--', label='expected')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.loglog(s, actual_laplace, label='actual')
    plt.loglog(s, expected_laplace, '--', label='actual')
    plt.show()
    
    assert (actual >= 0).all()


@pytest.mark.parametrize("gamma", [0.1, 0.5, 0.85])
def test_gamma_as_exponential_mixture_quad(gamma):
    x = np.logspace(-1, 2, num=101)
    numerical = pyspectra.stats.gamma_ExponentialMixture.quad(x, gamma)
    expected = x**(gamma - 1) / special.gamma(gamma) * np.exp(-x)
    assert np.allclose(numerical, expected, rtol=1e-3)


@pytest.mark.parametrize("gamma", [0.1, 0.5, 0.85])
def _test_gamma_as_exponential_mixture(gamma):
    x = np.logspace(-2, 2, num=101)
    actual = pyspectra.stats.gamma_ExponentialMixture(x, 51, gamma)
    numerical = pyspectra.stats.gamma_ExponentialMixture.quad(x, gamma)
    expected = x**(gamma - 1) / special.gamma(gamma) * np.exp(-x)

    import matplotlib.pyplot as plt
    plt.loglog(x, actual, label='actual')
    plt.loglog(x, numerical, label='numerical')
    plt.loglog(x, expected, '--', label='expected')
    plt.legend()
    plt.show()

