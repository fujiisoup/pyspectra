import numpy as np
from scipy import stats, integrate
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


def generalized_mittag(x, alpha, nu):
    r"""
    Mixture representation of generalized mittag_leffler distribution by 
    generalized-gamma mixture 

    On Mixture Representations for the Generalized Linnik Distribution and Their Applications in Limit Theorems
    Korolev et al.
    Theorem 4
    """
    x = np.array(x)[..., np.newaxis]
    num_points = 301

    delta = alpha
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
        pyspectra.stats.positive_stable(y, alpha=delta, method='scipy') * 
        pyspectra.stats.generalized_gamma(x / (y * scale), nu, delta) / (y * scale) * 
        w, axis=-1)


@pytest.mark.parametrize("alpha", [0.4, 0.9, 0.99])
@pytest.mark.parametrize("nu", [1.4, 0.9])
def test_generalized_mittag_leffler_laplace(alpha, nu):
    s = np.logspace(-2, 2, num=31)
    actual_laplace = laplace_transform(
        s, lambda x: pyspectra.stats.generalized_mittag_leffler(
            x, alpha=alpha, nu=nu, method='exponential_mixture',
            options={'num_points': 301}
        )
    )
    expected_laplace = 1 / (1 + s**alpha)**nu

    x = np.logspace(-4.5, 3, num=101)
    actual = pyspectra.stats.generalized_mittag_leffler(
            x, alpha=alpha, nu=nu, method='exponential_mixture',
            options={'num_points': 301})
    integ = pyspectra.stats.generalizedMittagLeffler_ExponentialMixture.quad(
            x, gamma=alpha, delta=1/nu)
    
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


@pytest.mark.parametrize("alpha", [0.4, 0.9])
def test_generalized_mittag_leffler_compare_ggamma_mixture(alpha):
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


@pytest.mark.parametrize("alpha", [0.9])
def _test_symmetric_geostable(alpha):
    x = np.logspace(-2, 2, num=100)
    actual = pyspectra.stats.mittag_leffler(x, alpha=alpha, method='mixture', options={'num_points': 301})
    actual2 = pyspectra.stats.mittag_leffler(x, alpha=alpha, method='mixture', options={'num_points': 602})
    assert (actual != actual2).any()
    assert np.allclose(actual, actual2, rtol=1e-2)


@pytest.mark.parametrize("alpha", [0.4, 0.9, 0.99])
def _test_symmetric_geostable_laplace(alpha):
    s = np.logspace(-2, 2, base=10, num=31)
    actual = laplace_transform(
        s, lambda x: pyspectra.stats.mittag_leffler(x, alpha=alpha, method='mixture', options={'num_points': 1001}),
    )
    expected = 1 / (1 + s**alpha)
    assert np.allclose(actual, expected, rtol=1e-1)  # TODO increase the accuracy
    


@pytest.mark.parametrize(("alpha", "nu"), [(0.4, 1.5), (0.9, 1.2)])
def _test_generalized_mittag_leffler_laplace(alpha, nu):
    # make sure the distribution with nu = 1 is same with the usual mittag_leffler
    x = np.logspace(-3, 3, num=100)
    actual = pyspectra.stats.generalized_mittag_leffler(
        x, alpha=alpha, nu=nu, method='mixture', options={'num_points': 301})

    import matplotlib.pyplot as plt
    plt.loglog(x, actual, label='actual')
    # plt.loglog(x, expected, '--', label='expected')
    plt.legend()
    plt.show()




