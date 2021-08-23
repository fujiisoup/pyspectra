import numpy as np
from scipy import stats, integrate
import pytest
from pyspectra import profiles


def laplace_transform(s, func, a=0, b=np.infty):
    if hasattr(s, '__len__'):
        return np.array([
            laplace_transform(s1, func, a, b) for s1 in s
        ])

    def f(x, s):
        return func(x) * np.exp(-s * x)
    return integrate.quad(f, a, b, args=(s))[0]


@pytest.mark.parametrize("alpha", [0.4, 0.9])
def _test_symmetric_geostable_laplace(alpha):
    s = np.logspace(-2, 2, base=10, num=31)
    actual = laplace_transform(
        s, lambda x: profiles.mittag_leffler(x, alpha=alpha, method='mixture', options={'num_points': 1001}),
    )
    expected = 1 / (1 + s**alpha)
    assert np.allclose(actual, expected, rtol=1e-1)  # TODO increase the accuracy
    
    '''
    import matplotlib.pyplot as plt
    plt.loglog(s, actual, label='actual')
    plt.loglog(s, expected, label='expected')
    plt.legend()
    plt.show()
    '''


@pytest.mark.parametrize("alpha", [0.8, 0.9])
def test_generalized_mittag_leffler(alpha):
    # make sure the distribution with nu = 1 is same with the usual mittag_leffler
    x = np.logspace(-3, 3, num=100)
    expected = profiles.mittag_leffler(x, alpha=alpha, method='mixture', options={'num_points': 301})
    actual = profiles.generalized_mittag_leffler(
        x, alpha=alpha, nu=1, method='mixture', options={'num_points': 301})

    import matplotlib.pyplot as plt
    plt.loglog(x, actual, label='actual')
    plt.loglog(x, expected, label='expected')
    plt.legend()
    plt.show()
    raise ValueError


@pytest.mark.parametrize("alpha", [0.9])
def test_symmetric_geostable(alpha):
    x = np.logspace(-2, 2, num=100)
    actual = profiles.mittag_leffler(x, alpha=alpha, method='mixture', options={'num_points': 301})
    actual2 = profiles.mittag_leffler(x, alpha=alpha, method='mixture', options={'num_points': 602})
    assert (actual != actual2).any()
    assert np.allclose(actual, actual2, rtol=1e-2)


#@pytest.mark.parametrize("alpha", [1.2, 1.5, 1.9, 1.99])
@pytest.mark.parametrize("alpha", [1.9, 1.99])
def test_symmetric_stable(alpha):
    x = np.logspace(-2, 2, num=100)
    expected = profiles.symmetric_stable(x, alpha=alpha, method='scipy')
    actual = profiles.symmetric_stable(x, alpha=alpha, method='pylevy')

    '''
    import matplotlib.pyplot as plt
    plt.loglog(x, expected, label='expected')
    plt.loglog(x, actual, label='actual')
    plt.ylim(1e-6, 1)
    plt.legend()
    plt.show()
    '''

    assert np.allclose(expected, actual, atol=1e-3, rtol=0.1)


@pytest.mark.parametrize("alpha", [0.9, 0.99])
def test_positive_stable(alpha):
    x = np.logspace(-2, 2, num=100)
    expected = profiles.positive_stable(x, alpha=alpha, method='scipy')
    actual = profiles.positive_stable(x, alpha=alpha, method='pylevy')

    import matplotlib.pyplot as plt
    plt.loglog(x, expected, label='expected')
    plt.loglog(x, actual, '--', label='actual')
    plt.ylim(1e-6, 1)
    plt.legend()
    plt.show()

    assert np.allclose(expected, actual, atol=1e-3, rtol=0.1)


@pytest.mark.parametrize("df", [1, 2, 10])
@pytest.mark.parametrize("gamma", [0.1, 1, 10])
def test_generalized_voigt1(df, gamma):
    # compute the convolution numerically
    xmax = 300
    x = np.linspace(-xmax, xmax, num=100000)
    normal_x = stats.norm.pdf(x, loc=0, scale=1)
    t_x = stats.t.pdf(x, loc=0, scale=gamma, df=df)
    convolved = np.convolve(normal_x, t_x, mode="same")
    expected = convolved / np.trapz(convolved, x)
    # semi-analytical formula
    actual = profiles.generalized_voigt1(x, gamma=gamma, df=df)
    print(np.max(np.abs(actual - expected)))
    assert np.allclose(actual, expected, atol=0.005)