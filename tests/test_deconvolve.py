import numpy as np
import pytest
from pyspectra import deconvolve, profiles


@pytest.mark.parametrize(('n', 'm'), [
    (5, 2), (1000, 300)
])
def test_convolve(n, m):
    # make sure pyspectra.deconvolve.convolve 
    # gives the same results with np.convolve
    # template = np.random.randn(m)
    template = np.arange(1, m + 1)
    x = np.random.randn(n - m + 1)
    actual = deconvolve.convolve(x, template)
    expected = np.convolve(x, template, mode='full')
    assert np.allclose(actual, expected)


def test_nnls():
    template = np.array([1] + [0] * 10 + [0.5])
    x = profiles.Lorentz(
        np.linspace(-1, 1, 101), 
        1.0, 0.0, 0.1, 0.0)
    y_true = np.convolve(x, template, mode='full')
    actual = deconvolve.nnls(y_true, template)
    assert np.allclose(actual, x, atol=1e-3)

    rng = np.random.RandomState(0)
    y = y_true + 0.05 * rng.randn(*y_true.shape)
    actual = deconvolve.nnls(y, template)
    
    # test weight
    sigma = np.sqrt(y_true)
    actual = deconvolve.nnls(y, template, sigma=sigma)
    
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(y_true, '-')
    plt.plot(y, '.')
    
    plt.subplot(1, 2, 2)
    plt.plot(actual)
    plt.plot(x)
    plt.show()
    """


@pytest.mark.parametrize(('type', 'alpha'), [
    ('ridge', 1e-3), ('lasso', 1e-3)
])
def test_nnls_reg(type, alpha):
    template = np.array([1] + [0] * 10 + [0.5])
    x = profiles.Gauss(
        np.linspace(-1, 1, 101), 
        1.0, 0.0, 0.1, 0.0)
    y_true = np.convolve(x, template, mode='full')
    
    rng = np.random.RandomState(0)
    y = y_true + 0.1 * rng.randn(*y_true.shape)
    actual = deconvolve.nnls(
        y, template, 
        regularization_type=type, 
        regularization_parameter=alpha,
        )
    
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(y_true, '-')
    plt.plot(y, '.')
    
    plt.subplot(1, 2, 2)
    plt.plot(actual)
    plt.plot(x)
    plt.show()
    """
