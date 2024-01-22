import numpy as np
from scipy import stats, integrate, signal, special
import pytest
from pyspectra import profiles


@pytest.mark.parametrize("sigma", [1, 2, 0.5, 0.1])
def test_sinc_gauss(sigma):
    xmax = np.maximum(10.0, sigma * 20)
    num = 100000
    x = np.linspace(-xmax, xmax, num=num)
    dx = x[1] - x[0]

    sinc = profiles.sinc(x)
    gauss = profiles.Gauss(x, 1, 0, sigma, 0)
    expected = signal.convolve(sinc, gauss, 'same') * dx
    actual = profiles.sinc_gauss(x, sigma)
    
    assert np.allclose(expected[num//4: -num//4], actual[num//4: -num//4], atol=1e-3)


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
