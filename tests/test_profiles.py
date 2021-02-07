import numpy as np
from scipy import stats
import pytest
from pyspectra import profiles


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
