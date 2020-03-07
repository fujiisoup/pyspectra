import numpy as np
import pytest
from pyspectra import fit, profiles


@pytest.mark.parametrize(('n', 'sn', 'x0', 'width'),
[
    (100, 100.0, 0.0, 10.0),
    (100, 30.0, 0.0, 10.0),
    (100, 100.0, 0.0, 30.0),
    (100, 100.0, 30.0, 30.0),
    (1000, 10.0, 30.0, 30.0),
])
@pytest.mark.parametrize('seed', [0, 1])
def test_gaussian(n, sn, x0, width, seed):
    rng = np.random.RandomState(seed)
    x = np.linspace(-100, 100, n)
    A = sn * width * np.sqrt(2 * np.pi)
    y = profiles.Gauss(x, A, x0, width, 0) + rng.randn(n)

    popt, perr = fit.singlepeak_fit(x, y)
    
    expected = np.array((A, x0, width, 0))
    assert ((popt - 5 * perr < expected) * (expected < popt + 5 * perr)).all()
