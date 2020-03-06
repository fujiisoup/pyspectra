import numpy as np
import pytest
from pyspectra import units


@pytest.mark.parametrize('seed', [0, 1, 2])
def test(seed):
    rng = np.random.RandomState(seed)

    x = rng.randn(100)
    assert np.allclose(x, units.eV_to_cm(units.cm_to_eV(x)))

    x = rng.randn(3)
    actual = units.nm_to_eV(units.cm_to_nm(units.eV_to_cm(x)))
    assert np.allclose(x, actual)
