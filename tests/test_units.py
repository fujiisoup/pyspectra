import numpy as np
import pytest
from pyspectra import units


@pytest.mark.parametrize("seed", [0, 1, 2])
def test(seed):
    rng = np.random.RandomState(seed)

    x = rng.randn(100)
    assert np.allclose(x, units.eV_to_cm(units.cm_to_eV(x)))

    x = rng.randn(3)
    actual = units.nm_to_eV(units.cm_to_nm(units.eV_to_cm(x)))
    assert np.allclose(x, actual)


def test_misc():
    assert np.allclose(units.a0, 5.291772109e-11)


def test_unit():
    check_list = {
        3: 'III', 4: 'IV', 5: 'V', 
        11: 'XI', 31: 'XXXI', 33: 'XXXIII',
    }
    for key, expected in check_list.items():
        actual = units.int_to_roman(key)
        assert actual == expected 
        actual = units.roman_to_int(actual)
        assert actual == key 

    # test with array
    keys = np.array(list(check_list.keys()))
    expected = np.array(list(check_list.values()))
    actual = units.int_to_roman(keys)
    assert np.all(actual == expected)

    # inverse conversion
    actual = units.roman_to_int(actual)
    assert np.all(actual == keys)
