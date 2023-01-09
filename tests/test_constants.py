import pytest
from pyspectra import constants


@pytest.mark.parametrize(
    ("symbol", "mass"), 
    [
        ('H', 1.0079), ('Li', 6.941), ('W',  183.84),
    ],
)
def test_mass(symbol, mass):
    actual = constants.mass(symbol)
    assert actual == mass * constants.mu
