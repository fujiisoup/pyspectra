import pytest
from pyspectra import constants


@pytest.mark.parametrize(
    "symbol", ['H', 'Li', 'Be'],
)
def test_mass(symbol):
    constants.mass(symbol)
