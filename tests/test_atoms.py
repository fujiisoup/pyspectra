import numpy as np
import pytest
from pyspectra import atoms


def test_roman_number():
    assert atoms.ROMAN_NUMBER.index("XX") == 20
    assert atoms.ROMAN_NUMBER.index("XXX") == 30
    assert atoms.ROMAN_NUMBER.index("XL") == 40
    assert atoms.ROMAN_NUMBER.index("L") == 50


def test_decode_charge():
    atom, charge = atoms.decode_charge("FeI")
    assert atom == "Fe"
    assert charge == 0

    atom, charge = atoms.decode_charge("CII")
    assert atom == "C"
    assert charge == 1

    atom, charge = atoms.decode_charge("KrLI")
    assert atom == "Kr"
    assert charge == 50
