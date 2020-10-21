from pyspectra import data


def test_diatomic_molecule():
    ds = data.diatomic_molecules('I2', force_download=True)
    