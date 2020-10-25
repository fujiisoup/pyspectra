from pyspectra import data


def test_diatomic_molecule():
    ds = data.diatomic_molecules('I2', force_download=True)
    ds2 = data.diatomic_molecules('I2')

    assert ds == ds2


def test_atom():
    ds = data.atom('Li', force_download=True)
    ds2 = data.atom('Li', force_download=False)
    
    assert ds == ds2