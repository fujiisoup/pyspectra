from pyspectra import data


def test_diatomic_molecule():
    ds = data.diatomic_molecules('I2', force_download=True)
    ds2 = data.diatomic_molecules('I2')

    assert ds == ds2


def test_atom_levels():
    ds = data.atom_levels('Li', force_download=True)
    ds2 = data.atom_levels('Li', force_download=False)
    
    assert ds == ds2

def test_atom_lines():
    ds = data.atom_lines('Li', force_download=True)
    ds2 = data.atom_lines('Li', force_download=False)
    
    assert ds == ds2