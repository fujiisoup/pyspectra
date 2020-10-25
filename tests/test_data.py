from pyspectra import data


def test_diatomic_molecule():
    ds = data.diatomic_molecules('I2', force_download=True)
    ds2 = data.diatomic_molecules('I2')

    assert ds == ds2


def test_atom_levels():
    ds = data.atom_levels('Li', force_download=True)
    ds2 = data.atom_levels('Li', force_download=False)

    assert 'energy' in ds.coords
    assert ds['energy'].attrs['unit'] == 'eV'
    assert ds['ionization_energy'].attrs['unit'] == 'eV'
    assert ds['ionization_energy_err'].attrs['unit'] == 'eV'
    assert ds == ds2

def test_atom_lines():
    ds = data.atom_lines('Li', force_download=True)
    ds2 = data.atom_lines('Li', force_download=False)
    
    assert 'wavelength' in ds.coords
    assert ds['wavelength'].attrs['unit'] == 'nm(vacuum)'
    assert ds == ds2

    ds = data.atom_lines('Li', unit='nm(air)')
    assert ds['wavelength'].attrs['unit'] == 'nm(air)'
    
    ds = data.atom_lines('Li', unit='eV')
    assert ds['wavelength'].attrs['unit'] == 'eV'
    
    ds = data.atom_lines('Li', unit='cm')
    assert ds['wavelength'].attrs['unit'] == 'cm^{-1}'
