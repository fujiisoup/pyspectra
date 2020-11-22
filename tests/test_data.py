import pytest
from pyspectra import data


@pytest.mark.parametrize('molecule', [
    'I2', 'HO', 'H2', 'OH', 'OD',
])
def test_diatomic_molecule(molecule):
    ds = data.diatomic_molecules(molecule, force_download=True)
    ds2 = data.diatomic_molecules(molecule)
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

def test_atom_lines_uncertainty():
    ds = data.atom_lines('Ne', force_download=True)
    assert ds['wavelength_err'].dtype == float

    ds = data.atom_lines('Ne', unit='nm(air)')
    assert ds['wavelength_err'].attrs['unit'] == 'nm(air)'
    assert ds['wavelength_ritz_err'].attrs['unit'] == 'nm(air)'
