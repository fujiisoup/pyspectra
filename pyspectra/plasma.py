import numpy as np
from scipy import special
from . import constants
from . import units


def bremsstrahlung(lam, Te, ne, Zeff, approximation='Bekefi'):
    r''' Bremsstrahlung intensity. 

    Parameters
    ----------
    lam: wavelength in nm
    Te: electron temperature in eV
    ne: electron density in m-3
    Zeff: effective charge
    approximation: method of approximation. One of ['Bekefi', 'Ichimaru', ...]

    Returns
    -------
    intensity per wavelength [W/nm], integrated over the 4-pi solid angle.
    '''
    if approximation == 'Bekefi':
        return _bremsstrahlung_bekefi(lam, Te, ne, Zeff)
    
    raise NotImplementedError(
        'Approximation method {} is not implemented.'.format(approximation)
    )


def _bremsstrahlung_bekefi(lam, Te, ne, Zeff):
    ebar = constants.e / np.sqrt(4 * np.pi * constants.eps0)
    # square of plasma frequency
    wp_square = ne * constants.e**2 / (constants.eps0 * constants.me)
    # square of photon frequency
    w_square = units.nm_to_Hz(lam)**2
    # unitless quantity, 
    # assuming Te > Zeff**2 Eh with Eh = 27.2 eV
    y = 0.5 * (
        units.nm_to_eV(lam) / Te
    )**2

    dPdw = (
        8 * np.sqrt(2) / (3 * np.sqrt(np.pi)) * 
        ebar**6 / (constants.me * constants.c**2)**(1.5) * 
        np.sqrt(1 - wp_square / w_square) * 
        Zeff**2 * ne**2 / np.sqrt(units.eV_to_joule(Te)) *
        special.exp1(y)
    )
    dPdlam = dPdw * constants.c * 1e9 / lam**2
    return dPdlam
