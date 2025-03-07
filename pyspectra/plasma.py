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


def thomson_incoherent(wavelength, wavelength_laser, Te, theta, ne=1, laser_power=1):
    r'''Calculate Thomson scattering
    
    Source:
    Matoba, Tohru, Tokiyoshi Itagaki, Toshihiko Yamauchi, and Akimasa Funahashi. 1979. 
    “Analytical Approximations in the Theory of Relativistic Thomson Scattering for 
     High Temperature Fusion Plasma.” 
    Japanese Journal of Applied Physics 18 (6): 1127–33.
    
    Parameters
    ----------
    wavelength: float or 1d array
        Wavelength in nm.
    wavelength_laser: float
        Wavelength of the laser in nm.
    theta: float
        angle of the laser and scattering light in radian.
    Te: float
        Electron temperature in eV.
    ne: float
        Electron density in m-3.
    laser_power: float
        Laser power in W.
    
    Returns
    -------
    float or 1d array
        Scattered power in W.
    '''
    # Eq. 10
    alpha = (
        constants.electron_mass * constants.c**2 /
        (2 * units.eV_to_joule(Te))
    )
    # Eq.15 epsilon = dlambda / lambda 
    epsilon = (wavelength - wavelength_laser) / wavelength_laser
    # Eq. 23
    cos_theta = np.cos(theta)
    alpha_costheta = 0.5 * alpha / (1 - cos_theta)
    P = ne * laser_power * (
        np.sqrt(alpha_costheta / np.pi) * 
        np.exp(-alpha_costheta * epsilon**2) * 
        (
            1 - 3.5 * epsilon + alpha_costheta * epsilon**3 
            - 1 / (8 * alpha) * (9.75 - 5 * cos_theta)
            + 0.125 * (29 + 5 / (1 - cos_theta)) * epsilon**2
            - 1 / 16 * alpha_costheta * (28 - 1 / (1 - cos_theta)) * epsilon**4
            + alpha_costheta**2 * epsilon**6 / 8
        )
    )
    return P / wavelength_laser  