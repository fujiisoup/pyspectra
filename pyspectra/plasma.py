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


def coulomb_log_ii_identical(n_i_m3: float, Ti_eV: float) -> float:
    """
    NRL-style ion-ion Coulomb logarithm for identical ions (approx).
    n_i_m3 -> convert to cm^-3 inside formula.
    """
    n_i_cm3 = n_i_m3 * 1e-6
    if n_i_cm3 <= 0 or Ti_eV <= 0:
        raise ValueError("Density and temperature must be positive.")
    # lnΛ ≈ 23 - ln( sqrt(2 n) / T^(3/2) ), with n in cm^-3, T in eV
    return 23.0 - np.log(np.sqrt(2.0 * n_i_cm3) / (Ti_eV ** 1.5))


def tau_i_nrl(n_i_m3: float, Ti_eV: float, mi_kg: float, lnL: float) -> float:
    """
    Ion collision time τ_i [s] using NRL practical form:
      τ_i = 2.09e7 * Ti^(3/2) / ( n[cm^-3] * lnΛ * sqrt(μ) )
    where μ = mi / m_p and n is ion density in cm^-3.
    """
    n_cm3 = n_i_m3 * 1e-6
    mu = mi_kg / constants.proton_mass
    if n_cm3 <= 0 or Ti_eV <= 0 or lnL <= 0 or mu <= 0:
        raise ValueError("All inputs must be positive and lnΛ > 0.")
    return 2.09e7 * (Ti_eV ** 1.5) / (n_cm3 * lnL * np.sqrt(mu))


def D_parallel(ne_m3: float, Ti_eV: float, mi_kg: float):
    """
    Compute D_parallel and return (D, lnL, tau, nu).
    """
    n_i_m3 = ne_m3  # quasi-neutral, single ion species
    lnL = coulomb_log_ii_identical(n_i_m3, Ti_eV)
    tau = tau_i_nrl(n_i_m3, Ti_eV, mi_kg, lnL)
    nu = 1.0 / tau
    Ti_J = units.eV_to_joule(Ti_eV)
    D = (Ti_J * tau) / mi_kg  # = (kT/m) * (1/nu) with kT in J
    return D, lnL, tau, nu
