import numpy as np

C = 2.99792458e8  # light speed in m/s
H = 6.62607004e-34  # planck's constant m2 kg/s
KB = 1.38064852e-23  # boltzmann's constant in m2 kg s^(-2) K^-1
J2EV = 6.24150962915265e18  # eV / J
ALPHA = 7.2973525664e-3  # fine structure constant
RATE_AU = 4.13413733E16  # inverse of time in atomic unit
EV2CM = 8065.54429  # eV to cm^-1


def hartree_to_eV(hartree):
    """ Convert hartlee to eV """
    return hartree * 27.21138602


def eV_to_hartree(eV):
    """ Convert hartlee to eV """
    return eV / 27.21138602


def eV_to_nm(eV):
    """ Convert eV to nm """
    # h c / lambda
    # eV -> J : 0.1602e-18
    hc = H * C * 1.0e9 * J2EV  # to nm
    return hc / eV


def nm_to_eV(nm):
    """ Convert nm to eV 
    
    Make sure your wavelength is in vacuum not in airk
    """
    # h c / lambda
    # eV -> J : 0.1602e-18
    hc = H * C * 1.0e9 * J2EV  # to nm
    return hc / nm


def joule_to_eV(joule):
    """ convert joule value to eV """
    return joule * J2EV


def eV_to_joule(eV):
    """ convert eV to joule """
    return eV / J2EV


def eV_to_cm(eV):
    """ convert eV to cm^-1 """
    return eV * EV2CM


def cm_to_eV(cm):
    """ convert cm^-1 to eV """
    return cm / EV2CM


def nm_to_cm(nm):
    """ convert nm to cm^-1 

    Make sure your wavelength is in vacuum not in air
    """
    return 1 / nm * 1e7


def cm_to_nm(cm):
    """ convert nm to cm^-1 

    The return wavelength is in cm
    """
    return 1 / cm * 1e7
