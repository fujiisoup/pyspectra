import numpy as np

C = 2.99792458e8  # light speed in m/s
H = 6.62607004e-34  # planck's constant m2 kg/s
KB = 1.38064852e-23  # boltzmann's constant in m2 kg s^(-2) K^-1
J2EV = 6.24150962915265e18  # eV / J
ALPHA = 7.2973525664e-3  # fine structure constant
RATE_AU = 4.13413733e16  # inverse of time in atomic unit
EV2CM = 8065.54429  # eV to cm^-1
Eh = 27.211386245988  # hartree in eV [eV / hartree]
e = 1.60217662e-19  # elementary charge in [C]
Me = 9.10938356e-31  # electron mass in [kg]
a0 = H / (2 * np.pi * Me * C * ALPHA)  # bohr radius in [m]


def hartree_to_eV(hartree):
    """ Convert hartlee to eV """
    return hartree * Eh


def eV_to_hartree(eV):
    """ Convert hartlee to eV """
    return eV / Eh


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

def nm_to_joule(nm):
    return eV_to_joule(nm_to_eV(nm))


def nm_to_Hz(nm):
    ''' Convert nm to Hz'''
    return C / (nm * 1e-9)


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


def eV_to_K(eV):
    """ Convert eV to K """
    return eV * 11600.0


def K_to_eV(K):
    """ Convert kelvin to eV """
    return K / 11600.0


def int_to_roman(number):
    """ Convert integer to roman symbol """
    if hasattr(number, 'shape'):
        shape = number.shape
        number = number.flatten()
        return np.array(
            [int_to_roman(int(i)) for i in number]
        ).reshape(shape)

    symbols = {
        "C": 100, "XC": 90, 
        "L": 50, "XL": 40,
        "X": 10, "IX": 9, 
        "V": 5, "IV": 4, 
        "I": 1
    }
    
    numeral = ""
    for sym, num in symbols.items():
        div = number // num
        number %= num
        while div:
            numeral += sym
            div -= 1
    return numeral


def roman_to_int(roman):
    """ Convert roman symbol to integer """
    if hasattr(roman, 'shape'):
        shape = roman.shape
        number = roman.flatten()
        return np.array(
            [roman_to_int(r.item()) for r in roman]
        ).reshape(shape)

    symbols = {
        "C": 100, "XC": 90, 
        "L": 50, "XL": 40,
        "X": 10, "IX": 9, 
        "V": 5, "IV": 4, 
        "I": 1
    }
    
    number = 0
    while len(roman) > 0:
        for sym, num in symbols.items():
            if sym == roman[:len(sym)]:
                number += num
                roman = roman[len(sym):]
                break
    return number