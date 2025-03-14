import numpy as np
from .atoms import ATOMIC_MASS, ATOMIC_SYMBOLS
from . import units

# light speed
c = 2.99792458e8  # m/s
light_speed = c

# boltzmann's constant
kb = 1.38064852e-23  # m2 kg s-2 K-1 or J/K
boltzmann_constant = kb

# proton mass
mp = 1.6726219e-27  # kg
proton_mass = mp

# electron mass
me = 9.10938356e-31  # kg
electron_mass = me

# fine structure constant
alpha = 7.2973525664e-3  # fine structure constant
fine_structure_constant = alpha

# atomic mass
mu = 1.66053906606e-27  # kg
atomic_mass = mu

# planck constant
h = 6.62607004e-34  # planck's constant m2 kg/s
planck_constant = h

# elementary charge
e = units.e
elementary_charge = e

# vacuum permittivity
eps0 = 8.8541878128e-12 # F/m
vacuum_permittivity = eps0

# mass of several atoms
def mass(symbol):
    if symbol in ATOMIC_SYMBOLS:
        return mu * ATOMIC_MASS[ATOMIC_SYMBOLS.index(symbol)]

    raise ValueError("mass for {} is not implemented.".format(symbol))

# atomic untis
class AtomicUnit:
    a0 = h / (2 * np.pi * me * c * alpha)  # bohr radius in [m]

# pressure
def Pa_to_Torr(p):
    '''convert Pa to Torr'''
    return p * 0.00750062

def Torr_to_Pa(torr):
    '''convert Torr to Pa'''
    return torr / 0.00750062