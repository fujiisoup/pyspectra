from .atoms import ATOMIC_MASS, ATOMIC_SYMBOLS

# light speed
c = 2.99792458e8  # m/s
light_speed = c

# boltzmann's constant
kb = 1.38064852e-23  # m2 kg s-2 K-1 or J/K
boltzmann_constant = kb

# proton mass
mp = 1.6726219e-27  # kg
proton_mass = mp

# atomic mass
mu = 1.66053906606e-27  # kg
atomic_mass = mu

# mass of several atoms
def mass(symbol):
    if symbol in ATOMIC_SYMBOLS:
        return mu * ATOMIC_MASS[ATOMIC_SYMBOLS.index(symbol)]

    raise ValueError("mass for {} is not implemented.".format(symbol))
