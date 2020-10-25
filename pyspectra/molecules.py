import numpy as np
from . import data
from . import units

try:
    import xarray as xr
except ImportError:
    raise ImportError('xarray needs to be installed to use molecules module.')


def level(molecule, state, v, J):
    """
    Diatomic molecular levels.
    For notations, see webbok.nist.gov, such as 
    https://webbook.nist.gov/cgi/cbook.cgi?ID=C1333740&Mask=1000#Diatomic
    """
    constants = data.diatomic_molecules(molecule)
    if state not in constants['state']:
        raise ValueError('{} is not available. Available states are {}'.format(
            state, constants['state'].values
        ))
    constants = constants.sel(state=state).fillna(0)
    Evib = (
        constants['we'] * (v + 0.5) 
        - constants['wexe'] * (v + 0.5)**2 
        + constants['weye'] * (v + 0.5)**3
    )
    Erot = (
        (
            constants['Be'] 
            - constants['alpha_e'] * (v + 0.5)
            + constants['gamma_e'] * (v + 0.5)**2
        ) * J * (J + 1)
        -
        (
            constants['De']
            - constants['beta_e'] * (v + 0.5)
        ) * J**2 * (J + 1)**2
    )
    return units.cm_to_eV(constants['Te'] + Evib + Erot)


def level_OH_X2(v, J, parity, spin):
    r"""
    Energy level of OH X^2 \Pi state.

    From 
    Abrams, M. C., Davis, S. P., Rao, M. L. P., Engleman, R., & Brault, J. W. (1994). 
    HIGH-RESOLUTION FOURIER TRANSFORM SPECTROSCOPY OF THE MEINEL SYSTEM OF OH. 
    In The Astrophysical Journal Supplement Series (Vol. 93).
    
    Parameters
    ----------
    v: array
        vibrational quantum number
    J: array
        rotational quantum number
    parity: array
        e and f parity sublevels
        it should be +1 (for e parity) or -1 (for f parity)
    spin: array
        1 for 3/2, 2 for 1/2

    Returns
    -------
    energy: v.shape + (2, ) array
        Energies of the v, J and spin=(1/2, 3/2) levels in cm^{-1}
    """
    if not np.isin(spin, [1, 2]).all():
        raise ValueError("Spin should be either 1 or 2, for 3/2 or 1/2, respectively.")

    v, J, parity, spin = np.broadcast_arrays(v, J, parity, spin)
    energies = _level_OH_X2(v, J, parity)
    return np.where(spin == 2, energies[..., 1], energies[..., 0])


def _level_OH_X2(v, J, parity):
    r"""
    Energy level of OH X^2 \Pi state.

    From 
    Abrams, M. C., Davis, S. P., Rao, M. L. P., Engleman, R., & Brault, J. W. (1994). 
    HIGH-RESOLUTION FOURIER TRANSFORM SPECTROSCOPY OF THE MEINEL SYSTEM OF OH. 
    In The Astrophysical Journal Supplement Series (Vol. 93).
    
    Parameters
    ----------
    v: array
        vibrational quantum number
    J: array
        rotational quantum number
    parity: array
        e and f parity sublevels
        it should be +1 (for e parity) or -1 (for f parity)

    Returns
    -------
    energy: v.shape + (2, ) array
        Energies of the v, J and spin=(1/2, 3/2) levels in cm^{-1}
    """
    # Table 26A and B  All the unit is in cm^{-1}
    # v:          0           1           2           3           4
    #             5           6           7           8           9          10
    Te = np.array([
                0.0, 3569.64083, 6973.67894, 10214.0374, 13291.8114, 
         16207.1055, 18958.7998, 21544.2746, 23959.0059, 26196.0397, 28245.3134])
    A = np.array([
        -139.115111, -139.38463, -139.65156, -139.90749, -140.14688,
        -140.35308 , -140.50665, -140.57257, -140.49428, -140.20552, -139.56879])
    B = np.array([
          18.550208, 17.8384461,  17.136263,  16.441072, 15.7493454,
          15.056964,  14.359072,  13.648691,  12.917041, 12.152874 , 11.338859])
    D = np.array([
         0.19120892,    0.18733,    0.18394,    0.18129,    0.17906,
         0.17730   ,    0.17686,    0.17781,    0.18062,    0.18771,   0.19842]) * 1e-2
    H = np.array([
              0.123,      0.116,      0.112,      0.116,      0.116,
              0.09556,  0.07861,    0.05878,    0.01484,    0.03317,      0.01]) * 1e-6
    AD = np.array([
             -0.746,     -0.710,     -0.665,     -0.623,      -0.50,
             -0.512,     -0.236,     -0.112,     -0.079,      0.124,     0.353]) * 1e-3
    P = np.array([
           0.235001,   0.224377,   0.213382,   0.203733,   0.191900,
           0.179240,   0.167682,   0.153709,   0.137667,   0.116907,  0.092975])
    pD = np.array([
             -0.187,     -0.177,     -0.157,     -0.274,     -0.275,
             -0.215,     -0.492,     -0.539,     -0.646,     -0.560,    -0.696]) * 1e-4
    q = np.array([
          -0.038689,  -0.036957,  -0.035169, -0.0337132, -0.0317617,
          -0.0298844,  -0.27983,   -0.25933,   -0.23866,  -0.021318,  -0.18656])
    qD = np.array([
              0.139,      0.137,      0.133,      0.178,      0.165,
              0.163,      0.168,      0.167,      0.182,      0.167,     0.171]) * 1e-4
    qH = np.array([
            -0.1801,    -0.1732,    -0.1751,      -0.17,      -0.17,
            -0.17  ,    -0.17  ,    -0.17  ,    -0.17  ,    -0.17  ,    -0.17]) * 1e-9
    L = np.array([
            -0.2298,    -0.2790,    -0.3301,      -0.33,      -0.33,
            -0.33  ,    -0.33  ,    -0.33  ,      -0.33,      -0.33,     -0.33]) * 1e-12
    AH = np.array([
             0.1852,     0.2001,     0.2094,      0.208,      0.208,
             0.206,      0.205 ,     0.204 ,      0.203,      0.202,     0.201]) * 1e-5

    v, J, parity = np.broadcast_arrays(v, J, parity)
    if not np.isin(parity, [-1, 1]).all():
        raise ValueError('parity should be given by +/- 1')
    
    if not ((0 <= v) * (v <= 10)).all():
        raise ValueError('Only v <= 10 are supported')
    
    v = v.astype(int)
    Te = Te[v]
    A = A[v]
    B = B[v]
    D = D[v]
    H = H[v]
    AD = AD[v]
    p = P[v] + J * (J + 1) * pD[v]
    q = q[v] + J * (J + 1) * (qD[v] + J * (J + 1) * qH[v])
    L = L[v]
    AH = AH[v]
    # see Table 2
    z = (J - 0.5) * (J + 1.5)
    zp1 = z + 1
    zm1 = z - 1
    Ham = np.zeros(v.shape + (2, 2), dtype=float)
    Ham[..., 0, 0] = (
        Te - 0.5 * A
        + B * zp1
        - D * (zp1**2 + z)
        + H * (zp1**3 + z * (3 * z + 1))
        - L * (zp1**4 + z * (6 * z**2 + 5 * z + 2))
        - 0.5 * AD * zp1
        - 0.25 * AH * (3 * zp1**2 + z)
        + 0.5 * p * (1 - parity * (J + 0.5))
        + 0.5 * q * (z - 2 * parity * (J + 0.5))  # <-- this part might be a typo in the original paper
    )
    Ham[..., 1, 1] = (
        Te + 0.5 * A 
        + B * zm1 
        - D * (zm1**2 + z) 
        + H * (zm1**3 + z * (3 * z - 1))
        - L * (zm1**4 + z * (6 * z**2 - 3 * z + 2)) 
        + 0.5 * AD * zm1
        + 0.25 * AH * (3 * zm1**2 + z) + 0.5 * q * z
    )
    Ham[..., 1, 0] = -np.sqrt(z) * (
        B - 2 * D * z 
        + H * (3 * z**2 + z + 1) 
        - L * (4 * z * (z**2 + z + 1)) 
        - 0.5 * AH
        + 0.25 * p
        - 0.5 * q * (-1 + parity * (J + 0.5))
    )
    Ham[..., 0, 1] = Ham[..., 1, 0]
    return np.linalg.eigvalsh(Ham, UPLO='L')
