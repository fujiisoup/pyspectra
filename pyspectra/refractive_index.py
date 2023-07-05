import numpy as np

def air(nm):
    """
    P. E. Ciddor. Refractive index of air: new equations for the visible and near infrared, Appl. Optics 35, 1566-1573 (1996)
    [Calculation script (Python) - can be used for calculating refractive index of air at a given humidity, temperatire, pressure, and CO2 concentration]
    """
    x = nm * 1e-3
    return 1 + 0.05792105 / (238.0185 - x ** -2) + 0.00167917 / (57.362 - x ** -2)


def vacuum_to_air(nm_in_vacuum):
    # Note: in air, the wavelength becomes short
    return nm_in_vacuum / air(nm_in_vacuum)


def air_to_vacuum(nm_in_air):
    return nm_in_air * air(nm_in_air)


REFRACTIVE_INDEX_COEFS = {
    # coefficients for schott formulae
    # (a0, b0, a1, b1, a2, a3): n^2 - 1 = sum_i ai lam^2 / (lam^2 - b_i)
    # fused silica
    'fused silica': (
        0.6961663, 0.0684043**2, 0.4079426, 0.1162414**2, 0.8974794, 9.896161**2
    ),
    # 
    'H-ZF3': (0.203624748, 0.0620712732, 1.64202576, 0.0127139992, 1.38768085, 128.254467),
    # For Schott
    'N-BK7': (1.03961212, 0.00600069867, 0.231792344, 0.0200179144, 1.01046945, 103.560653),
    'N-F2': (1.34533359, 0.00997743871, 0.209073176, 0.0470450767, 0.937357162, 111.886764),
    'N-KF9': (1.19286778, 0.00839154696, 0.0893346571, 0.0404010786, 0.920819805, 112.572446),
    'N-SF6': (1.72448482, 0.0134871947, 0.390104889, 0.0569318095, 1.04572858, 118.557185),
    'N-SF10': (1.61625977, 0.0127534559, 0.259229334, 0.0581983954, 1.07762317, 116.60768),
    'N-SF11': (1.73848403, 0.0136068604, 0.311168974, 0.0615960463, 1.17490871, 121.922711),
    'N-LASF31A': (1.96485075, 0.00982060155, 0.475231259, 0.0344713438, 1.48360109, 110.739863),
    'N-LASF43': (1.93502827,0.0104001413, 0.23662935, 0.0447505292, 1.26291344, 87.437569),
    'N-LASF46B': (2.17988922, 0.0125805384, 0.306495184, 0.0567191367, 1.56882437, 105.316538),
    # LAK
    'N-LAK10': (1.72878017, 0.00886014635, 0.169257825, 0.0363416509, 1.19386956, 82.9009069),
    'N-LAK35': (1.3932426, 0.00715959695, 0.418882766, 0.0233637446, 1.043807, 88.3284426),
    'N-FK51A': (0.971247817, 0.00472301995, 0.216901417, 0.0153575612, 0.904651666, 168.68133),
}
    

class Glass:
    def __init__(self, name):
        if name not in REFRACTIVE_INDEX_COEFS.keys():
            raise NotImplementedError('{} is not supported'.format(name))
        
        self.coefs = REFRACTIVE_INDEX_COEFS[name]

    def __call__(self, wavelength_in_nm):
        ''' Returns the refractive index for given wavelength.'''
        lam = wavelength_in_nm * 1e-3
        return np.sqrt(
            1.0 + 
            self.coefs[0] * lam**2 / (lam**2 - self.coefs[1]) + 
            self.coefs[2] * lam**2 / (lam**2 - self.coefs[3]) + 
            self.coefs[4] * lam**2 / (lam**2 - self.coefs[5])
        )
