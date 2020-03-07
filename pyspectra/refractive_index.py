def air(nm):
    """
    P. E. Ciddor. Refractive index of air: new equations for the visible and near infrared, Appl. Optics 35, 1566-1573 (1996)
    [Calculation script (Python) - can be used for calculating refractive index of air at a given humidity, temperatire, pressure, and CO2 concentration]
    """
    x = nm * 1e-3
    return 1 + 0.05792105 / (238.0185 - x**-2) + 0.00167917 / (57.362 - x**-2)


def vacuum_to_air(nm_in_vacuum):
    # Note: in air, the wavelength becomes short
    return nm_in_vacuum / air(nm_in_vacuum)


def air_to_vacuum(nm_in_air):
    return nm_in_air * air(nm_in_air)
