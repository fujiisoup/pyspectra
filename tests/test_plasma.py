import numpy as np
import pytest
from pyspectra import fit, plasma, profiles


def test_thomson():
    wavelength = np.linspace(300, 1000, 10001)
    import matplotlib.pyplot as plt
    for Te in [1, 10, 100, 1000]:
    #for Te in [500, 1000]:
        thomson = plasma.thomson_incoherent(wavelength, 532, Te, np.pi/2)
        popt, _ = fit.singlepeak_fit(wavelength, thomson)
        fwhm = profiles.Gauss.sigma_to_FWHM(popt[2])
        if Te < 1000:
            print(Te / profiles.Gauss.sigma_to_FWHM(popt[2])**2)
            assert np.allclose((Te / fwhm**2), 0.163, atol=0.01)

        epsilon = (wavelength - 532) / 532
        plt.plot(epsilon, thomson, label='Te:{}, width:{}'.format(
            Te, fwhm))
    plt.legend()
    plt.xlim(-0.3, 0.3)
    plt.grid()
    plt.show()
