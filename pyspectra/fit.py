"""
Some simple fitting procedures
"""
import numpy as np
from scipy import optimize
from .profiles import Gauss


def _initial_guess(x, y, p0):
    pass


def singlepeak_fit(x, y, p0=None, profile='gauss'):
    """
    Fit single peak to a spectrum
    """
    if p0 is None:
        # automatic estimate
        max_idx = np.argmax(y)
        offset = np.min(y)
        # centroid
        x0 = x[max_idx]
        ymax = y[max_idx]
        # estimate width
        idx = np.argsort(np.abs(x - x0))
        x_diff = (x - x0)[idx]
        y_diff = y[idx]
        i_half = np.argmin(np.abs(y_diff - offset - ymax / 2))
        width = x_diff[i_half]
        width = np.maximum(width, x[max_idx + 3] - x[max_idx])
        p0 = (ymax * width, x0, width, offset)
    
    if profile == 'gauss':
        popt, pcov = optimize.curve_fit(Gauss, x, y, p0)
    return popt, np.sqrt(np.diagonal(pcov))