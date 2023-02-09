"""
Some simple fitting procedures
"""
import numpy as np
from scipy import optimize, sparse

try:
    from sklearn import linear_model
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    pass

from .profiles import Gauss, Lorentz, Voigt


def _initial_guess(x, y, profile, method="nearest"):
    """
    initial guess based on the median

    method: 'nearest' | 'cumulative'

    methods
    -------
    nearest
        Estimate width based on the nearest points satisfying fwhm
    """
    # automatic estimate
    max_idx = np.argmax(y)
    offset = np.min(y)
    # centroid
    x0 = x[max_idx]
    ymax = y[max_idx]

    if method == "nearest":
        # estimate width
        idx = np.argsort(np.abs(x - x0))
        x_diff = np.abs(x - x0)[idx]
        y_diff = y[idx]
        width = x_diff[y_diff - offset - ymax / 2 < 0][0]
        # make it
        width = np.maximum(width, x_diff[2])
        if profile in ["gauss", "gaussian"]:
            width = width / np.sqrt(2 * np.log(2))
            p0 = (ymax * width, x0, width, offset)
        elif profile in ["lorentz", "lorentzian"]:
            p0 = (ymax * width, x0, width, offset)
        elif profile == "voigt":
            # half-width at 10% of maximum
            width = width / np.sqrt(2 * np.log(2))
            width_01 = x_diff[y_diff - offset - ymax / 10 < 0][0]
            gamma = np.maximum(width_01 - 2 * width, width * 0.1)
            p0 = (ymax * width, x0, width, gamma / 1.5, offset)
    return p0


def singlepeak_fit(x, y, p0=None, profile="gauss"):
    """
    Fit single peak to a spectrum
    """
    if p0 is None:
        p0 = _initial_guess(x, y, profile, method="nearest")

    if profile.lower() in ["gauss", "gaussian"]:
        popt, pcov = optimize.curve_fit(Gauss, x, y, p0)
    elif profile.lower() in ["lorentz", "lorentzian"]:
        popt, pcov = optimize.curve_fit(Lorentz, x, y, p0)
    elif profile.lower() in ["voigt"]:
        popt, pcov = optimize.curve_fit(Voigt, x, y, p0)
    else:
        raise NotImplementedError("fitting with {} is not implemented.".format(profile))
    return popt, np.sqrt(np.diagonal(pcov))


def multiframe_fit(x, y, A0, x0, w0, y0, fix=''):
    '''
    Run multiframe fit for x and y.
    
    Parameters
    ----------
    x: 1d or 2d matrix for the x-axis.
    y: 2d matrix of the intensity, shape (n, m)
    A0, x0, w0:
        Initial guess of the parameters to be evaluated:
        They are intensity, centroids, widths. 
        They must be all 2d, but their shape determines the parameter 
        sharing during the fit.
         
        (n, k): n-frame for k-lines.
        (1, k): The same values for all the frames, but different values for k-lines.
        (1, 1): The same values for all the frames and lines.
    y0: 
        Initial guess of the background. The shape should be (n,)
    fix:
        Fixed parameters. A comma-separated keywords.
        e.g., 'A0,x0'
    '''
    fixed = []
    for f in fix.split(','):
        if len(f) == 0:
            continue
        f = f.strip().lower()
        if f not in ['A0', 'x0', 'w0', 'y0']:
            raise ValueError('Fixing {} is invalid.'.format(f))
        fixed.append(f)

    n, l = y.shape
    shape_x = x.shape
    k = 1
    for key, p in [('A0', A0), ('x0', x0), ('w0', w0)]:
        ndim = getattr(p, 'ndim', 0)
        if ndim != 2:
            raise ValueError(
                '{} should be 2-dimensional, not {}-dimensional'.format(key, ndim)
            )
        if p.shape[0] not in [1, n]:
            raise ValueError(
                'Initial guess must have the shape of ({}, k). {}.shape = {}'.format(
                    n, key, p.shape)
            )
        if p.shape[1] not in (k, 1):
            if k == 1:
                k = p.shape[1]
            else:
                raise ValueError('Wrong shape of {}. This should be ({}, {}) or ({}, 1)'.format(
                    key,  n, k, n
                ))
    
    assert y0.ndim == 1
    shape_A0 = A0.shape
    size_A0 = A0.size
    shape_x0 = x0.shape
    size_x0 = x0.size
    shape_w0 = w0.shape
    size_w0 = w0.size
    shape_y0 = y0.shape
    size_y0 = y0.size

    p0 = []
    if 'A0' not in fixed:
        p0.append(A0.ravel())
    if 'x0' not in fixed:
        p0.append(x0.ravel())
    if 'w0' not in fixed:
        p0.append(w0.ravel())
    if 'y0' not in fixed:
        p0.append(y0.ravel())

    p0 = np.concatenate(p0)

    def to_param(p):
        ''' Divide and reshaping a 1d-array to parameters each with 2d. '''
        if 'A0' in fixed:
            A = A0
        else:
            A, p = p[:size_A0], p[size_A0:]

        if 'x0' in fixed:
            x = x0
        else:
            x, p = p[:size_x0], p[size_x0:]
        
        if 'w0' in fixed:
            w = w0
        else:
            w, p = p[:size_w0], p[size_w0:]
        
        if 'y0' in fixed:
            y = y0
        else:
            y = p[:size_y0]
        return A.reshape(*shape_A0), x.reshape(*shape_x0), w.reshape(*shape_w0), y.reshape(*shape_y0)

    def compose_profile(x, p):
        # overall profile
        A0, x0, w0, y0 = to_param(p)
        return y0[:, np.newaxis] + np.sum(Gauss(
            x[..., np.newaxis],   # shape (n, m, 1) or (m, 1)
            A0[:, np.newaxis], x0[:, np.newaxis], w0[:, np.newaxis], 0  # shape (n, 1, k) or (1, 1, k)
        ), axis=-1)  # -> (n, m)

    def func(x, *p):
        fit = compose_profile(x, np.array(p))
        return fit.ravel()

    popt, pcov = optimize.curve_fit(func, x, y.ravel(), p0)
    perr = np.sqrt(np.diagonal(pcov))
    A0, x0, w0, y0 = to_param(popt)
    A0_err, x0_err, w0_err, y0_err = to_param(perr)
    if 'A0' in fixed:
        A0_err = np.zeros_like(A0_err)
    if 'x0' in fixed:
        x0_err = np.zeros_like(x0_err)
    if 'w0' in fixed:
        w0_err = np.zeros_like(w0_err)
    if 'y0' in fixed:
        y0_err = np.zeros_like(y0_err)

    fit = compose_profile(x, popt)

    result = {
        'A0': A0, 'A0_err': A0_err,
        'x0': x0, 'x0_err': x0_err,
        'w0': w0, 'w0_err': w0_err,
        'y0': y0, 'y0_err': y0_err,
        'fit': fit
    }
    return result


def _make_template_matrix(template, size):
    """
    Make matrix by sliding a temlate
    """
    n = len(template)
    row_ind, col_ind = [], []
    values = []
    for i in range(n - 1):
        col_ind.append(np.arange(i + 1))
        row_ind.append(np.ones(i + 1, dtype=int) * i)
        values.append(template[-i - 1 :])

    for i in range(size - n + 1):
        col_ind.append(np.arange(i, i + n))
        row_ind.append(np.ones(n, dtype=int) * (n + i - 1))
        values.append(template)

    for i in range(n - 1):
        col_ind.append(np.arange(size - n + i + 1, size))
        row_ind.append(np.ones(n - i - 1, dtype=int) * (size + i))
        values.append(template[: -i - 1])

    values = np.concatenate(values, axis=0)
    row_ind = np.concatenate(row_ind, axis=0)
    col_ind = np.concatenate(col_ind, axis=0)
    return sparse.csc_matrix((values, (col_ind, row_ind)), shape=(size, size + n - 1))


def _robust_width(x, min_dx):
    x2 = np.maximum(x, 0)
    x2[0] += min_dx
    return x2


def multipeak_nnls(
    x,
    y,
    width=None,
    fit_width=False,
    profile="gauss",
    alpha=0.0,
    delta_sigma=5.0,
    intercept="fit",
    max_iter=1000,
    tol=0.0001,
    fit_method="Nelder-Mead",
):
    """
    Multipeak fit by nnls.

    Parameters
    ----------
    x: 1d-array. Should be almost equidistant
    y: 1d-array containing peaks. Should contain only positive peaks.
    width: initial value of width. For voigt profile, it should be a 2-element tuple
    fit_width: bool. If True, estimate the best (shared) width.
    profile: one of {'gauss', 'lorentz', 'voigt'}
    alpha: regularization factor
    delta_sigma: float. How far wings should be considered. Default 5
    intercept: one of {'fit', 'min', 'fit_slope'} or float
    """
    if not HAS_SKLEARN:
        raise ImportError('scikit-learn is necessary for "multipeak_fit"')

    x = np.asanyarray(x)
    y = np.asanyarray(y)

    if width is None:
        raise NotImplementedError("width None is not implemented.")

    if not fit_width:
        return _multipeak_nnls1(
            x, y, width, profile, alpha, delta_sigma, intercept, max_iter, tol
        )

    min_dx = np.mean(np.diff(x)) * 0.1
    # optimizing the width.
    width_init = np.atleast_1d(width)
    func = lambda w: _multipeak_nnls1(
        x,
        y,
        _robust_width(w, min_dx),
        profile,
        alpha,
        delta_sigma,
        intercept,
        max_iter,
        tol,
    )["loss"]
    res = optimize.minimize(func, width_init, method=fit_method)

    result = _multipeak_nnls1(
        x,
        y,
        _robust_width(res["x"], min_dx),
        profile,
        alpha,
        delta_sigma,
        intercept,
        max_iter,
        tol,
    )
    result["width"] = _robust_width(res["x"], min_dx)
    return result


def _multipeak_nnls1(
    x,
    y,
    width,
    profile="gauss",
    alpha=0.0,
    delta_sigma=5.0,
    intercept="fit",
    max_iter=1000,
    tol=0.0001,
):
    dx = np.mean(np.diff(x))
    try:
        n = int(width / dx * delta_sigma)
    except TypeError:
        n = int((width[0] + width[1]) / dx * delta_sigma)
    n = np.minimum(np.maximum(n, 3), len(x) // 2)
    t = np.arange(-n, n + 1) * dx

    if profile.lower() in ["gauss", "gaussian"]:
        template = Gauss(t, 1, 0, width, 0)
    elif profile.lower() in ["lorentz", "lorentzian"]:
        template = Lorentz(t, 1, 0, width, 0)
    elif profile.lower() in ["voigt"]:
        if len(width) != 2:
            raise ValueError("For Voigt profile, width must be a 2-element tuple.")
        template = Voigt(t, 1, 0, width[0], width[1], 0)
    else:
        raise NotImplementedError("fitting with {} is not implemented.".format(profile))

    max_height = np.max(template)
    mat = _make_template_matrix(template / max_height, len(y))

    if intercept == "fit":
        model = linear_model.Lasso(
            alpha=alpha,
            fit_intercept=True,
            positive=True,
            max_iter=max_iter,
            tol=tol,
        )
        model.fit(X=mat, y=y)
        intercept = model.intercept_
    elif intercept == "fit_slope":
        raise NotImplementedError
    else:
        if intercept == "min":
            intercept = np.min(y)
        else:
            intercept = intercept
        model = linear_model.Lasso(
            alpha=alpha, fit_intercept=False, positive=True, max_iter=max_iter, tol=tol
        )
        model.fit(X=mat, y=y - intercept)

    result = {
        "coef": model.coef_[n:-n] / max_height,
        "peak": model.coef_[n:-n] + intercept,
        "fit": model.predict(X=mat) + intercept,
        "intercept": intercept,
    }

    # loss
    result["loss"] = np.sum((result["fit"] - y) ** 2) / (2 * len(y)) + alpha * np.sum(
        model.coef_
    )

    # discretized
    centers, intensities, nonzero_points = _discretize_coef(x, result["coef"])
    result["center"] = centers
    result["intensity"] = intensities
    result["peak_height"] = intensities * max_height + intercept
    result["nonzero_points"] = nonzero_points
    return result


def _discretize_coef(x, coef):
    """
    From x and coef, which is almost zero, estimate the discrete line center and intensity
    """
    nonzero = np.arange(len(x))
    nonzero = np.concatenate([[-1], nonzero[coef > 0], [len(x) - 1]])
    indexes = nonzero[:-1][np.diff(nonzero) > 1] + 1
    centers = []
    intensities = []
    nonzero_points = []  # number of successive nonzero points

    for i in range(len(indexes) - 1):
        sl = slice(indexes[i], indexes[i + 1])
        centers.append(np.sum(x[sl] * coef[sl]) / np.sum(coef[sl]))
        intensities.append(np.sum(coef[sl]))
        nonzero_points.append((coef[sl] > 0).sum())
    return np.array(centers), np.array(intensities), np.array(nonzero_points)
