import numpy as np
from scipy import optimize


def nnls(
    y, template, maxiter=None, regularization_type="none", regularization_parameter=0.0
):
    """
    Find the non-negative solution to
    
    template * x = y
    
    where '*' is the convolution and x is a nonnegative-valued vector.

    Parameters
    ----------
    y: 1d array (shape n)
    template: 1d array (shape m)
    regularization_type: string indicating the regularization used.
        One of {'none', 'ridge', 'lasso'}

    Returns
    -------
    x: 1d array (shape n - m + 1)
    """
    X = _template_to_matrix(template, len(y))
    if regularization_type == "none":
        return optimize.nnls(X, y)[0]

    try:
        from sklearn.linear_model import ElasticNet, Lasso
    except ImportError:
        raise ImportError(
            "sklearn must be installed for deconvolve with regularization"
        )

    if regularization_type == "ridge":
        reg_nnls = ElasticNet(
            alpha=regularization_parameter,
            l1_ratio=0.0,
            fit_intercept=False,
            copy_X=True,
            positive=True,
        )
    elif regularization_type == "lasso":
        reg_nnls = Lasso(
            alpha=regularization_parameter,
            fit_intercept=False,
            copy_X=True,
            positive=True,
        )
    else:
        raise ValueError(
            'regularization type "{}" is not supported'.format(regularization_type)
        )
    return reg_nnls.fit(X, y).coef_


def convolve(x, template):
    """
    Convolution of x and template

    Parameters
    ----------
    x: 1d array (shape n - m + 1)
    template: 1d array (shape m)

    Returns
    -------
    y: 1d array (shape n)
    """
    n = len(x) + len(template) - 1
    X = _template_to_matrix(template, n)
    return X @ x


def _template_to_matrix(template, n):
    """
    Construct a matrix from a template so that the convolution 
    can be mapped to a matrix-vector product.

    Parameter
    ---------
    template: 1d-array shape m
    n: scalar

    Returns
    -------
    X: 2d-array shape [n, n - m + 1]
    """
    if template.ndim != 1:
        raise ValueError("Currently only 1d-convolution is supported.")

    template = np.array(template, copy=False, ndmin=1)
    m = len(template)
    # shape: 2 * n - m
    padded = np.pad(template[::-1], (n - m, n - m), mode="constant", constant_values=0)

    shape = (n, n - m + 1)
    strides = padded.strides + (padded.strides[-1],)
    return np.lib.stride_tricks.as_strided(
        padded, shape=shape, strides=strides, writeable=False
    )[::-1]
