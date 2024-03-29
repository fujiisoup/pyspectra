import os
import numpy as np
import pytest
from pyspectra import fit, profiles


THIS_DIR = os.path.dirname(__file__)


@pytest.mark.parametrize(
    ("n", "sn", "x0", "width"),
    [
        (100, 100.0, 0.0, 10.0),
        (100, 30.0, 0.0, 10.0),
        (100, 100.0, 0.0, 30.0),
        (100, 100.0, 30.0, 30.0),
        (1000, 10.0, 30.0, 30.0),
    ],
)
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("profile", ["gauss", "lorentz"])
def test_simple_fit(n, sn, x0, width, seed, profile):
    rng = np.random.RandomState(seed)
    x = np.linspace(-100, 100, n)
    A = sn * width * np.sqrt(2 * np.pi)
    if profile == "gauss":
        y = profiles.Gauss(x, A, x0, width, 0) + rng.randn(n)
    elif profile == "lorentz":
        y = profiles.Lorentz(x, A, x0, width, 0) + rng.randn(n)

    popt, perr = fit.singlepeak_fit(x, y, profile=profile)
    print(popt[2], width)
    expected = np.array((A, x0, width, 0))
    assert ((popt - 5 * perr < expected) * (expected < popt + 5 * perr)).all()


@pytest.mark.parametrize(
    ("n", "sn", "x0", "sigma", "gamma"),
    [
        (100, 100.0, 0.0, 10.0, 4.0),
        (100, 30.0, 0.0, 10.0, 30.0),
        (1000, 10.0, 30.0, 30.0, 0.2),
    ],
)
@pytest.mark.parametrize("seed", [0, 1])
def test_voigt(n, sn, x0, sigma, gamma, seed):
    rng = np.random.RandomState(seed)
    x = np.linspace(-100, 100, n)
    A = sn * sigma * np.sqrt(2 * np.pi)
    y = profiles.Voigt(x, A, x0, sigma, gamma, 0) + rng.randn(n)

    popt, perr = fit.singlepeak_fit(x, y, profile="voigt")

    expected = np.array((A, x0, sigma, gamma, 0))
    assert ((popt - 5 * perr < expected) * (expected < popt + 5 * perr)).all()


@pytest.mark.parametrize("n", [10, 100])
def test_make_template_matrix(n):
    # with size 1 template, it becomes an identity
    template = [1]
    expected = np.eye(n)
    actual = np.asarray(fit._make_template_matrix(template, n).todense())
    assert expected.shape == actual.shape
    assert (expected == actual).all()

    # this makes also an identity
    template = [0, 1, 0]
    actual = np.array(fit._make_template_matrix(template, n).todense())
    actual = actual[:, 1:-1]
    assert expected.shape == actual.shape
    assert (expected == actual).all()


@pytest.mark.parametrize(
    ["peaks", "width", "noise"], [(((3, 10), (20, 3), (60, 3)), 1, 0.01)]
)
def test_multipeak_nnls(peaks, width, noise):
    x = np.linspace(0, 100, 301)
    y = np.random.RandomState(0).randn(301) * noise
    for peak in peaks:
        y = y + profiles.Gauss(x, peak[1], peak[0], width, 0)

    res = fit.multipeak_nnls(x, y, width=width, alpha=0.001, fit_width=False)
    assert np.sum(res["coef"] > 0) == len(peaks)


@pytest.mark.parametrize(
    ["peaks", "width", "noise"], [(((3, 10), (20, 3), (60, 3)), 2, 0.01)]
)
def test_multipeak_nnls_fitwidth(peaks, width, noise):
    x = np.linspace(0, 100, 101)
    y = np.random.RandomState(0).randn(101) * noise
    for peak in peaks:
        y = y + profiles.Gauss(x, peak[1], peak[0], width, 0)

    res = fit.multipeak_nnls(
        x, y, width=width * 1.3, alpha=0.001, fit_width=True, fit_method="L-BFGS-B"
    )
    assert np.sum(res["coef"] > 0) == len(peaks)
    assert np.allclose(res["width"], width, atol=0.03, rtol=0.03)

    res = fit.multipeak_nnls(
        x, y, width=width * 1.3, alpha=0.001, fit_width=True, fit_method="Nelder-Mead"
    )
    assert np.sum(res["coef"] > 0) == len(peaks)
    assert np.allclose(res["width"], width, atol=0.03, rtol=0.03)


def test_discretize():
    x = np.linspace(0, 1, 31)
    coef = np.zeros_like(x)
    coef[4] = 1
    coef[5] = 1
    centers_expected = [(x[4] + x[5]) / 2.0]
    intensities_expected = [2]
    centers, intensities, nonzero_points = fit._discretize_coef(x, coef)
    assert np.allclose(centers, centers_expected)
    assert np.allclose(intensities, intensities_expected)
    assert np.allclose(nonzero_points, [2])

    coef[13] = 1.0
    coef[14] = 2.0
    centers_expected = np.concatenate([centers_expected, [(x[13] + x[14] * 2) / 3.0]])
    intensities_expected = np.concatenate([intensities_expected, [3]])
    centers, intensities, nonzero_points = fit._discretize_coef(x, coef)
    assert np.allclose(centers, centers_expected)
    assert np.allclose(intensities, intensities_expected)
    assert np.allclose(nonzero_points, [2, 2])

    coef[20] = 0.1
    coef[21] = 2.0
    coef[22] = 3.0
    centers_expected = np.concatenate(
        [centers_expected, [(x[20] * 0.1 + x[21] * 2 + x[22] * 3) / 5.1]]
    )
    intensities_expected = np.concatenate([intensities_expected, [5.1]])
    centers, intensities, nonzero_points = fit._discretize_coef(x, coef)
    assert np.allclose(centers, centers_expected)
    assert np.allclose(intensities, intensities_expected)
    assert np.allclose(nonzero_points, [2, 2, 3])
