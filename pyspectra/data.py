import os
import ssl
import numpy as np
import requests
from .atoms import ATOMIC_SYMBOLS
from . import units, refractive_index


ssl._create_default_https_context = ssl._create_unverified_context

_default_cache_dir = os.sep.join((os.path.expanduser("~"), ".pyspectra_data"))


def assure_directory(dirname):
    if not os.path.exists(dirname):
        parent_dir = dirname[: dirname.rfind(os.sep)]
        assure_directory(parent_dir)
        os.mkdir(dirname)


def atom_levels(
    element,
    nele=None,
    force_download=False,
    source="nist",
    cache_dir=_default_cache_dir,
):
    return _atom("levels", element, nele, force_download, source, cache_dir)


def atom_lines(
    element,
    nele=None,
    unit="nm(vacuum)",
    force_download=False,
    source="nist",
    cache_dir=_default_cache_dir,
):
    ds = _atom("lines", element, nele, force_download, source, cache_dir)

    keys = ["nm(air)", "nm(vacuum)", "eV", "cm"]
    if unit not in keys:
        raise NotImplementedError(
            "unit {} is not supported. Supported units are {}.".format(unit, keys)
        )

    for key in ["wavelength", "wavelength_ritz"]:
        key_err = key + "_err"
        if key_err in ds:
            error_ratio = (ds[key_err] / ds[key]).values

        if unit == "nm(air)":
            ds[key] = refractive_index.vacuum_to_air(ds[key])
            ds[key].attrs["unit"] = "nm(air)"
        elif unit == "eV":
            ds[key] = units.nm_to_eV(ds[key])
            ds[key].attrs["unit"] = "eV"
        elif unit == "cm":
            ds[key] = units.nm_to_cm(ds["wavelength"])
            ds[key].attrs["unit"] = "cm^{-1}"

        if key_err in ds:
            ds[key_err] = ds[key] * error_ratio
            ds[key_err].attrs["unit"] = ds[key].attrs["unit"]
    return ds


def search_atom_lines(
    elements='', wavelength_start=None, wavelength_stop=None,
    source="nist"
):
    """
    Search atomic emission lines from NIST.
    
    """

    if source not in ["nist"]:
        raise NotImplementedError("Only NIST is available yet.")

    from ._data_atom_nist import get_lines

    ds = get_lines(
        elements, nele=None, 
        low_w=str(wavelength_start), upp_w=str(wavelength_stop)
    )
    return ds

def _atom(
    kind,
    element,
    nele=None,
    force_download=False,
    source="nist",
    cache_dir=_default_cache_dir,
):
    import xarray as xr
    from ._data_atom_nist import get_levels, get_lines

    if nele is None:  # neutral is assumed
        nele = ATOMIC_SYMBOLS.index(element)

    if source not in ["nist"]:
        raise NotImplementedError("Only NIST is available yet.")

    cache_dir = os.sep.join((_default_cache_dir, "atom", source))
    assure_directory(cache_dir)

    if kind == "levels":
        filename = os.sep.join((cache_dir, element + str(nele) + ".nc"))
        if not os.path.exists(filename) or force_download:
            ds = get_levels(element, nele)
            ds.to_netcdf(filename)
    elif kind == "lines":
        filename = os.sep.join((cache_dir, element + str(nele) + "_lines.nc"))
        if not os.path.exists(filename) or force_download:
            ds = get_lines(element, nele)
            ds.to_netcdf(filename)
    return xr.load_dataset(filename)


def diatomic_molecules(
    molecule, force_download=False, source="nist", cache_dir=_default_cache_dir
):
    import xarray as xr

    if source not in ["nist"]:
        raise NotImplementedError("Only NIST is available yet.")

    cache_dir = os.sep.join((_default_cache_dir, "diatomic_molecules", source))
    assure_directory(cache_dir)

    filename = os.sep.join((cache_dir, molecule + ".nc"))
    if not os.path.exists(filename) or force_download:
        ds = download_diatomic_molecule_nist(molecule)
        ds.to_netcdf(filename)
    return xr.load_dataset(filename)


def download_diatomic_molecule_nist(molecule):
    """
    Download molecular data if not exist.
    They are saved in 'THIS_DIR'/molecule/nist/'molecule'
    """
    import xarray as xr
    import pandas as pd
    import urllib.request

    # support some notation fluctuation
    notations = {
        "HO": "Hydroxyl-radical",
        "DO": "Hydroxyl-d",
        "TO": "Hydroxyl-t",
        "OH": "Hydroxyl-radical",
        "OD": "Hydroxyl-d",
        "OT": "Hydroxyl-t",
    }
    molecule = notations.get(molecule, molecule)

    def robust_float(s):
        if isinstance(s, (float, int)):
            return float(s)
        s = s.strip()
        # remove parenthesis
        if s[0] == "(":
            s = s[s.find("(") + 1 : s.find(")")]
        # if there is a space, remove it
        s = s.split(" ")[0]
        try:
            return float(s)
        except ValueError:
            return np.nan

    def to_xarray(table):
        n = len(table.index)
        data = []
        
        try:
            pandas_version = [int(v) for v in pd._version.get_versions()['version'].split('.')]
        except AttributeError:
            pandas_version = [int(v) for v in pd.__version__.split('.')]
        if pandas_version[0] > 1 or (pandas_version[0] == 1 and pandas_version[1] > 3):
            # 'State', 'Te', 'ωe', 'ωexe', 'ωeye', 'Be', 'αe', 'γe', 'De', 'βe', 're', 'Trans.', 'ν00'
            float_columns = {
                "Te": "Te",
                "ωe": "we",
                "ωexe": "wexe",
                "ωeye": "weye",
                "Be": "Be",
                "αe": "alpha_e",
                "γe": "gamma_e",
                "De": "De",
                "βe": "beta_e",
                "re": "re",
                "Trans.": "trans.",
                "ν00": "nu00",
            }
        else:
            float_columns = {
                "Te": "Te",
                "e": "we",
                "exe": "wexe",
                "eye": "weye",
                "Be": "Be",
                "e.1": "alpha_e",
                "e.2": "gamma_e",
                "De": "De",
                "e.3": "beta_e",
                "re": "re",
                "Trans.": "trans.",
                "00": "nu00",
            }
        string_columns = {
            "State": "state",
        }
        for i in range(n):
            series = table.loc[i]
            Te = robust_float(series["Te"])
            if Te == Te:
                # valid data
                res = xr.Dataset(
                    {
                        item: robust_float(series[key])
                        for key, item in float_columns.items()
                    }
                )
                for key, col in string_columns.items():
                    res[col] = series[key]
                data.append(res)
        return xr.concat(data, dim="state")

    def download(name):
        url = "https://webbook.nist.gov/cgi/cbook.cgi?Name={}&Units=SI".format(name)
        header = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest"
        }
        req = urllib.request.Request(url=url, headers=header)

        with urllib.request.urlopen(req) as f:
            lines = f.read().decode("utf-8").split("\n")

        for line in lines:
            if "Constants of diatomic molecules" in line:
                url = line[line.find("/cgi") : line.find('#Diatomic">Constants')]
                break
        url = "https://webbook.nist.gov" + url
        print("downloading from {}".format(url))
        header = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest"
        }
        r = requests.get(url, headers=header)
        tables = pd.read_html(r.text, flavor="bs4", match="Diatomic constants ")
        ds = to_xarray(tables[0])
        ds.attrs["name"] = name
        ds.attrs["url"] = url
        return ds

    ds = download(molecule)
    return ds
