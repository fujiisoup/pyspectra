import os
import numpy as np
from .atoms import ATOMIC_SYMBOLS


_default_cache_dir = os.sep.join(
    (os.path.expanduser("~"), ".pyspectra_data")
)


def assure_directory(dirname):
    if not os.path.exists(dirname):
        parent_dir = dirname[:dirname.rfind(os.sep)]
        assure_directory(parent_dir)
        os.mkdir(dirname)


def atom_levels(
        element, nele=None, force_download=False, source='nist',
        cache_dir=_default_cache_dir
    ):
    return _atom(
        'levels', element, nele, force_download, source,
        cache_dir)

def atom_lines(
        element, nele=None, force_download=False, source='nist',
        cache_dir=_default_cache_dir
    ):
    return _atom(
        'lines', element, nele, force_download, source,
        cache_dir)

def _atom(
        kind,
        element, nele=None, force_download=False, source='nist',
        cache_dir=_default_cache_dir
    ):
    import xarray as xr
    from ._data_atom_nist import get_levels, get_lines

    if nele is None:  # neutral is assumed
        nele = ATOMIC_SYMBOLS.index(element)

    if source not in ['nist']:
        raise NotImplementedError('Only NIST is available yet.')

    cache_dir = os.sep.join((
        _default_cache_dir, "atom", source))
    assure_directory(cache_dir)
    
    if kind == 'levels':
        filename = os.sep.join((cache_dir, element + str(nele) + '.nc'))
        if not os.path.exists(filename) or force_download:
            ds = get_levels(element, nele)
            ds.to_netcdf(filename)
    elif kind == 'lines':
        filename = os.sep.join((cache_dir, element + str(nele) + '_lines.nc'))
        if not os.path.exists(filename) or force_download:
            ds = get_levels(element, nele)
            ds.to_netcdf(filename)
    return xr.load_dataset(filename)


def diatomic_molecules(
        molecule, force_download=False, source='nist',
        cache_dir=_default_cache_dir
    ):
    import xarray as xr

    if source not in ['nist']:
        raise NotImplementedError('Only NIST is available yet.')

    cache_dir = os.sep.join((
        _default_cache_dir, "diatomic_molecules", source))
    assure_directory(cache_dir)

    filename = os.sep.join((cache_dir, molecule + '.nc'))
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

    def robust_float(s):
        if isinstance(s, (float, int)):
            return float(s)
        s = s.strip()
        # remove parenthesis
        if s[0] == '(':
            s = s[s.find('(') + 1: s.find(')')]
        # if there is a space, remove it
        s = s.split(' ')[0]
        try:
            return float(s)
        except ValueError:
            return np.nan

    def to_xarray(table):
        n = len(table.index)
        data = []
        float_columns = {
            'Te': 'Te', 
            'e': 'we', 'exe': 'wexe', 'eye': 'weye',
            'Be': 'Be', 'e.1': 'alpha_e', 'e.2': 'gamma_e',
            'De': 'De', 'e.3': 'beta_e', 're': 're',
            'Trans.': 'trans.', '00': 'nu00'}
        string_columns = {'State': 'state', }
        for i in range(n):
            series = table.loc[i]
            Te = robust_float(series['Te'])
            if Te == Te:
                # valid data
                res = xr.Dataset({
                    item: robust_float(series[key]) for key, item 
                    in float_columns.items()
                })
                for key, col in string_columns.items():
                    res[col] = series[key]
                data.append(res)
        return xr.concat(data, dim='state')    

    def download(name):
        url = 'https://webbook.nist.gov/cgi/cbook.cgi?Name={}&Units=SI'.format(name)
        req = urllib.request.Request(url=url)
        
        with urllib.request.urlopen(req) as f:
            lines = f.read().decode('utf-8').split('\n')
        
        for line in lines:
            if 'Constants of diatomic molecules' in line:
                url = line[line.find('/cgi'): line.find('#Diatomic">Constants')]
                break
        url = 'https://webbook.nist.gov' + url
        print('downloading from {}'.format(url))
        tables = pd.read_html(url, flavor='bs4', match='Diatomic constants ')
        ds = to_xarray(tables[0])
        ds.attrs['name'] = name
        return ds

    ds = download(molecule)
    return ds

