import os
import numpy as np

THIS_DIR = os.path.dirname(__file__) + '/'
DIATOMIC_DIR = THIS_DIR + 'downloaded/diatomic_molecules/'


def assure_directory(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)


assure_directory(THIS_DIR)
assure_directory(THIS_DIR + 'downloaded/')
assure_directory(DIATOMIC_DIR)
assure_directory(DIATOMIC_DIR + 'nist/')


def diatomic_molecules(molecule, force_download=False, source='nist'):
    import xarray as xr

    filename = DIATOMIC_DIR + source + '/' + molecule + '.nc'
    if source not in ['nist']:
        raise NotImplementedError('Only NIST is available yet.')
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

