"""
Download nist level page
"""
from collections import OrderedDict
from functools import total_ordering
import urllib.request
import numpy as np
import xarray as xr
from .atoms import ATOMIC_SYMBOLS


class DataNotFoundError(ValueError):
    pass

def get_level_url(atom, nele):
    charge = ATOMIC_SYMBOLS.index(atom) - nele

    queries = OrderedDict()
    queries['spectrum'] = atom + str(charge)
    queries['units'] = '1'  # eV
    queries['format'] = '3' # tab-deliminated
    queries['output'] = '0' # entire
    queries['page_size'] = '15'
    queries['multiplet_ordered'] = 'checked'
    queries['conf_out'] = 'checked'
    queries['term_out'] = 'checked'
    queries['level_out'] = 'checked'
    queries['unc_out'] = 'checked'
    queries['j_out'] = 'checked'
    queries['lande_out'] = 'checked'
    queries['perc_out'] = 'checked'
    queries['biblio'] = 'on'

    url = 'https://physics.nist.gov/cgi-bin/ASD/energy1.pl?'
    url += '&'.join([k + '=' + v for k, v in queries.items()])
    return url


def get_line_url(atom, nele):
    # https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=Fe+Be-like&limits_type=0&low_w=&upp_w=&unit=1&de=0&format=3&\
    # line_out=0&remove_js=on&en_unit=1&output=0&bibrefs=1&page_size=15&show_obs_wl=1&show_calc_wl=1&unc_out=1&order_out=0&\
    # max_low_enrg=&show_av=3&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&f_out=on&intens_out=on&max_str=&allowed_out=1&
    # forbid_out=1&min_accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out=on&submit=Retrieve+Data

    charge = ATOMIC_SYMBOLS.index(atom) - nele

    queries = OrderedDict()
    queries['spectra'] = atom + str(charge)
    queries['limits_type'] = '0'  # no limit
    queries['low_w'] = ''
    queries['upp_w'] = ''
    queries['unit'] = '1'  # eV
    queries['de'] = '0'  # ?
    queries['format'] = '3' # tab-deliminated
    queries['line_out'] = '0'  # ?
    queries['remove_js'] = 'on'
    queries['en_unit'] = '1'
    queries['output'] = '0' # entire
    queries['bibrefs'] = '1' # with reference
    queries['page_size'] = '15'
    queries['show_obs_wl'] = '1'
    queries['show_calc_wl'] = '1'
    queries['unc_out'] = '1'
    queries['order_out'] = '0'
    queries['max_low_enrg'] = ''
    queries['show_av'] = '3'  # show air-vaccum ?
    queries['max_upp_enrg'] = ''
    queries['tsb_value'] = '0'
    queries['min_str'] = ''
    queries['A_out'] = '0'
    queries['f_out'] = 'on'
    queries['S_out'] = 'on'
    queries['intens_out'] = 'on'
    queries['max_str'] = ''
    queries['allowed_out'] = '1'
    queries['forbid_out'] = '1'
    queries['min_accur'] = ''
    queries['min_intens'] = ''
    queries['conf_out'] = 'on'
    queries['term_out'] = 'on'
    queries['enrg_out'] = 'on'
    queries['J_out'] = 'on'
    queries['submit'] = 'Retrieve+Data'

    url = 'https://physics.nist.gov/cgi-bin/ASD/lines1.pl?'
    url += '&'.join([k + '=' + v for k, v in queries.items()])
    return url


def get_levels(atom, nele):
    """
    Scrape NIST web page.
    """
    url = get_level_url(atom, nele)
    contents = urllib.request.urlopen(url).read().decode('utf-8')
    lines = contents.split('\n')
    data = _parse_levels(lines)

    # renaming coordinates
    renames = {}
    for c in data.coords:
        if ' ' in c:
            renames[c] = c.replace(' ', '')
    return data.rename(renames).dropna('ilev')


def get_lines(atom, nele):
    """
    Scrape NIST web page.
    """
    url = get_line_url(atom, nele)
    contents = urllib.request.urlopen(url).read().decode('utf-8')
    lines = contents.split('\n')
    data = _parse_lines(lines)

    # no data found
    if 'obs_wl_vac(nm)' not in data:
        raise DataNotFoundError

    # renaming coordinates
    renames = {}
    for c in data.coords:
        if ' ' in c:
            renames[c] = c.replace(' ', '')
    data = data.rename(renames)
    # unit should be in attrs
    renames = {}
    renames['obs_wl_vac(nm)'] = 'obs_wl_vac'
    renames['obs_wl_vac(nm)_uncertain'] = 'obs_wl_vac_uncertain'
    renames['ritz_wl_vac(nm)'] = 'ritz_wl_vac'
    renames['ritz_wl_vac(nm)_uncertain'] = 'ritz_wl_vac_uncertain'
    renames['Aki(s^-1)'] = 'Aki'
    renames['Aki(s^-1)_uncertain'] = 'Aki_uncertain'
    renames['S(a.u.)'] = 'S'
    renames['S(a.u.)_uncertain'] = 'S_uncertain'
    data = data.rename(renames)
    data['obs_wl_vac'].attrs['unit'] = 'nm'
    data['obs_wl_vac_uncertain'].attrs['unit'] = 'nm'
    data['ritz_wl_vac'].attrs['unit'] = 'nm'
    data['ritz_wl_vac_uncertain'].attrs['unit'] = 'nm'
    data['Aki'].attrs['unit'] = 's^-1'
    data['Aki_uncertain'].attrs['unit'] = 's^-1'
    data['S'].attrs['unit'] = 'a.u.'
    data['S_uncertain'].attrs['unit'] = 'a.u.'
    return data


def parity_term(term):
    if len(term) == 0:
        return np.nan, term
    elif term[-1] == '*':
        return 'odd', term[:-1]
    else:
        return 'even', term


def _parse_levels(lines):
    data = OrderedDict()

    headers = lines[0].split('\t')
    keys = [key.strip("\"") for key in headers]
    for key in keys:
        data[key.strip()] = []
        data[key.strip() + '_uncertain'] = []
    data['Level(eV)_is_theoretical'] = []
    data['Level(eV)_is_predicted'] = []
    data['Level(eV)_digits'] = []
    data['parity'] = []

    ionization_limit = None
    lines = [[it.strip("\"").strip() for it in line.split('\t')]
             for line in lines]

    for i, items in enumerate(lines):
        if len(items) < 4:
            break
        # first ionization limit
        if items[1].lower() == 'limit' and ionization_limit is None:
            ionization_limit = _energy(items[3])
            ionization_limit_err = _energy(items[4])
        else:
            # duplicate lines to take care of "J=2,3,4" representation
            if ',' in items[2]:
                new_line = items.copy()
                new_line[2] = items[2][items[2].find(',') + 1:]
                items[2] = items[2][:items[2].find(',')]
                lines.insert(i + 1, new_line)

            for key, item in zip(keys, items):
                if len(item) == 0 or item[-1] == '?':
                    data[key + '_uncertain'].append(True)
                    item = item[:-1]
                else:
                    data[key + '_uncertain'].append(False)

                if key == 'J':
                    data['J'].append(_two_j(item))
                elif key == 'Term':
                    parity, term = parity_term(item)
                    data['parity'].append(parity)
                    data[key].append(term)
                elif key == 'Level (eV)':
                    eng, digit = _energy(item, return_digit=True)
                    data['Level (eV)'].append(eng)
                    data['Level(eV)_digits'].append(digit)
                    if len(item) > 0:
                        data['Level(eV)_is_predicted'].append(item[-1] == ']')
                        data['Level(eV)_is_theoretical'].append(item[-1] == ')')
                    else:
                        data['Level(eV)_is_predicted'].append(True)
                        data['Level(eV)_is_theoretical'].append(True)
                elif key == 'Uncertainty (eV)':
                    data['Uncertainty (eV)'].append(_energy(item))
                elif key == 'Reference':  # propagate reference forward
                    if item != '':
                        # only consider the reference number
                        data[key].append(item[:5])
                    elif len(data[key]) == 0:
                        data[key].append('missing')
                    else:
                        data[key].append(data[key][-1])
                else:
                    data[key].append(item)

    # convert to xarray
    energy = data.pop('Level (eV)')
    da = xr.DataArray(energy, dims='ilev',
                      coords={k: ('ilev', it) for k, it in data.items()
                              if k != ''},
                      attrs={'ionization_limit': ionization_limit,
                             'ionization_limit_err': ionization_limit_err,
                             'unit': 'eV'},
                      name='energy')
    if 'j' in da.coords:
        da = da.rename({'j': 'J'})
    return da


def _parse_lines(lines):
    data = OrderedDict()

    headers = lines[0].split('\t')
    keys = [key.strip("\"") for key in headers]
    for key in keys:
        data[key.strip()] = []
        data[key.strip() + '_uncertain'] = []
    if 'conf_i' in keys:
        data['parity_i'] = []
    if 'conf_k' in keys:
        data['parity_k'] = []

    ionization_limit = None
    lines = [[it.strip("\"").strip() for it in line.split('\t')]
             for line in lines]

    for i, items in enumerate(lines[1:]):
        if len(items) < 4:
            break
        # duplicate lines to take care of "J=2,3,4" representation
        if ',' in items[2]:
            new_line = items.copy()
            new_line[2] = items[2][items[2].find(',') + 1:]
            items[2] = items[2][:items[2].find(',')]
            lines.insert(i + 1, new_line)

        for key, item in zip(keys, items):
            if len(item) == 0 or item[-1] == '?':
                data[key + '_uncertain'].append(True)
                item = item[:-1]
            else:
                data[key + '_uncertain'].append(False)

            if key == 'J_i':
                data['J_i'].append(_two_j(item))
            elif key == 'J_k':
                data['J_k'].append(_two_j(item))
            elif key in ['obs_wl_vac(nm)', 'ritz_wl_vac(nm)', 'unc_ritz_wl',
                         'intens', 'Aki(s^-1)', 'fik', 'Acc', 'Ei(eV)',
                         'Ek(eV)', 'S(a.u.)']:
                data[key].append(_energy(item))
            elif key == 'term_i':
                parity, term = parity_term(item)
                data['parity_i'].append(parity)
                data[key].append(term)
            elif key == 'term_k':
                parity, term = parity_term(item)
                data['parity_k'].append(parity)
                data[key].append(term)
            else:
                data[key].append(item)

    # convert to xarray
    da = xr.Dataset({k: ('itrans', it) for k, it in data.items() if k != ''})
    return da


def _energy(string, return_digit=False):
    value = np.nan
    try:
        value = float(string)
        new_string = string
    except ValueError:
        new_string = ''
        for s in string:
            if s == '<':
                break
            if s in '0123456789.':
                new_string += s
        try:
            value = float(new_string)
        except ValueError:
            pass
    if return_digit:
        digit = len(new_string) - new_string.find('.') - 1
        return value, digit
    else:
        return value


def _two_j(string):
    """ Multiply string (possibly half integer) into integer """
    try:
        value = int(string) * 2
    except ValueError:
        strings = string.split('/')
        try:
            value = int(strings[0])
        except ValueError:
            value = -1  # invalid J is converted to -1
    return value


CLOSED = ['1s2', '2s2', '2p6', '3s2', '3p6', '3d10', '4s2', '4p6', '4d10',
          '5s2']


class ConfigJ(object):
    """ A simple class to find the same pair of configuration and J """
    def __init__(self, config, j):
        self.config = str(config)
        self.j = int(j)

    def __eq__(self, other):
        return (self.config == other.config) and (self.j == other.j)

    def __lt__(self, other):
        if self.config == other.config:
            return self.j < other.j
        return self.config < other.config

    def __hash__(self):
        return hash(self)

    def __str__(self):
        return '({})-{}'.format(self.config, self.j)

    def __repr__(self):
        return self.__str__()


def match_levels(self, other):
    """ find a pair of energy levels """
    config_j = np.array([
        ConfigJ(sname.values.item(), j.values.item())
        for sname, j in zip(self['sname'], self['j'])],
        dtype=object)
    levels = np.unique(config_j)

    level_pairs = []
    for level in levels:
        src = self.isel(ilev=(self['sname']==level.config)
            &(self['j']==level.j))
        dest = other.isel(ilev=(other['sname']==level.config)
            &(other['j']==level.j))
        if len(src['ilev']) > len(dest['ilev']):
            raise ValueError('Invalid level mismatch for {}.\n'.format(level) +
                             'Found \n{}\n and \n{}'.format(src, dest))
        level_pairs.append((src, dest))
    return level_pairs


def fuse(levels, lines):
    """ Fuse lines and levels data """
    levels = levels.isel(ilev=levels['J'] > -1)
    levels = levels.set_index(
        ilev=['Configuration', 'Term', 'J'])
    v, idx, counts = np.unique(levels['ilev'], return_counts=True,
                               return_index=True)
    if (counts > 1).any():
        print(v[counts > 1])
        levels = levels.isel(ilev=idx)
    index = levels.get_index('ilev')

    if any(k not in lines for k in ['conf_i', 'term_i', 'J_i', 'conf_k', 'term_k',
                                    'J_k', 'Aki']):
        raise DataNotFoundError

    lines = lines.isel(itrans=(
        (~lines['Aki'].isnull()) *
         (lines['conf_i'] != '') * (lines['J_i'] > -1) *
         (lines['term_i'] != '') *
         (lines['conf_k'] != '') * (lines['J_k'] > -1) *
         (lines['term_k'] != '')))

    idx_i = []
    idx_k = []

    def maybe_convert_to_int(index):
        if isinstance(index, slice):
            assert index.start == index.stop - 1
            return index.start
        return index

    for i in range(len(lines['itrans'])):
        l = lines.isel(itrans=i)
        idx_i.append(maybe_convert_to_int(index.get_loc(
            (l['conf_i'].values.item(), l['term_i'].values.item(),
             l['J_i'].values.item()))))
        idx_k.append(maybe_convert_to_int(index.get_loc(
            (l['conf_k'].values.item(), l['term_k'].values.item(),
             l['J_k'].values.item()))))

    idx_i = xr.DataArray(idx_i, dims=['i'])
    idx_k = xr.DataArray(idx_k, dims=['i'])
    if len(idx_i) == 0 or len(idx_k) == 0:
        raise DataNotFoundError

    data = xr.Dataset({'levels': levels})
    shape = (len(data['ilev']), len(data['ilev']))
    for k in ['Aki', 'fik', 'Aki_uncertain', 'fik_uncertain', 'S',
              'S_uncertain']:
        data[k] = ('ilev', 'ilev2'), np.full(shape, np.nan), lines[k].attrs
        data[k][idx_k, idx_i] = lines[k].values

    # line strength should be symmetric
    for k in ['S', 'S_uncertain']:
        data[k][idx_i, idx_k] = lines[k].values
    return data.reset_index('ilev')