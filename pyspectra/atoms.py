ATOMIC_SYMBOLS = [
    '__dummy__',  # no Z=0
    # period 1
    'H', 'He',
    # period 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # period 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # period 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
    'Ge', 'As', 'Se', 'Br', 'Kr',
    # period 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # period 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
    'At', 'Rn',
    # period 7
    'Fr', 'Ra', 'Ac', 'Th', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 
]


ATOMIC_MASS = [
    1.0079,  #	Hydrogen	H	1
    4.0026,  #	Helium	He	2
    6.941,  #	Lithium	Li	3
    9.0122,  #	Beryllium	Be	4
    10.811,  #	Boron	B	5
    12.0107,  #	Carbon	C	6
    14.0067,  #	Nitrogen	N	7
    15.9994,  #	Oxygen	O	8
    18.9984,  #	Fluorine	F	9
    20.1797,  #	Neon	Ne	10
    22.9897,  #	Sodium	Na	11
    24.305,  #	Magnesium	Mg	12
    26.9815,  #	Aluminum	Al	13
    28.0855,  #	Silicon	Si	14
    30.9738,  #	Phosphorus	P	15
    32.065,  #	Sulfur	S	16
    35.453,  #	Chlorine	Cl	17
    39.0983,  #	Potassium	K	19
    39.948,  #	Argon	Ar	18
    40.078,  #	Calcium	Ca	20
    44.9559,  #	Scandium	Sc	21
    47.867,  #	Titanium	Ti	22
    50.9415,  #	Vanadium	V	23
    51.9961,  #	Chromium	Cr	24
    54.938,  #	Manganese	Mn	25
    55.845,  #	Iron	Fe	26
    58.6934,  #	Nickel	Ni	28
    58.9332,  #	Cobalt	Co	27
    63.546,  #	Copper	Cu	29
    65.39,  #	Zinc	Zn	30
    69.723,  #	Gallium	Ga	31
    72.64,  #	Germanium	Ge	32
    74.9216,  #	Arsenic	As	33
    78.96,  #	Selenium	Se	34
    79.904,  #	Bromine	Br	35
    83.8,  #	Krypton	Kr	36
    85.4678,  #	Rubidium	Rb	37
    87.62,  #	Strontium	Sr	38
    88.9059,  #	Yttrium	Y	39
    91.224,  #	Zirconium	Zr	40
    92.9064,  #	Niobium	Nb	41
    95.94,  #	Molybdenum	Mo	42
    98,  #	Technetium	Tc	43
    101.07,  #	Ruthenium	Ru	44
    102.9055,  #	Rhodium	Rh	45
    106.42,  #	Palladium	Pd	46
    107.8682,  #	Silver	Ag	47
    112.411,  #	Cadmium	Cd	48
    114.818,  #	Indium	In	49
    118.71,  #	Tin	Sn	50
    121.76,  #	Antimony	Sb	51
    126.9045,  #	Iodine	I	53
    127.6,  #	Tellurium	Te	52
    131.293,  #	Xenon	Xe	54
    132.9055,  #	Cesium	Cs	55
    137.327,  #	Barium	Ba	56
    138.9055,  #	Lanthanum	La	57
    140.116,  #	Cerium	Ce	58
    140.9077,  #	Praseodymium	Pr	59
    144.24,  #	Neodymium	Nd	60
    145,  #	Promethium	Pm	61
    150.36,  #	Samarium	Sm	62
    151.964,  #	Europium	Eu	63
    157.25,  #	Gadolinium	Gd	64
    158.9253,  #	Terbium	Tb	65
    162.5,  #	Dysprosium	Dy	66
    164.9303,  #	Holmium	Ho	67
    167.259,  #	Erbium	Er	68
    168.9342,  #	Thulium	Tm	69
    173.04,  #	Ytterbium	Yb	70
    174.967,  #	Lutetium	Lu	71
    178.49,  #	Hafnium	Hf	72
    180.9479,  #	Tantalum	Ta	73
    183.84,  #	Tungsten	W	74
    186.207,  #	Rhenium	Re	75
    190.23,  #	Osmium	Os	76
    192.217,  #	Iridium	Ir	77
    195.078,  #	Platinum	Pt	78
    196.9665,  #	Gold	Au	79
    200.59,  #	Mercury	Hg	80
    204.3833,  #	Thallium	Tl	81
    207.2,  #	Lead	Pb	82
    208.9804,  #	Bismuth	Bi	83
    209,  #	Polonium	Po	84
    210,  #	Astatine	At	85
    222,  #	Radon	Rn	86
    223,  #	Francium	Fr	87
    226,  #	Radium	Ra	88
    227,  #	Actinium	Ac	89
    231.0359,  #	Protactinium	Pa	91
    232.0381,  #	Thorium	Th	90
    237,  #	Neptunium	Np	93
    238.0289,  #	Uranium	U	92
    243,  #	Americium	Am	95
    244,  #	Plutonium	Pu	94
    247,  #	Curium	Cm	96
    247,  #	Berkelium	Bk	97
    251,  #	Californium	Cf	98
    252,  #	Einsteinium	Es	99
    257,  #	Fermium	Fm	100
    258,  #	Mendelevium	Md	101
    259,  #	Nobelium	No	102
    261,  #	Rutherfordium	Rf	104
    262,  #	Lawrencium	Lr	103
    262,  #	Dubnium	Db	105
    264,  #	Bohrium	Bh	107
    266,  #	Seaborgium	Sg	106
    268,  #	Meitnerium	Mt	109
    272,  #	Roentgenium	Rg	111
    277,  #	Hassium	Hs	108
    ]
    # Read more: https://www.lenntech.com/periodic/mass/atomic-mass.htm#ixzz5FXLOrRO4


ROMAN_NUMBER = [
    '__dummy__',
    'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
    'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX',
    'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI', 'XXVII', 'XXVIII', 'XXIX', 'XXX',
    'XXXI', 'XXXII', 'XXXIII', 'XXXIV', 'XXXV', 'XXXVI', 'XXXVII', 'XXXVIII', 'XXXIX', 'XL',
    'XLI', 'XLII', 'XLIII', 'XLIV', 'XLV', 'XLVI', 'XLVII', 'XLVIII', 'XLIX', 'L',
    'LI', 'LII', 'LIII', 'LIV', 'LV', 'LVI', 'LVII', 'LVIII', 'LIX', 'LX',
    'LXI', 'LXII', 'LXIII', 'LXIV', 'LXV', 'LXVI', 'LXVII', 'LXVIII', 'LXIX', 'LXX',
]

def decode_charge(string):
    """
    Decode 'FeI' into 'Fe' and 0 (charge)
    """
    if string[:2] in ATOMIC_SYMBOLS:
        return string[:2], ROMAN_NUMBER.index(string[2:]) - 1
    return string[:1], ROMAN_NUMBER.index(string[1:]) - 1