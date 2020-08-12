import numpy as np
from pyspectra import molecules


def test_OH_X2():
    """
    randomly choose levels from Table 27
    """
    qnums = []
    levels = []
    #             v,   J, parity, 3/2 or 1/2
    # qnums.append([0, 0.5,     +1, 1])  # F1e
    # levels.append(0.0000)
    qnums.append([0, 0.5,     +1, 2])  # F2e
    levels.append(88.1066)
    # qnums.append([0, 0.5,     -1, 1])  # F1f
    # levels.append(0.0000)
    qnums.append([0, 0.5,     -1, 2])  # F2f
    levels.append(88.2642)

    qnums.append([0, 1.5,     +1, 1])  # F1e
    levels.append(-38.2480)
    qnums.append([0, 1.5,     +1, 2])  # F2e
    levels.append(149.3063)
    qnums.append([0, 1.5,     -1, 1])  # F1f
    levels.append(-38.1926)
    qnums.append([0, 1.5,     -1, 2])  # F2f
    levels.append(149.5662)

    qnums.append([0, 10.5,     +1, 1])  # F1e
    levels.append(1976.8000)
    qnums.append([0, 10.5,     +1, 2])  # F2e
    levels.append(2414.9290)
    qnums.append([0, 10.5,     -1, 1])  # F1f
    levels.append(1981.4015)
    qnums.append([0, 10.5,     -1, 2])  # F2f
    levels.append(2412.0731)

    v, J, parity, spin = np.array(qnums).T
    energies = molecules.level_OH_X2(v, J, parity, spin)
    # for lev, en in zip(levels, energies):
    #     print('{} : {}'.format(lev, en))
    assert np.allclose(energies, levels, atol=0.1)

    qnums = []
    levels = []
    qnums.append([4, 13.5,     +1, 1])  # F1e
    levels.append(16062.2776)
    qnums.append([4, 13.5,     +1, 2])  # F2e
    levels.append(16522.0293)
    qnums.append([4, 13.5,     -1, 1])  # F1f
    levels.append(16068.1260)
    qnums.append([4, 13.5,     -1, 2])  # F2f
    levels.append(16517.9751)

    v, J, parity, spin = np.array(qnums).T
    energies = molecules.level_OH_X2(v, J, parity, spin)
    # for lev, en in zip(levels, energies):
    #     print('{} : {}'.format(lev, en))
    assert np.allclose(energies, levels, atol=0.1)
