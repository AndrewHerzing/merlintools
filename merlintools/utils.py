import numpy as np
from scipy import optimize
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

e = 1.602e-19       # Charge of electron (Coulombs)
m0 = 9.109e-31      # Rest mass of electron (kg)
m0c2 = 511          # Rest energy of electron (keV)
h = 6.626e-34       # Planck's constant
c = 2.998e8         # Speed of light in vacuum (m/s)
Na = 6.0221409e23   # Avogadro's number


def mrads_to_hkl(angle, voltage):
    """
    Convert from an diffraction angle (mrads) to lattice spacing (nm)

    Args
    ----------
    mrads : float
        Scattering angle in mrads
    voltage : float or int
        Electron beam voltage (kV)

    Returns
    ----------
    d : float
        Lattice spacing in nanometers
    """

    wavelength = voltage_to_wavelength(300, True)
    d = wavelength / (2 * np.sin(angle / 1000))
    return d


def mrads_to_k(angle, voltage):
    """
    Convert from an angular measurement (mrads) to reciprocal space (nm^-1)

    Args
    ----------
    mrads : float
        Scattering angle in mrads
    voltage : float or int
        Electron beam voltage (kV)

    Returns
    ----------
    k : float
        Reciprocal lattice spacing in either inverse nanometers
    """
    wavelength = voltage_to_wavelength(voltage, True)
    k = (2 * np.sin(angle / 1000)) / wavelength
    return k


def k_to_mrads(k, voltage):
    """
    Convert from a reciprocal space (nm^-1) value an angular value (mrads)

    Args
    ----------
    k : float
        Reciprocal lattice spacing in either inverse nanometers
    voltage : float or int
        Electron beam voltage (kV)

    Returns
    ----------
    angle : float
        Scattering angle in mrads
    """
    wavelength = voltage_to_wavelength(voltage, True)
    angle = 1000 * np.arcsin(k * wavelength / 2)
    return angle


def voltage_to_wavelength(voltage, relativistic=False):
    """
    Calculates electron wavelength given voltage

    Args
    ----------
    voltage : float
        Accelerating voltage (in kV)

    relativistic : bool
        If True, calculate the relatistically corrected wavelength

    Returns
    ----------
    wavelength : float
        Calculated wavelength (in nanometers)
    """
    if relativistic:
        correction = (1 + ((e * voltage * 1000) / (2 * m0 * c**2)))
        wavelength = h / np.sqrt(2 * m0 * e * voltage * 1000 * correction)
    else:
        wavelength = h / np.sqrt(2 * m0 * e * voltage * 1000)
    return wavelength * 1e9


def get_relativistic_mass(voltage):
    """
    Calculates relativistic mass given voltage

    Args
    ----------
    voltage : float
        Accelerating voltage (in kV)

    Returns
    ----------
    rel_mass : float
        Calculated relativistic mass (in kg)
    """
    rel_mass = (1 + voltage / m0c2) * m0
    return rel_mass


def fit_func(x, A, exp):
    return A*x**exp


def extrapolate_calibration(cl, calibrations):
    xdata = [int(i) for i in calibrations.keys()]
    ydata = [calibrations[i] for i in calibrations.keys()]
    res, cov = optimize.curve_fit(fit_func, xdata, ydata)
    return res


# Calibrations in mrads/pixel
cal_80kV = {'38': 1.511111111111111,
            '48': 1.1333333333333333,
            '60': 0.9066666666666666,
            '77': 0.6799999999999999,
            '100': 0.5230769230769231,
            '130': 0.39999999999999997,
            '160': 0.3238095238095238,
            '195': 0.26153846153846155,
            '245': 0.20923076923076922,
            '300': 0.17662337662337663,
            '380': 0.13877551020408163}

cal_200kV = {}

cal_300kV = {'77': 16.5,
             '130': 26.5,
             '160': 32.5,
             '195': 40.5,
             '245': 51.5,
             '300': 62.5}


def get_calibration(beam_energy, cl, units='mrads'):
    beam_energy = int(beam_energy)
    cl = str(int(cl))

    if beam_energy == 80:
        calibration_dictionary = cal_80kV
    elif beam_energy == 200:
        calibration_dictionary = cal_200kV
    elif beam_energy == 300:
        calibration_dictionary = cal_300kV
    else:
        raise(ValueError, "No calibration for beam energy: %s. "
                          "Must be 80, 200, or 300." % str(beam_energy))
    if cl in calibration_dictionary.keys():
        calibration = calibration_dictionary[cl]
        logger.info("Camera length found in calibration table.")
    else:
        calibration = extrapolate_calibration(cl, calibration_dictionary)
        logger.info("Camera length not in calibration table. "
                    "Calibration will be extrapolated.")

    if units == 'mrads':
        pass
    elif units == 'q':
        wavelength = voltage_to_wavelength(beam_energy, True)
        calibration = (2 * np.sin(calibration / 1000)) / wavelength
    elif units == 'angstroms':
        wavelength = voltage_to_wavelength(beam_energy, True)
        calibration = (2 * np.sin(calibration / 1000)) / wavelength
        calibration = calibration / (2*np.pi)
    else:
        raise(ValueError,
              "Units (%s) not understood. "
              "Must be 'mrads', 'q', or 'angstroms'" % units)
    return calibration
