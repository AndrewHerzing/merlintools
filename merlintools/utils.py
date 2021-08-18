import numpy as np
from scipy import optimize
import logging
import json
import os
import merlintools

merlintools_path = os.path.dirname(merlintools.__file__)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

e = 1.602e-19       # Charge of electron (Coulombs)
m0 = 9.109e-31      # Rest mass of electron (kg)
m0c2 = 511          # Rest energy of electron (keV)
h = 6.626e-34       # Planck's constant
c = 2.998e8         # Speed of light in vacuum (m/s)
Na = 6.0221409e23   # Avogadro's number


def get_lattice_spacings(material):
    """
    Return a dictionary of primary lattice spacings for a given material

    Args
    ----------
    material : str
        Identity of material for which to return hkl spacings.
        Must be 'AuPd', 'Au', or 'Si'.

    Returns
    ----------
    d : float
        Lattice spacing in nanometers
    """

    if material.lower() == 'aupd':
        hkls = {'111': 2.31, '200': 2.00, '220': 1.41,
                '311': 1.21, '222': 1.15}
        return hkls
    elif material.lower() == 'au':
        hkls = {'111': 2.36, '200': 2.04, '220': 1.44,
                '311': 1.23, '222': 1.18}
        return hkls
    elif material.lower() == 'si':
        hkls = {'111': 3.14, '200': 2.72, '220': 1.92,
                '311': 1.64, '222': 1.57}
        return hkls
    else:
        raise(ValueError, "Unknown material.  Must be 'Au-Pd', 'Au', or 'Si'")


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


def mrads_to_q(angle, voltage):
    """
    Convert from an angular value (mrads) to momentum value (nm^-1)

    Args
    ----------
    mrads : float
        Scattering angle in mrads
    voltage : float or int
        Electron beam voltage (kV)

    Returns
    ----------
    q : float
        Momentum transfer (2*pi*k) in inverse nanometers
    """
    wavelength = voltage_to_wavelength(voltage, True)
    q = 4 * np.pi / wavelength * np.sin(angle / 2000)
    return q


def mrads_to_k(angle, voltage):
    """
    Convert from an angular value (mrads) to reciprocal space (nm^-1)

    Args
    ----------
    mrads : float
        Scattering angle in mrads
    voltage : float or int
        Electron beam voltage (kV)

    Returns
    ----------
    k : float
        Reciprocal lattice spacing in inverse nanometers
    """
    wavelength = voltage_to_wavelength(voltage, True)
    k = 2 / wavelength * np.sin(angle / 2000)
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

def q_to_mrads(q, voltage):
    """
    Convert from momentum transfer (nm^-1) value an angular value (mrads)

    Args
    ----------
    q : float
        Momentum transfer (2*pi*k) in inverse nanometers
    voltage : float or int
        Electron beam voltage (kV)

    Returns
    ----------
    angle : float
        Scattering angle in mrads
    """
    wavelength = voltage_to_wavelength(voltage, True)
    angle = 1000 * np.arcsin(q / (2 * np.pi) * wavelength / 2)
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


def calc_dose_pixel(probe_current, pixel_size, dwell_time):
    """
    Calculate electron dose given probe current, dwell time, and pixel size

    Args
    ---------
    probe_current : float
        Probe current in amps

    dwell_time : float
        Dwell time per pixel in seconds

    pixel_size : float
        Pixel size in nanometers

    Returns
    ----------
    dose : float
        Calculated electron dose in electrons per square nanometer
    """
    dose = 6.242e18 * probe_current * dwell_time / pixel_size**2
    return dose

def calc_dose_probe(probe_current, probe_fwhm, dwell_time):
    '''
    Calulate electron based on the measured probe size
    
    Args
    ---------
    probe_current : float
        Measured probe current in amps
    probe_fwhm : float
        Measured probe FWHM in nanometers
    dwell_time : float
        Per pixel dwell time in seconds
    
    Returns
    ---------
    dose : float
        Calculated electron dose in electrons per square nanometer
    
    '''
    n_electrons = 6.242e18 * probe_current * dwell_time
    dose = n_electrons / (np.pi*(probe_fwhm/2)**2)
    return dose

def fit_func(x, A, exp):
    return A*x**exp


def extrapolate_calibration(cl, calibrations):
    xdata = [int(i) for i in calibrations.keys()]
    ydata = [calibrations[i] for i in calibrations.keys()]
    res, cov = optimize.curve_fit(fit_func, xdata, ydata)
    return fit_func(cl, *res)


# Calibrations in mrads/pixel
calibration_file_80kV = os.path.join(merlintools_path, "calibrations",
                                     "80kV_calibrations.json")
with open(calibration_file_80kV, 'r') as fp:
    cal_80kV = json.load(fp)

calibration_file_200kV = os.path.join(merlintools_path, "calibrations",
                                      "200kV_calibrations.json")
with open(calibration_file_200kV, 'r') as fp:
    cal_200kV = json.load(fp)

calibration_file_300kV = os.path.join(merlintools_path, "calibrations",
                                      "300kV_calibrations.json")
with open(calibration_file_300kV, 'r') as fp:
    cal_300kV = json.load(fp)


def get_calibration(beam_energy, cl, units='mrads'):
    """
    Returns reciprocal space calibration given the beam energy and
    camera length.

    Args
    ----------
    beam_energy : float
        Beam energy in keV

    cl : float
        Camera length in mm

    units : str
        Units for returned calibration.  Must be 'mrads', 'q', or 'k'. q and k
        are both in A^-1/pixel

    Returns
    ----------
    calibration : float
        Reciprocal space calibration in mrads/pixel or A^-1/pixel
    """

    beam_energy = int(beam_energy)
    if beam_energy == 303:
        beam_energy = 300
    cl = int(cl)

    if beam_energy == 80:
        calibration_dictionary = cal_80kV
    elif beam_energy == 200:
        calibration_dictionary = cal_200kV
    elif beam_energy == 300:
        calibration_dictionary = cal_300kV
    else:
        raise ValueError("No calibration for beam energy: %s. "
                         "Must be 80, 200, or 300." % str(beam_energy))
    if str(cl) in calibration_dictionary.keys():
        calibration = calibration_dictionary[str(cl)]
        logger.info("Camera length found in calibration table.")
    else:
        calibration =\
            np.round(extrapolate_calibration(cl, calibration_dictionary), 3)
        logger.info("Camera length not in calibration table. "
                    "Calibration will be extrapolated.")

    if units == 'mrads':
        pass
    elif units == 'q':
        wavelength = 10 * voltage_to_wavelength(beam_energy, True)
        calibration = (4 * np.pi * np.sin(calibration / 2000)) / wavelength
    elif units == 'k':
        wavelength = 10 * voltage_to_wavelength(beam_energy, True)
        calibration = (2 * np.sin(calibration / 2000)) / wavelength
    else:
        raise ValueError("Units (%s) not understood. "
                         "Must be 'mrads', 'q', or 'k'" % units)
    return calibration

def optimize_parameters(beam_energy, min_q, max_q):

    cls = ['38','48','60','77','100','130','160','195','245','300','380',
           '450', '600', '770', '900', '1200']

    cals = np.zeros(len(cls))
    q_256 = np.zeros(len(cls))
    for i in range(0, len(cls)):
        cals[i] = get_calibration(beam_energy, cls[i], units='q')
        q_256[i] = cals[i] * 256

    max_cl = cls[np.where(q_256>1.5*max_q)[0][-1]]
    max_alpha = q_to_mrads(10*min_q, beam_energy)/1.5


    print("Merlin Experimental Parameters")
    print("##############################")
    print("Beam energy (keV): \t\t%.1f" % beam_energy)
    print("Maximum Required q (A^-1): \t%.2f" % max_q)
    print("Minimum Required q (A^-1): \t%.2f\n" % min_q)
    print("Recommended Parameters")
    print("Highest CL (mm): \t\t%s" % max_cl)
    print("Highest alpha (mrads): \t\t%.1f\n\n" % max_alpha)

    print("Suggested ADF Mask Locations")
    print("############################\n")
    print("CL (mm)\t\tLocation for q_min (pixels))\t\tLocation for q_max (pixels)")
    print("--------------------------------------------------------------------------------------")
    for i in range(0, len(cals)):
        print("%s\t\t\t%.1f\t\t\t\t\t%.1f" % (cls[i], min_q/cals[i], max_q/cals[i]))
    return