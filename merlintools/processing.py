# -*- coding: utf-8 -*-
#
# This file is part of MerlinTools

"""
processing module for MerlinTools package.

@author: Andrew Herzing
"""

import os
import hyperspy.api as hs
import numpy as np
import fpd.fpd_processing as fpdp
import fpd.fpd_file as fpdf
from scipy import ndimage
import matplotlib.pylab as plt
from merlintools import color


def get_radial_profile(ds, com_yx):
    """
    Calculate radial profile for a 4D dataset.

    Dataset does not need to be aligned.  Instead, the center-of-mass is provided
    for each diffraction pattern and the profiles are calculated using the CoM as
    as the center.  After all calculations, all profiles are truncated to the length of
    the smallest profile.

    Args
    ----------
    ds : NumPy array
        4D-STEM dataset
    com_yx : tuple
        List of center point for each frame, usually determined by center
        of mass analysis

    Returns
    ----------
    s : HyperSpy Signal1D
        Radial average as a function of beam scan position
    """
    radial_mean = [None] * ds.shape[0] * ds.shape[1]
    idx = 0
    min_length = np.inf
    # r_pix_min = None
    for i in range(0, ds.shape[0]):
        for j in range(0, ds.shape[1]):
            r_pix, radial_mean[idx] = fpdp.radial_profile(ds[i, j, :, :],
                                                          com_yx[:, i, j],
                                                          plot=False, spf=1)
            if radial_mean[idx].shape[0] < min_length:
                min_length = radial_mean[idx].shape[0]
                # r_pix_min = r_pix
            idx += 1

    result = np.zeros([ds.shape[0] * ds.shape[1], min_length])
    for i in range(0, len(radial_mean)):
        result[i, :] = radial_mean[i][0:min_length]
    result = result.reshape([ds.shape[0], ds.shape[1], min_length])
    s = hs.signals.Signal1D(result)
    return s


def shift_func(image, scanYind, scanXind, shift_array, sub_pixel=True,
               interpolation=3):
    """
    Center 4D-STEM data using align_merlin.

    Args
    ----------
    image : NumPy array
        4D-STEM dataset
    scanYind, scanXind : int
        Scan indices for iteration
    shift_array : NumPy array
        Array containing x and y shifts for each pattern
    sub_pixel : bool
        If True, perform sub-pixel alignment. May cause interpolation
        artifacts.
    interpolation : int
        Order of interpolation for sub-pixel alignment

    Returns
    ----------
    new_im : NumPy array
        Shifted version of input image
    """
    if sub_pixel:
        syx = shift_array[:, scanYind, scanXind]
    else:
        syx = np.int32(np.round(shift_array[:, scanYind, scanXind], 0))
        interpolation = 0
    new_im = ndimage.shift(image, -syx, order=interpolation)
    return new_im


def align_merlin(h5filename, sub_pixel=True, interpolation=3,
                 apply_threshold=True, apply_mask=True):
    """
    Align the data using fpd center of mass analysis.

    Results are written to a new HDF5 file.

    Args
    ----------
    h5filename : str
        Filename of FPD data
    sub_pixel : bool
        If True, perform sub-pixel alignment.  May cause interpolation
        artifacts.
    interpolation : int
        Order of interpolation for sub-pixel alignment
    apply_threshold : bool
        If True, threshold data before further processing. Threshold is set to
        half of the maximum value of the central pattern.
    apply_mask : bool
        If True, mask data to region around central beam.


    """
    coms_file = os.path.splitext(h5filename)[0] + "_CoMs.npy"
    shifts_file = os.path.splitext(h5filename)[0] + "_Shifts.npy"
    ali_file = os.path.splitext(h5filename)[0] + "_Aligned.hdf5"

    nt = fpdf.fpd_to_tuple(h5filename, fpd_check=False)
    sum_dif = nt.fpd_sum_dif.data
    ds = nt.fpd_data.data
    h5f = nt.file

    idx_x, idx_y = np.int32(np.array(ds.shape[0:2]) / 2)

    if apply_threshold:
        thresh_val = 0.5 * ds[idx_x, idx_y, :, :].max()
    else:
        thresh_val = None

    if apply_mask:
        cyx, cr = fpdp.find_circ_centre(sum_dif, 10, (6, 20, 2), pct=90,
                                        spf=1, plot=False)

        mask = fpdp.synthetic_aperture(shape=ds.shape[-2:], cyx=cyx,
                                       rio=(0, cr * 2.5), sigma=0)[0]
        mask = np.ceil(mask)
    else:
        mask = None

    com_yx = fpdp.center_of_mass(ds, nr=None, nc=None, aperture=mask,
                                 progress_bar=False, print_stats=False,
                                 parallel=False, thr=thresh_val)
    h5f.close()

    np.save(coms_file, com_yx)
    com_yx[np.where(np.isnan(com_yx[:, :, :]))] = 128.

    shifts_yx = com_yx - 128.
    np.save(shifts_file, shifts_yx)
    fpdf.make_updated_fpd_file(h5filename, ali_file, shift_func,
                               func_kwargs={'shift_array': shifts_yx,
                                            'sub_pixel': sub_pixel,
                                            'interpolation': interpolation},
                               ow=True, progress_bar=False)
    return


def get_segmented_annular_aperture(ds, cyx=(128, 128),
                                   rio=[[0, 20], [30, 60]], plot_result=False,
                                   sigma=0, aaf=3, axis=None, color_list=None):
    """
    Create a segmented annular aperture.

    A circular bright-field aperture plus a segmented annular aperture with four
    quadrants are created using the synthetic aperture function of FPD.

    Args
    ----------
    ds : NumPy array
        4D-STEM dataset
    cyx : tuple
        List of center point for each frame, usually determined by center
        of mass analysis
    rio : list
        Inner and outer radii for the bright-field.  First entry is for the
        bright field aperture, the second applies to the four segmented
        apertures
    plot_result : bool
        If True, plot resulting apertures in color
    sigma : float
        Sigma for Gaussian blur function
    aaf : float
        Anti-aliasing factor to pass to synthetic_aperture function in FPD.
        Use 1 for none.
    axis : MatPlotlib axis
        If provided, plot aperture image in pre-defined axis.
    color_list : list
        Defines colors for plot result

    Returns
    ----------
    aps : NumPy array
        Resulting apertures
    """
    rio = np.array(rio)
    rio = np.vstack((rio, np.zeros([4, 2])))
    rio[2:, :] = rio[1, :]

    aps = fpdp.synthetic_aperture(shape=ds.shape[-2:], cyx=cyx, rio=rio,
                                  sigma=sigma, aaf=aaf)

    aps[2, 128:, :] = 0
    aps[2, :, 128:] = 0

    aps[3, 128:, :] = 0
    aps[3, :, :128] = 0

    aps[4, :128, :] = 0
    aps[4, :, 128:] = 0

    aps[5, :128, :] = 0
    aps[5, :, :128] = 0

    if plot_result:
        if color_list is None:
            color_list = ['magenta', 'red', 'green', 'blue', 'yellow']
        rgb = color.merge_color_channels(aps[[0, 2, 3, 4, 5], :, :],
                                         color_list=color_list)
        if axis:
            axis.imshow(rgb)
        else:
            plt.figure()
            plt.imshow(rgb)
    return aps


def get_max_dps(data_4d, image, n_pix=100):
    """
    Return a HyperSpy signal containing the diffraction patterns from the most intense pixels in an image.

    For example, an ADF image can be provided and the diffraction patterns which
    contributed the most intensity to the ADF aperture will be returned.

    Args
    ----------
    ds : NumPy array
        4D-STEM dataset
    image : NumPy array
        Image to be used for locating diffraction patterns.
    n_pix : int
        The number of patterns to return.

    Returns
    ----------
    dps : HyperSpy Signal2D

    """
    dps = np.zeros([n_pix, data_4d.shape[2], data_4d.shape[3]])
    max_locs = image.ravel().argsort()[::-1]

    for i in range(0, n_pix):
        row, col = np.unravel_index(max_locs[i], [128, 128])
        dps[i] = data_4d[0][row, col]

    dps = hs.signals.Signal2D(dps)
    return dps
