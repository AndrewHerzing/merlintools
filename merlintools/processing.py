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
from scipy import ndimage, optimize
import matplotlib.pylab as plt
from merlintools import color

def radial_profile(ds, center_yx, recip_calib=None, crop=True, spf=1.0):
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
    center_yx : tuple
        List of center point for each frame, usually determined by center
        of mass analysis
    crop : bool
        If True, crop all profiles to a common size.
    recip_calib : float
        Reciprocal space calibration
    spf : float
        Subsample pixel factor passed to fpd radial_profile function.  Default is 1.0.


    Returns
    ----------
    bins : NumPy Array
        Bins for radial average result
    radial : NumPy Array
        Radial average as a function of beam scan position
    """
    def _radial_func(frame, center, r_nm_pp, spf):
        r_pix, rms = fpdp.radial_profile(frame, center, r_nm_pp=r_nm_pp, spf=spf)
        return r_pix, rms

    cyx = np.moveaxis(center_yx, 0, -1)

    res = fpdp.map_image_function(ds, nr=32, nc=32,
                                  func=_radial_func,
                                  mapped_params={'center': cyx},
                                  params={'r_nm_pp': recip_calib,
                                          'spf': spf})

    min_length = np.inf
    for i in range(0, res.shape[0]):
        for j in range(0, res.shape[1]):
            if res[i, j][0].shape[0] < min_length:
                min_length = res[i, j][0].shape[0]

    if crop:
        radial = np.zeros([res.shape[0], res.shape[1], min_length])
        bins = res[0, 0][0][0:min_length]
        for i in range(0, res.shape[0]):
            for j in range(0, res.shape[1]):
                radial[i, j, :] = res[i, j][1][0:min_length]
    else:
        bins = np.array(np.empty([ds.shape[0], ds.shape[1]]), dtype='object')
        radial = np.array(np.empty([ds.shape[0], ds.shape[1]]), dtype='object')
        for i in range(0, res.shape[0]):
            for j in range(0, res.shape[1]):
                bins[i, j] = res[i, j][0]
                radial[i, j] = res[i, j][1]
    return bins, radial

def shift_align(ds, shifts, nr=16, nc=16, sub_pixel=True, interpolation=3):
    """
    Align data given shift array.

    Args
    ----------
    ds : NumPy array
        4D-STEM dataset
    shifts : NumPy array
        Array of shifts of shape (2, scanY, scanX)
    sub_pixel : bool
        If True, use interpolation to perform sub-pixel shifts.
    interpolation : int
        Degree of interpolation

    Returns
    ----------
    ali : NumPy Array
        Aligned version of ds
    """
    def ali_func(image, shift_array, sub_pixel, interpolation):
        if sub_pixel:
            syx = shift_array
        else:
            syx = np.int32(np.round(shift_array, 0))
            interpolation = 0
        new_im = ndimage.shift(image, -syx, order=interpolation)
        return new_im
    
    ali = fpdp.map_image_function(ds, nr, nc, func=ali_func,
                                  mapped_params={'shift_array': np.moveaxis(shifts, 0, -1)},
                                  params={'sub_pixel': sub_pixel,
                                          'interpolation': interpolation})
    ali = ali.T.reshape(ds.shape)
    return ali

def align_merlin(h5filename, sub_pixel=True, interpolation=3,
                 apply_threshold=True, apply_mask=True, output_path=None):
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
    output_path : str
        If None, save to same path as input data.  Otherwise, save to the
        specified path.

    """
    def shift_func(image, scanYind, scanXind, shift_array, sub_pixel=True,
               interpolation=3):
        if sub_pixel:
            syx = shift_array[:, scanYind, scanXind]
        else:
            syx = np.int32(np.round(shift_array[:, scanYind, scanXind], 0))
            interpolation = 0
        new_im = ndimage.shift(image, -syx, order=interpolation)
        return new_im

    if not output_path:
        coms_file = os.path.splitext(h5filename)[0] + "_CoMs.npy"
        shifts_file = os.path.splitext(h5filename)[0] + "_Shifts.npy"
        ali_file = os.path.splitext(h5filename)[0] + "_Aligned.hdf5"
    else:
        if output_path[-1] != '/':
            output_path = output_path + '/'
        rootname = os.path.splitext(os.path.split(h5filename)[1])[0]
        coms_file = output_path + rootname + "_CoMs.npy"
        shifts_file = output_path + rootname + "_Shifts.npy"
        ali_file = output_path + rootname + "_Aligned.hdf5"

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
    data_4d : NumPy array
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
    locs = [None]*n_pix
    for i in range(0, n_pix):
        row, col = np.unravel_index(max_locs[i], data_4d.shape[0:2])
        locs[i] = [row, col]
        dps[i] = data_4d[row, col, :, :]

    dps = hs.signals.Signal2D(dps)
    return dps, locs

def get_virtual_images(data4d, com_yx, apertures, sub_pixel=True, nr=128, nc=128):
    """
    Extract a virtual image using a radial mask with varying center.

    Args
    ----------
    data4d : NumPy array
        4D-STEM dataset
    com_yx : NumPy array
        Center location for each diffraction pattern.
    apertures : NumPy array
        Masks for virtual images with shape [n_masks, DetY, DetX]
    sub_pixel : bool
        If True, shift masks using cubic interpolation. Otherwise, shifts are
        rounded to the nearest integer and no interpolation is used.
    nr, nc : int
        Chunks

    Returns
    ----------
    v_images : NumPy array
        Virtual images

    """

    def f_pixel(image, mask, shift):
        mask_shifted = np.zeros_like(mask)
        mask_shifted = [ndimage.shift(mask[i], shift, order=0) for i in range(0,mask.shape[0])]
        res = (image * mask_shifted)
        return res
    
    def f_subpixel(image, mask, shift):
        mask_shifted = np.zeros_like(mask)
        mask_shifted = [ndimage.shift(mask[i], shift, order=3) for i in range(0,mask.shape[0])]
        res = (image * mask_shifted)
        return res
    
    scanY, scanX, detY, detX = data4d.shape
    center_yx = np.array(data4d.shape[-2:])/2

    n_apts = apertures.shape[0]
    
    com_shifts = np.moveaxis(com_yx, 0, -1) - center_yx

    if not sub_pixel:
        com_shifts = np.int32(np.round(com_shifts))
        v_images = fpdp.map_image_function(data4d, nr=nr, nc=nc,
                                           func=f_pixel,
                                           params={'mask': apertures},
                                           mapped_params={'shift': com_shifts})
    else:
        v_images = fpdp.map_image_function(data4d, nr=nr, nc=nc,
                                           func=f_subpixel,
                                           params={'mask': apertures},
                                           mapped_params={'shift': com_shifts})

    v_images = v_images.reshape([n_apts, detY, detX, scanY, scanX]).sum((1,2))
    return v_images

def get_q_images(data, q_ranges):
    """
    Extract images from various q ranges.

    Args
    ----------
    data : NumPy array
        4D-STEM dataset
    q_vals : list
        List of tuples defining q range for each window

    Returns
    ----------
    ims : NumPy array
        Extracted images
    """
    if type(q_ranges) is not list:
        q_ranges = [q_ranges]
    ims = np.array(np.zeros([len(q_ranges), data['sum_image'].shape[0], data['sum_image'].shape[1]]))
    
    for i in range(0, len(q_ranges)):
        idx = np.where(np.logical_and(data['radial_profile'][0] > q_ranges[i][0], data['radial_profile'][0] < q_ranges[i][1]))[0]
        ims[i] = data['radial_profile'][1][:,:,idx].sum(2)
    return ims

def remove_bckg(data, q_range, scanYX=None, plot_result=False, model='power'):
    """
    Remove background from 4D-STEM dataset using a power law model.

    Args
    ----------
    data : NumPy array
        Radially averaged or integrated 4D-STEM dataset
    q_range : list
        Range for background fit
    scanYX : None or tuple
        Scan position at which to perform fitting.  If None, fitting is performed
        on the sum profile.
    plot_result : bool
        If True, plot the profile and fitting result
    model : string
        Model to use for background fitting.

    Returns
    ----------
    corrected : NumPy array
        Background subtracted radial profile

    Examples
    --------
    >>> import merlintools as merlin
    >>> data = merlin.io.read_h5_results("./merlin_data.hdf5")
    >>> corrected = merlin.processing.remove_bckg(data['radial_profile'], [0.5,1.3])
    """
    def _pl_func(x, A, r):
        return A*x**-r
    
    def _poly_func(x, a4, a3, a2, a1, c):
        return a4*x**4 + a3*x**3 + a2*x**2 + a1*x + c
    
    xaxis = data['radial_profile'][0]
    if scanYX is None:
        yaxis = data['radial_profile'][1].sum((0,1))
    else:
        yaxis = data['radial_profile'][1][scanYX[0], scanYX[1], :]
    fit_range=[np.where(xaxis > q_range[0])[0][0], np.where(xaxis>q_range[1])[0][0]]
    if model.lower() in ['power','pl']:
        res, cov = optimize.curve_fit(_pl_func, xaxis[fit_range[0]:fit_range[1]], yaxis[fit_range[0]:fit_range[1]])
        fit = _pl_func(xaxis[fit_range[0]:], res[0], res[1])
    elif model.lower() in ['poly',]:
        res, cov = optimize.curve_fit(_poly_func, xaxis[fit_range[0]:fit_range[1]], yaxis[fit_range[0]:fit_range[1]])
        fit = _poly_func(xaxis[fit_range[0]:], res[0], res[1], res[2], res[3], res[4])
    
    corrected = yaxis[fit_range[0]:] - fit
    
    if plot_result:
        plt.figure()
        plt.plot(xaxis[fit_range[0]:], yaxis[fit_range[0]:], 'ro', label='Data')
        plt.plot(xaxis[fit_range[0]:], fit, 'blue', label='Fit')
        plt.plot(xaxis[fit_range[0]:], corrected, 'go', label='Corrected')
        plt.legend()
    
    return corrected