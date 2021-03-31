import os
import hyperspy.api as hs
import numpy as np
import fpd.fpd_processing as fpdp
import fpd.fpd_file as fpdf
from scipy import ndimage
import matplotlib.pylab as plt
from merlintools import color


def get_radial_profile(ds, com_yx):
    radial_mean = [None] * ds.shape[0]*ds.shape[1]
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

    result = np.zeros([ds.shape[0]*ds.shape[1], min_length])
    for i in range(0, len(radial_mean)):
        result[i, :] = radial_mean[i][0:min_length]
    result = result.reshape([ds.shape[0], ds.shape[1], min_length])
    s = hs.signals.Signal1D(result)
    return s


def shift_func(image, scanYind, scanXind, shift_array, sub_pixel=True,
               interpolation=3):
    if sub_pixel:
        syx = shift_array[:, scanYind, scanXind]
    else:
        syx = np.int32(np.round(shift_array[:, scanYind, scanXind], 0))
        interpolation = 0
    new_im = ndimage.shift(image, -syx, order=interpolation)
    return new_im


def align_merlin(h5filename, sub_pixel=True, interpolation=3):
    coms_file = os.path.splitext(h5filename)[0] + "_CoMs.npy"
    shifts_file = os.path.splitext(h5filename)[0] + "_Shifts.npy"
    ali_file = os.path.splitext(h5filename)[0] + "_Aligned.hdf5"

    nt = fpdf.fpd_to_tuple(h5filename, fpd_check=False)
    sum_dif = nt.fpd_sum_dif.data
    ds = nt.fpd_data.data
    h5f = nt.file

    idx_x, idx_y = np.int32(np.array(ds.shape[0:2])/2)
    thresh_val = 0.5 * ds[idx_x, idx_y, :, :].max()
    cyx, cr = fpdp.find_circ_centre(sum_dif, 10, (6, 20, 2), pct=90,
                                    spf=1, plot=False)

    mask = fpdp.synthetic_aperture(shape=ds.shape[-2:], cyx=cyx,
                                   rio=(0, cr*2.5), sigma=0)[0]
    mask = np.ceil(mask)

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
    dps = np.zeros([n_pix, data_4d.shape[2], data_4d.shape[3]])
    max_locs = image.ravel().argsort()[::-1]

    for i in range(0, n_pix):
        row, col = np.unravel_index(max_locs[i], [128, 128])
        dps[i] = data_4d[0][row, col]

    dps = hs.signals.Signal2D(dps)
    return dps
