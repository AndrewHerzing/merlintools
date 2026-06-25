import numpy as np
import scipy as sp
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
# from scipy.ndimage.measurements import center_of_mass

import matplotlib as mpl
import matplotlib.pyplot as plt

from skimage.feature import canny, peak_local_max
from skimage.transform import hough_circle
from skimage import color
from skimage.draw import circle_perimeter

# from skimage.transform import pyramid_expand
from skimage.filters import threshold_otsu
from skimage.morphology import disk, binary_closing, binary_opening

import h5py
import datetime
import os
import multiprocessing as mp
from functools import partial
import sys
import itertools
import time
import warnings
from numbers import Number
from tqdm import tqdm

from itertools import combinations
from collections import namedtuple
from collections.abc import Iterable
from threadpoolctl import threadpool_limits


# favour sp.fft over np.fft
try:
    from scipy import fft as fft_module
except:
    from numpy import fft as fft_module

from .utils import int_factors, nearest_int_factor


def _check_libs():
    try:
        # check if openblas
        import ctypes
        from ctypes.util import find_library

        # https://stackoverflow.com/questions/29559338/set-max-number-of-threads-at-runtime-on-numpy-openblas

        # np.show_config()
        # this is hard coded so may not be always reliable
        blas_libs = np.__config__.openblas_info["libraries"]
        openblas_lib = None
        if any([x.lower() == "openblas" for x in blas_libs]):
            libpath = find_library("openblas")
            openblas_lib = ctypes.cdll.LoadLibrary(libpath)
        if openblas_lib:
            ob_threads = openblas_lib.openblas_get_num_threads()
            if ob_threads != 1:
                # doesn't seem to work:
                openblas_lib.openblas_set_num_threads(1)
                print(
                    "-------------------------------------------------------------------"
                )
                print(
                    "FPD: It looks like numpy is using OpenBLAS with %d threads."
                    % (ob_threads)
                )
                print("FPD: Performance might be improved by running with 1 thread.")
                print(
                    "FPD: Try setting env variable 'OMP_NUM_THREADS=1' before importing."
                )
                print(
                    "-------------------------------------------------------------------\n"
                )
    except:
        multi_thread = True
        try:
            nthreads = int(os.environ["OMP_NUM_THREADS"])
            if nthreads == 1:
                multi_thread = False
        except KeyError:
            # multi_thread = True
            pass

        if multi_thread:
            print("-------------------------------------------------------------------")
            print("FPD: It looks like numpy is using multiple threads.")
            print("FPD: Performance might be improved by running with 1 thread.")
            print("FPD: Try setting env variable 'OMP_NUM_THREADS=1' before importing.")
            print(
                "-------------------------------------------------------------------\n"
            )


# not needed due to threadpoolctl
# _check_libs()


def cpu_thread_lib_check(n=2000):
    """
    Check is numpy is using multiple threads by running known multithreaded code.

    Parameters
    ----------
    n : integer
        Size of one side of nxn test data array.

    Returns
    -------
    multi_thread : bool or None
        True if multithreading is detected. None if undetermined.

    """

    import psutil
    from threading import Thread, Lock

    a = np.ones((n,) * 2)
    lock = Lock()

    def mt_func():
        lock.acquire()
        _ = np.dot(a, a)
        lock.release()

    t = Thread(target=mt_func)
    t.start()

    cpu_use = []
    while lock.locked():
        cpu_use.append(psutil.cpu_percent())
        time.sleep(0.005)

    multi_thread = True
    try:
        if max(cpu_use) > 90:
            print("-------------------------------------------------------------------")
            print("FPD: It looks like numpy is using multiple threads.")
            print("FPD: Performance might be improved by running with 1 thread.")
            print("FPD: Try setting env variable 'OMP_NUM_THREADS=1' before importing.")
            print(
                "-------------------------------------------------------------------\n"
            )
        else:
            multi_thread = False
    except:
        multi_thread = None

    return multi_thread


# --------------------------------------------------
def rebinA(a, *args, **kwargs):
    """
    Return array 'a' rebinned to shape provided in args.

    Parameters
    ----------
    a : array-like
        Array to be rebinned.
    args : expanded tuple
        New shape.
    dtype : numpy dtype or a string representation thereof.
        For integer input, if specified as a keyword argument, this is used
        instead of the data dtype when determining the returned dtype.
    bitdepth : int
        For integer input, if specified as a keyword argument, the maximum data
        value is calculated using this bitdepth.

    Returns
    -------
    b : ndarray
        The rebinned array. If of integer type, the returned data dtype is
        appropriate to suit the maximum possible value. Otherwise, the output
        dtype is determined by the behaviour of np.sum. The dtype can be further
        modified by specifying `dtype` and / or `bitdepth`.

    Notes
    -----
    Based on http://scipy-cookbook.readthedocs.io/items/Rebinning.html



    Examples
    --------
    >>> from merlintools.vendor.fpd import fpd_processing as fpdp
    >>> import numpy as np

    >>> a = np.random.rand(6, 4, 2).astype('u2')
    >>> print(a.shape)
    (6, 4, 2)

    >>> b = fpdp.rebinA(a, *[i//2 for i in a.shape])
    >>> print(b.shape)
    (3, 2, 1)
    >>> print(b.dtype)
    uint32

    >>> b = fpdp.rebinA(a, *[i//2 for i in a.shape], bitdepth=12)
    >>> print(b.dtype)
    uint16

    """

    dtype = None

    input_dtype = kwargs.pop("dtype", None)
    if input_dtype is not None:
        input_dtype = np.dtype(input_dtype)
    else:
        input_dtype = a.dtype

    in_k = input_dtype.kind
    input_is_int = in_k in "ui"

    if input_is_int:
        # multiple
        m = np.prod([o // n for (o, n) in zip(a.shape, args)])

        signed = in_k == "i"
        prefix = ""
        if not signed:
            prefix += "u"

        bitdepth = kwargs.pop("bitdepth", None)
        if bitdepth is None:
            in_pix_max = np.iinfo(input_dtype).max
        else:
            in_pix_max = 2 ** (bitdepth - 1 * signed) - 1
        out_pix_max = m * in_pix_max

        maxes = np.array(
            [
                np.iinfo(prefix + "int%d" % (d)).max
                for d in 8 * 2 ** np.array([0, 1, 2, 3])
            ]
        )
        inds = np.where(maxes > out_pix_max)[0]
        if len(inds) != 0:
            dtype = "'" + prefix + "int%d" % (8 * 2 ** inds[0]) + "'"

    shape = a.shape
    lenShape = len(shape)
    factor = (np.asarray(shape) / np.asarray(args)).astype(int)
    evList = (
        ["a.reshape("]
        + ["args[%d],factor[%d]," % (i, i) for i in range(lenShape)]
        + [")"]
        + [".sum(%d, dtype=%s)" % (i + 1, dtype) for i in range(lenShape)]
    )
    # print(''.join(evList))
    return eval("".join(evList))


# --------------------------------------------------
def _block_indices(dshape, nrnc):
    """
    Generate list of indices of blocks of data of shape dshape of size nrnc.

    Parameters
    ----------
    dshape : tuple
        Shape of data array.
    nrnc : tuple, None
        Chunk length in each axis.
        If any entry is None, indices will be for all data.

    Returns
    -------
    List of lists by which chunks of array may be indixed.

    Examples
    --------
    >>> from merlintools.vendor.fpd import fpd_processing as fpdp
    >>> fpdp._block_indices(dshape=(5,8), nrnc=(None,)*2)
    [[(0, 5)], [(0, 8)]]

    >>> r_if, c_if = fpdp._block_indices(dshape=(5,8), nrnc=(3,)*2)
    >>> print(r_if, c_if)
    [(0, 3), (3, 5)] [(0, 3), (3, 6), (6, 8)]

    >>> for i,(ri, rf) in enumerate(r_if):
    ...     for j,(ci, cf) in enumerate(c_if):
    ...         print('\tScan [row,col] chunk [%d, %d] of [%d, %d] - %05.1f%%' %(i+1, j+1, len(r_if), len(c_if), (j+1+i*len(c_if))*100.0/(len(c_if)*len(r_if))), end='\r')
    >>>
    >>>     # data_out[ri:rf,ci:cf] = f(data[ri:rf,ci:cf])

    """

    assert len(dshape) >= len(nrnc)

    inds = [list(range(x)) for x in dshape[: len(nrnc)]]
    ns = [n if n is not None else dshape[i] for (i, n) in enumerate(nrnc)]
    rc_ifs = [
        list(zip([0] + inds[i][n::n], inds[i][n::n] + [inds[i][-1] + 1]))
        for i, n in enumerate(ns)
    ]
    return rc_ifs


# --------------------------------------------------
def sum_im(data, nr, nc, mask=None, nrnc_are_chunks=False, progress_bar=True):
    """
    Return a real-space sum image from data.

    Parameters
    ----------
    data : array_like
        Multidimensional fpd data of shape (scanY, scanX, ..., detY, detX).
    nr : integer or None
        Number of rows to process at once (see Notes).
    nc : integer or None
        Number of columns to process at once (see Notes).
    mask : 2-D array or None
        Mask is applied to data before taking sum.
        Shape should be that of the detector.
    nrnc_are_chunks : bool
        If True, `nr` and `nc` are interpreted as the number of chunks to
        process at once. If `data` is not chunked, `nr` and `nc` are used
        directly.
    progress_bar : bool
        If True, progress bars are printed.

    Returns
    -------
    Array of shape (scanY, scanX, ...).

    Notes
    -----
    If `nr` or `nc` are None, the entire dimension is processed at once.
    For chunked data, setting `nrnc_are_chunks` to True, and `nr` and `nc`
    to a suitable values can improve performance.


    """

    if nrnc_are_chunks:
        nr, nc = _condition_nrnc_if_chunked(data, nr, nc, True)

    nondet = data.shape[:-2]
    nonscan = data.shape[2:]
    scanY, scanX = data.shape[:2]
    detY, detX = data.shape[-2:]

    r_if, c_if = _block_indices((scanY, scanX), (nr, nc))
    if mask is not None:
        for i in range(len(nondet)):
            mask = np.expand_dims(mask, 0)
            # == mask = mask[None,None,None,...]

    sum_im = np.empty(nondet)
    print("Calculating real-space sum images.")
    total_ims = np.prod(nondet)
    with tqdm(
        total=total_ims,
        mininterval=0,
        leave=True,
        unit="images",
        disable=(not progress_bar),
    ) as pbar:
        for i, (ri, rf) in enumerate(r_if):
            for j, (ci, cf) in enumerate(c_if):
                if mask is None:
                    d = data[ri:rf, ci:cf, ...]
                else:
                    d = data[ri:rf, ci:cf, ...] * mask
                sum_im[ri:rf, ci:cf, ...] = d.sum((-2, -1))
                pbar.update(np.prod(d.shape[:-2]))
    return sum_im


def _shift_func(d_dyx, dyx_mode):
    from fpd.synthetic_data import shift_im

    di, dyxi = d_dyx
    try:
        shifts_are_ints = np.all([dyxij.is_integer() for dyxij in dyxi])
    except AttributeError:
        shifts_are_ints = False
    if shifts_are_ints:
        dyxi = dyxi.astype(int)

    method_i = dyx_mode
    if shifts_are_ints:
        # override dyx_mode for efficiency
        method_i = "pixel"
    d = shift_im(di, dyxi, method=method_i)
    return d


# --------------------------------------------------
def _shift_images(
    d, dyx, dyx_mode, in_place=True, parallel=True, ncores=None, parallel_mode="thread"
):
    """
    Shift multiple images.

    Parameters
    ----------
    d : ndarray
        Image data of shape (..., Y, X).
    dyx : ndarray
        (y, x) vector of shifts to be applied to the data, of
        shape (2, ...).
    dyx_mode : str
        If 'pixel', `dyx` is converted to integer type for pixel resolution.
        If 'linear', `dyx` is used with sub-pixel resolution in linear interpolation.
        If 'fourier', `dyx` is used with sub-pixel resolution in Fourier shifting.
        The 'pixel' mode is very fast. The 'linear' mode is next fastest, while
        the 'fourier' method may give rise to ringing in some cases.
    in_place : bool
        If True, the input data array is modified. Otherwise, a copy is made.
    parallel : bool
        If True, the calculations are processed in parallel.
    ncores : None or int
        Number of cores to use for parallel execution. If None, all cores
        are used.
    parallel_mode : str
        The mode to use for parallel processing.
        If 'thread' use multithreading.
        If 'process' use multiprocessing.

    Returns
    -------
    d : ndarray
        The shifted images.

    TODO
    ----
    - incorporate w/ utils versions?
    - optimise for chunk?

    """

    from fpd.synthetic_data import shift_im
    from fpd.utils import seq_image_array, unseq_image_array

    dyx, dyx_mode = _condition_dyx(dyx, dyx_mode)

    if not in_place:
        d = d.copy()

    # flatten d to (n, Y, X)
    axes = (-2, -1)
    d, d_unflat_shape = seq_image_array(d, axes)

    # flatten dyx to (n, 2)
    dyx = np.asarray(dyx)
    dyx = np.moveaxis(dyx, 0, -1)
    ns = (np.prod(dyx.shape[:-1]),) + dyx.shape[-1:]
    dyx = np.reshape(dyx, ns)

    shift_func_partial = partial(_shift_func, dyx_mode=dyx_mode)
    if dyx_mode == "pixel":
        # slightly faster in single thread
        parallel = False
    rtn = _run(
        shift_func_partial,
        zip(d, dyx),
        parallel=parallel,
        parallel_mode=parallel_mode,
        ncores=ncores,
    )
    rtn = np.asarray(rtn)
    if in_place:
        d[:] = rtn
    else:
        d = rtn

    # unflatten d
    return unseq_image_array(d, axes, d_unflat_shape)


def _condition_dyx(dyx, dyx_mode):
    # needed here because of of different parameter names here and in shift_im
    if dyx is not None:
        dyx = np.asarray(dyx)
        dyx_modes = ["pixel", "linear", "fourier"]
        if dyx_mode not in dyx_modes:
            raise Exception("'dyx_mode' (%s) must be one of: " % (dyx_mode), dyx_modes)

        if dyx_mode == "pixel":
            dyx = dyx.round(0).astype(int)
        else:
            # 'linear' unless dyx is of integer dtype
            if np.issubdtype(type(dyx), np.integer):
                dyx_mode == "pixel"
    return dyx, dyx_mode


# --------------------------------------------------
def sum_dif(
    data,
    nr,
    nc,
    mask=None,
    dyx=None,
    dyx_mode="linear",
    fill_value=None,
    nrnc_are_chunks=False,
    progress_bar=True,
):
    """
    Return a summed diffraction image from data.

    Parameters
    ----------
    data : array_like
        Multidimensional fpd data of shape (scanY, scanX, ..., detY, detX).
    nr : integer or None
        Number of rows to process at once (see Notes).
    nc : integer or None
        Number of columns to process at once (see Notes).
    mask : array-like or None
        Mask applied to data before taking sum.
        Shape should be that of the scan.
    dyx : None or ndarray
        If not None, an ndarray of shape (2, scanY, scanX) with (y, x)
        vector of shifts to be applied to the data. See Notes.
    dyx_mode : str
        If 'pixel', `dyx` is converted to integer type for pixel resolution.
        If 'linear', `dyx` is used with sub-pixel resolution in linear interpolation.
        If 'fourier', `dyx` is used with sub-pixel resolution in Fourier shifting.
        The 'pixel' mode is very fast. The 'linear' mode is next fastest, while
        the 'fourier' method may give rise to ringing in some cases.
    fill_value : None or scalar
        Value replacing missing values. If None, the fill value is nan.
    nrnc_are_chunks : bool
        If True, `nr` and `nc` are interpreted as the number of chunks to
        process at once. If `data` is not chunked, `nr` and `nc` are used
        directly.
    progress_bar : bool
        If True, progress bars are printed.

    Returns
    -------
    Array of shape (..., detY, detX).

    Notes
    -----
    The data shift `dyx` may be determined by a number of methods, including:
    - fpd.fpd_processing.center_of_mass
    - fpd.fpd_processing.phase_correlation
    while remembering to negate and centre the returned positions. For example,
    for positions comyx, values for dyx may be obtained from:

    >>> import numpy as np
    >>> shifts = -(comyx - np.percentile(comyx, 50, (-2, -1))[..., None, None])

    If the data is to be aligned to a particular scan point, then it should be
    subtracted rather than the percentile in the example above.

    If `nr` or `nc` are None, the entire dimension is processed at once.
    For chunked data, setting `nrnc_are_chunks` to True, and `nr` and `nc`
    to a suitable values can improve performance.

    """
    from fpd.synthetic_data import _fill_shifted_image

    if fill_value == None:
        fill_value = np.nan

    if nrnc_are_chunks:
        nr, nc = _condition_nrnc_if_chunked(data, nr, nc, True)

    nondet = data.shape[:-2]
    nonscan = data.shape[2:]
    scanY, scanX = data.shape[:2]
    detY, detX = data.shape[-2:]

    r_if, c_if = _block_indices((scanY, scanX), (nr, nc))

    if mask is not None:
        for i in range(len(nonscan)):
            mask = np.expand_dims(mask, -1)
            # == mask = mask[..., None,None,None]

    sum_dif = np.zeros(nonscan)
    print("Calculating diffraction sum images.")
    total_ims = np.prod(nondet)
    with tqdm(
        total=total_ims,
        mininterval=0,
        leave=True,
        unit="images",
        disable=(not progress_bar),
    ) as pbar:
        for i, (ri, rf) in enumerate(r_if):
            for j, (ci, cf) in enumerate(c_if):
                d = data[ri:rf, ci:cf, ...]
                d = np.ascontiguousarray(d)
                if dyx is not None:
                    d = _shift_images(d, dyx, dyx_mode, in_place=False)
                if mask is not None:
                    d = d * mask[ri:rf, ci:cf, ...]
                sum_dif += d.sum((0, 1))
                pbar.update(np.prod(d.shape[:-2]))

    # set bad pixels from shifts
    _fill_shifted_image(sum_dif, dyx, fill_value)

    return sum_dif


# --------------------------------------------------
def sum_ax(data, axis=0, n=32, dtype=None, progress_bar=True):
    """
    Sum an nd-dataset along an axis.

    Parameters
    ----------
    data : array_like
        Multidimensional array-like object.
    axis : int
        Axis along which to sum.
    n : integer, None or a sequence thereof
        Number of pixels along each axis to process at once. If an integer or
        None, the value is used for all axes. If None, the entire axis is read
        in one go.
    dtype : dtype or None
        dtype of the returned array. If None, the dtype will be automatically
        determined by the maximum values of the input dtype and the sum axis
        size.
    progress_bar : bool
        If True, a progress bar is displayed.

    Returns
    -------
    rtn : ndarray
        `data` array summed across axis `axis`.

    """

    if not isinstance(n, Iterable):
        n = (n,) * len(data.shape)
    else:
        assert len(n) == len(data.shape)

    rtn_dtype = None
    if dtype is None:
        int_dtype = np.issubdtype(data.dtype, np.integer)
        if int_dtype:
            max_pix_val = np.iinfo(data.dtype).max
            min_pix_mav = np.iinfo(data.dtype).min
            if min_pix_mav != 0:
                # could be smarter here and use ints
                rtn_dtype = np.float
            else:
                max_new_val = max_pix_val * data.shape[axis]
                bits = int(np.ceil(np.log2(max_new_val + 1)))
                bits_aligned = int(2 ** (np.ceil(bits / 8 / 2)) * 8)
                if bits_aligned > 64:
                    rtn_dtype = np.float
                else:
                    rtn_dtype = "uint%d" % (bits_aligned)
        else:
            rtn_dtype = np.float
    else:
        rtn_dtype = dtype

    new_shape = list(data.shape)
    del new_shape[axis]
    rtn = np.zeros(new_shape, dtype=rtn_dtype)

    nd_ifs = _block_indices(data.shape, n)
    i_axs = [np.array(t)[:, 0] for t in nd_ifs]
    f_axs = [np.array(t)[:, 1] for t in nd_ifs]

    ii = np.array(np.meshgrid(*tuple(i_axs)))
    ff = np.array(np.meshgrid(*tuple(f_axs)))

    iif = np.moveaxis(ii, 0, -1).reshape((-1, len(ii)))
    fff = np.moveaxis(ff, 0, -1).reshape((-1, len(ff)))

    if progress_bar:
        print("Calculating sum along axis %d." % (axis))

    total_chunks = np.prod(iif.shape[0])
    with tqdm(
        total=total_chunks,
        mininterval=0,
        leave=True,
        unit="chunks",
        disable=(not progress_bar),
    ) as pbar:
        for i, (iv, fv) in enumerate(zip(iif, fff)):
            s = [slice(*t) for t in zip(iv, fv)]
            d = data[tuple(s)]
            _ = s.pop(axis)
            rtn[tuple(s)] += d.sum(axis)
            pbar.update(1)
    print("\n")
    return rtn


# --------------------------------------------------
def synthetic_aperture(
    shape, cyx, rio, sigma=1, dt=np.float, aaf=3, ds_method="rebin", norm=True
):
    """
    DEPRECATED in v0.2.3. Use fpd.fpd_processing.virtual_apertures instead.

    Create circular synthetic apertures. Sub-pixel accurate with aaf>1.

    Parameters
    ----------
    shape : length 2 iterable
        Image data shape (y,x).
    cyx : length 2 iterable
        Centre y, x pixel cooridinates
    rio : 2d array or length n itterable
        Inner and outer radii [ri,ro) in a number of forms.
        If a length n itterable and not 2d array, n-1 apertures are returned.
        If a 2d array of shape nx2, rio are taken from rows.
    sigma : scalar
        Stdev of Gaussian filter applied to aperture.
    dt : datatype
        Numpy datatype of returned array. If integer type, data is scaled.
    aaf : integer
        Anti-aliasing factor. Use 1 for none.
    ds_method : str
        String controlling the downsampling method. Methods are:
        'rebin'  for rebinning the data.
        'interp' for interpolation.
    norm : bool
        Controls normalisation of actual to ideal area.
        For apertures extending beyond the image border, the value is
        increase to give the same 'volume'.

    Returns
    -------
    Array of shape (n_ap, y, x).

    Notes
    -----
    Some operations may be more efficient if dt is of the same type as
    the data to which it will be applied.

    Examples
    --------
    >>> from merlintools.vendor.fpd import fpd_processing as fpdp
    >>> import matplotlib.pyplot as plt
    >>> plt.ion()

    >>> aps = fpdp.synthetic_aperture((256,)*2, (128,)*2, np.linspace(32, 192, 10))
    >>> _ = plt.matshow(aps[0])

    """

    with warnings.catch_warnings():
        warnings.simplefilter("once")
        msg = "This function is deprecated and marked for removal. Use the `virtual_apertures` function instead, which squeezes the output (removes singular axes)."
        warnings.warn(msg, DeprecationWarning)

    assert type(aaf) == int

    ds_methods = ["rebin", "interp"]
    ds_method = ds_method.lower()
    if ds_method not in ds_methods:
        erm = "'ds_method' must be one of: " + ", ".join(ds_methods)
        raise NotImplementedError(erm)
    im_shape = shape

    if type(rio) == np.ndarray and rio.ndim == 2:
        n = rio.shape[0]
    else:
        n = len(rio) - 1
        rio = list(zip(rio[:-1], rio[1:]))

    m = np.ones((n,) + shape, dtype=dt)

    # prepare boolean edge selection
    yi, xi = np.indices(shape)
    ri = np.hypot(xi - cyx[1], yi - cyx[0])
    yb = np.logical_or(yi == 0, yi == shape[0] - 1)
    xb = np.logical_or(xi == 0, xi == shape[1] - 1)
    bm = np.logical_or(xb, yb)
    ri_edge = ri[bm]
    ri_min = ri_edge.min()

    cy, cx = [t * aaf for t in cyx]
    shape = tuple([t * aaf for t in shape])
    y, x = np.indices(shape)
    sigma *= aaf

    for i, rio in enumerate(rio):
        ri, ro = [t * aaf for t in rio]
        r = np.hypot(x - cx, y - cy)
        mi = np.logical_and(r >= ri, r < ro)
        mi = gaussian_filter(mi.astype(np.float), sigma, order=0, mode="reflect")

        if np.issubdtype(dt, float):
            mi = mi.astype(dt)
        elif np.issubdtype(dt, "uint"):
            mi = (mi / mi.max() * np.iinfo(dt).max).astype(dt)
        else:
            print("WARNING: dtype '%s' not supported!" % (dt))
            mi = np.ones(shape) * np.nan
        if aaf != 1:
            if ds_method == "rebin":
                mi = rebinA(mi, *im_shape) / float(aaf**2)
            elif ds_method == "interp":
                mi = sp.ndimage.interpolation.zoom(
                    mi,
                    1.0 / aaf,
                    output=None,
                    order=1,
                    mode="constant",
                    cval=0.0,
                    prefilter=True,
                )
        # clip any values outside range coming from interpolation
        mi = mi.clip(0, 1)
        if norm:
            mi *= (np.pi * (ro**2 - ri**2) / aaf**2) / mi.sum()  # normalisation
        elif rio[1] > ri_min:
            # warnings.simplefilter('always', UserWarning)
            # warnings.warn(('Apperture may extend beyond image.'
            #               +' Consider setting norm to False.')
            #               , UserWarning)
            # warnings.filters.pop(0)
            print(
                "WARNING: Aperture extends beyond image (max r = %0.1f). Consider setting norm to True. 'rio':"
                % (ri_min),
                rio,
            )
        m[i, :, :] = mi
    return m


# --------------------------------------------------
def synthetic_images(
    data, nr, nc, apertures, rebin=None, nrnc_are_chunks=False, progress_bar=True
):
    """
    DEPRECATED in v0.2.3. Use fpd.fpd_processing.virtual_images instead.

    Make synthetic images from data and aperture.

    Parameters
    ----------
    data : array_like
        Multidimensional fpd data of shape (scanY, scanX, ..., detY, detX).
    nr : integer or None
        Number of rows to process at once (see Notes).
    nc : integer or None
        Number of columns to process at once (see Notes).
    apertures : array-like
        Mask applied to data before taking sum.
        Shape should 3-D (apN, detY, detX).
    rebin : integer or None
        Rebinning factor for detector dimensions. None or 1 for none.
        If the value is incompatible with the cropped array shape, the
        nearest compatible value will be used instead.
    nrnc_are_chunks : bool
        If True, `nr` and `nc` are interpreted as the number of chunks to
        process at once. If `data` is not chunked, `nr` and `nc` are used
        directly.
    progress_bar : bool
        If True, progress bars are printed.

    Returns
    -------
    Array of shape (apN, scanY, scanX, ...).

    Notes
    -----
    If `nr` or `nc` are None, the entire dimension is processed at once.
    For chunked data, setting `nrnc_are_chunks` to True, and `nr` and `nc`
    to a suitable values can improve performance.

    See Also
    --------
    fpd.fpd_processing.synthetic_aperture

    """

    with warnings.catch_warnings():
        warnings.simplefilter("once")
        msg = "This function is deprecated and marked for removal. Use the `virtual_images` function instead, which is ~2.3x faster and has more capabilities."
        warnings.warn(msg, DeprecationWarning)

    if nrnc_are_chunks:
        nr, nc = _condition_nrnc_if_chunked(data, nr, nc, True)

    apertures = np.rollaxis(apertures, 0, 3)
    apN = apertures.shape[-1]
    # now Y, X, apN

    nondet = data.shape[:-2]
    nonscan = data.shape[2:]
    scanY, scanX = data.shape[:2]
    detY, detX = data.shape[-2:]

    r_if, c_if = _block_indices((scanY, scanX), (nr, nc))

    # determine limits to index array for efficiency
    aps = apertures.sum(-1)
    rii, rif = np.where(aps.sum(axis=1) != 0)[0][[0, -1]]
    cii, cif = np.where(aps.sum(axis=0) != 0)[0][[0, -1]]

    # rebinning
    rebinning = rebin is not None and rebin != 1
    if rebinning:
        # min dim
        min_h = rif - rii + 1
        min_w = cif - cii + 1
        # spare pixels
        sph = detY - min_h
        spw = detX - min_w
        # on left and top: cii and rii

        # possible rebins
        fy, fsy = nearest_int_factor(min_h, rebin)
        fx, fsx = nearest_int_factor(min_w, rebin)

        if fy != rebin:
            h_from_rebin = int(np.ceil(min_h / float(rebin)) * rebin)
            if h_from_rebin <= detY:
                # expand crop
                py = h_from_rebin - min_h
                if py <= rii:
                    rii -= py
                else:
                    rif += py - rii
                    rii -= py
                fy = rebin
            else:
                # leave crop
                pass
        if fx != rebin:
            w_from_rebin = int(np.ceil(min_w / float(rebin)) * rebin)
            if w_from_rebin <= detX:
                # expand crop
                px = w_from_rebin - min_w
                if px <= cii:
                    cii -= px
                else:
                    cif += px - cii
                    cii -= px
                fx = rebin
            else:
                # leave crop
                pass
    cropped_im_shape = (rif + 1 - rii, cif + 1 - cii)
    print("Image data cropped to:", cropped_im_shape)

    if rebinning:
        rebina = np.array([fy, fx])
        if (rebina != rebin).any():
            print(
                "Requested rebin (%d) changed to nearest value: (%d, %d)."
                % (rebin, fy, fx)
            )
            print("Possible values are:", (fsy, fsx))
        rebinned_im_shape = tuple(
            [x // rebinf for (x, rebinf) in zip(cropped_im_shape, rebina)]
        )

    apertures = apertures[rii : rif + 1, cii : cif + 1]
    if rebinning:
        ns = (
            tuple([int(x / rebinf) for (x, rebinf) in zip(apertures.shape[:2], rebina)])
            + apertures.shape[2:]
        )
        apertures = rebinA(apertures, *ns)

    for i in range(len(nondet)):
        apertures = np.expand_dims(apertures, 0)
        # == apertures = apertures[None,None,None,...]

    sim = np.empty(nondet + (apN,))
    if progress_bar:
        print("Calculating synthetic aperture images.")
    total_ims = np.prod(nondet)
    with tqdm(
        total=total_ims,
        mininterval=0,
        leave=True,
        unit="images",
        disable=(not progress_bar),
    ) as pbar:
        for i, (ri, rf) in enumerate(r_if):
            for j, (ci, cf) in enumerate(c_if):
                d = data[ri:rf, ci:cf, ..., rii : rif + 1, cii : cif + 1]
                d = np.ascontiguousarray(d)
                if rebinning:
                    ns = d.shape[:-2] + tuple(
                        [int(x / rebinf) for (x, rebinf) in zip(d.shape[-2:], rebina)]
                    )
                    d = rebinA(d, *ns)
                d = d[..., None]
                sim[ri:rf, ci:cf] = (d * apertures).sum((-3, -2))
                pbar.update(np.prod(d.shape[:-3]))
    if progress_bar:
        print("\n")
    return np.rollaxis(sim, -1, 0)


# --------------------------------------------------
def virtual_apertures(
    shape, cyx, rio, sigma=1, dt=np.float, aaf=3, ds_method="rebin", norm=True
):
    """
    Create circular virtual apertures. Sub-pixel accurate with aaf>1.

    Parameters
    ----------
    shape : length 2 iterable
        Image data shape (y,x).
    cyx : length 2 iterable
        Centre y, x pixel cooridinates
    rio : 2d array or length n itterable
        Inner and outer radii [ri,ro) in a number of forms.
        If a length n itterable and not 2d array, n-1 apertures are returned.
        If a 2d array of shape nx2, rio are taken from rows.
    sigma : scalar
        Stdev of Gaussian filter applied to aperture.
    dt : datatype
        Numpy datatype of returned array. If integer type, data is scaled.
    aaf : integer
        Anti-aliasing factor. Use 1 for none.
    ds_method : str
        String controlling the downsampling method. Methods are:
        'rebin'  for rebinning the data.
        'interp' for interpolation.
    norm : bool
        Controls normalisation of actual to ideal area.
        For apertures extending beyond the image border, the value is
        increase to give the same 'volume'.

    Returns
    -------
    Array of shape ([n_ap,] y, x), with n_ap dimension removed if singular.

    Notes
    -----
    Some operations may be more efficient if dt is of the same type as
    the data to which it will be applied.

    Examples
    --------
    >>> from merlintools.vendor.fpd import fpd_processing as fpdp
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> plt.ion()

    # multiple apertures
    >>> aps = fpdp.virtual_apertures((256,)*2, (128,)*2, np.linspace(32, 192, 9))
    >>> _ = plt.matshow(aps[0])

    # one aperture
    >>> aps = fpdp.virtual_apertures((256,)*2, (128,)*2, np.linspace(32, 52, 2))
    >>> _ = plt.matshow(aps)

    See Also
    --------
    fpd.fpd_processing.virtual_images

    """

    assert type(aaf) == int

    ds_methods = ["rebin", "interp"]
    ds_method = ds_method.lower()
    if ds_method not in ds_methods:
        erm = "'ds_method' must be one of: " + ", ".join(ds_methods)
        raise NotImplementedError(erm)
    im_shape = shape

    if type(rio) == np.ndarray and rio.ndim == 2:
        n = rio.shape[0]
    else:
        n = len(rio) - 1
        rio = list(zip(rio[:-1], rio[1:]))

    m = np.ones((n,) + shape, dtype=dt)

    # prepare boolean edge selection
    yi, xi = np.indices(shape)
    ri = np.hypot(xi - cyx[1], yi - cyx[0])
    yb = np.logical_or(yi == 0, yi == shape[0] - 1)
    xb = np.logical_or(xi == 0, xi == shape[1] - 1)
    bm = np.logical_or(xb, yb)
    ri_edge = ri[bm]
    ri_min = ri_edge.min()

    cy, cx = [t * aaf for t in cyx]
    shape = tuple([t * aaf for t in shape])
    y, x = np.indices(shape)
    sigma *= aaf

    for i, rio in enumerate(rio):
        ri, ro = [t * aaf for t in rio]
        r = np.hypot(x - cx, y - cy)
        mi = np.logical_and(r >= ri, r < ro)
        mi = gaussian_filter(mi.astype(np.float), sigma, order=0, mode="reflect")

        if np.issubdtype(dt, float):
            mi = mi.astype(dt)
        elif np.issubdtype(dt, "uint"):
            mi = (mi / mi.max() * np.iinfo(dt).max).astype(dt)
        else:
            print("WARNING: dtype '%s' not supported!" % (dt))
            mi = np.ones(shape) * np.nan
        if aaf != 1:
            if ds_method == "rebin":
                mi = rebinA(mi, *im_shape) / float(aaf**2)
            elif ds_method == "interp":
                mi = sp.ndimage.interpolation.zoom(
                    mi,
                    1.0 / aaf,
                    output=None,
                    order=1,
                    mode="constant",
                    cval=0.0,
                    prefilter=True,
                )
        # clip any values outside range coming from interpolation
        mi = mi.clip(0, 1)
        if norm:
            mi *= (np.pi * (ro**2 - ri**2) / aaf**2) / mi.sum()  # normalisation
        elif rio[1] > ri_min:
            # warnings.simplefilter('always', UserWarning)
            # warnings.warn(('Apperture may extend beyond image.'
            #               +' Consider setting norm to False.')
            #               , UserWarning)
            # warnings.filters.pop(0)
            print(
                "WARNING: Aperture extends beyond image (max r = %0.1f). Consider setting norm to True. 'rio':"
                % (ri_min),
                rio,
            )
        m[i, :, :] = mi
    return np.squeeze(m)


# --------------------------------------------------
def _rebin_crop_parse(rebin, rii, rif, cii, cif, detY, detX):
    """
    Basic crop rebin parser for virtual_images.

    """

    # min dim
    min_h = rif - rii + 1
    min_w = cif - cii + 1
    # spare pixels
    sph = detY - min_h
    spw = detX - min_w
    # on left and top: cii and rii

    # possible rebins
    fy, fsy = nearest_int_factor(min_h, rebin)
    fx, fsx = nearest_int_factor(min_w, rebin)

    if fy != rebin:
        h_from_rebin = int(np.ceil(min_h / float(rebin)) * rebin)
        if h_from_rebin <= detY:
            # expand crop
            py = h_from_rebin - min_h
            if py <= rii:
                rii -= py
            else:
                rif += py - rii
                rii -= py
            fy = rebin
        else:
            # leave crop
            pass
    if fx != rebin:
        w_from_rebin = int(np.ceil(min_w / float(rebin)) * rebin)
        if w_from_rebin <= detX:
            # expand crop
            px = w_from_rebin - min_w
            if px <= cii:
                cii -= px
            else:
                cif += px - cii
                cii -= px
            fx = rebin
        else:
            # leave crop
            pass
    return (rii, rif, cii, cif), (fy, fx), (fsy, fsx)


# --------------------------------------------------
def _shift_image_chunk(
    data, dyx, dyx_mode, fill_value, ri, rf, ci, cf, rii, rif, cii, cif, debug=False
):

    if (dyx_mode.lower() == "fourier") and (fill_value == np.nan):
        raise ValueError("'fill_value' cannot be nan with 'dyx_mode' set to 'fourier'")

    detY, detX = data.shape[-2:]
    dyx_chnk = dyx[:, ri:rf, ci:cf].copy()

    # amount to extend data read for chunk (in pixels)
    dyx_min = np.floor(dyx_chnk.min((-2, -1))).astype(int)
    dyx_max = np.ceil(dyx_chnk.max((-2, -1))).astype(int)

    # these get added to rii... for read
    dy2, dx2 = -dyx_min
    dy1, dx1 = -dyx_max

    # could coerce indices to maintain or enlarge area, then shift and crop
    # instead, offset dyx and indices so net integer shifts are accommodated in read.

    # set up index for new area
    inds = np.array([[rii + dy1, rif + dy2 + 1], [cii + dx1, cif + dx2 + 1]], dtype=int)

    # determine needed pads
    inds_limits = np.array([[0, detY], [0, detX]], dtype=int)
    pads = (inds - inds_limits) * np.array([[-1, 1], [-1, 1]])
    pads = np.clip(pads, a_min=0, a_max=None)
    padding = np.any(pads > 0)

    # update inds for dims needing padding
    inds[pads > 0] = inds_limits[pads > 0]

    # build slice for indexing
    s = np.s_[ri:rf, ci:cf, ...]
    for i1, i2 in inds:
        s += (np.s_[i1:i2],)

    # read data into memory / index
    d = data[s]
    d = np.ascontiguousarray(d)

    # expand pad with zeros for :-2 axes and pad data
    if padding:
        pads = np.concatenate((np.zeros((data.ndim - 2, 2), dtype=int), pads), axis=0)
        d = np.pad(d, pads, mode="constant", constant_values=fill_value)
        # possible TODO: if fill_value=np.nan, convert data chunk to float? # (with ints, pad coerces nan to -ve or +ve max)

    # update shifts and the indices to be used for crop
    if dx2 < 0:
        dyx_chnk[1] += dx2
        dx1 -= dx2
        dx2 = 0
    elif dx1 > 0:
        dyx_chnk[1] += dx1
        dx2 -= dx1
        dx1 = 0
    if dy2 < 0:
        dyx_chnk[0] += dy2
        dy1 -= dy2
        dy2 = 0
    elif dy1 > 0:
        dyx_chnk[0] += dy1
        dy2 -= dy1
        dy1 = 0

    # shift data
    if debug:
        plt.matshow(d.sum((0, 1)))
        plt.title("pre-aligned sum")
    d = _shift_images(d, dyx_chnk, dyx_mode, in_place=False)

    # crop
    dy1 *= -1
    dy2 *= -1
    dx1 *= -1
    dx2 *= -1
    if dy2 == 0:
        dy2 = None
    if dx2 == 0:
        dx2 = None
    s2 = np.s_[..., dy1:dy2, dx1:dx2]
    d = d[s2]
    if debug:
        plt.matshow(d.sum((0, 1)))
        plt.title("post-aligned sum")

    return d


# --------------------------------------------------
def virtual_images(
    data,
    nr,
    nc,
    apertures,
    dyx=None,
    dyx_mode="linear",
    fill_value=0,
    rebin=None,
    nrnc_are_chunks=False,
    progress_bar=True,
    debug=False,
):
    """
    Make virtual images from data and aperture.

    Parameters
    ----------
    data : array_like
        Multidimensional fpd data of shape (scanY, scanX, ..., detY, detX).
    nr : integer or None
        Number of rows to process at once (see Notes).
    nc : integer or None
        Number of columns to process at once (see Notes).
    apertures : array-like
        Mask applied to data before taking sum.
        Shape should be ([apN,] detY, detX).
    dyx : None or ndarray
        If not None, an ndarray of shape (2, scanY, scanX) with (y, x)
        vector of shifts to be applied to the data. See Notes.
    dyx_mode : str
        If 'pixel', `dyx` is converted to integer type for pixel resolution.
        If 'linear', `dyx` is used with sub-pixel resolution in linear interpolation.
        If 'fourier', `dyx` is used with sub-pixel resolution in Fourier shifting.
        The 'pixel' mode is very fast. The 'linear' mode is next fastest, while
    fill_value : scalar
        If `dyx` is not None, images that require extending beyond their original
        extent (so that all aperture pixels have an equivalent image pixel) are
        padded with pixels of this value. Note that `fill_value=np.nan` may be used
        with all `dyx_modes` except 'fourier'. If `data` is of integer type (which
        does not support nans), then the value is coerced by np.pad to a dtype limit
        value.
    rebin : integer or None
        Rebinning factor for detector dimensions. None or 1 for none.
        If the value is incompatible with the cropped array shape, the
        nearest compatible value will be used instead.
    nrnc_are_chunks : bool
        If True, `nr` and `nc` are interpreted as the number of chunks to
        process at once. If `data` is not chunked, `nr` and `nc` are used
        directly.
    progress_bar : bool
        If True, progress bars are printed.
    debug : bool
        If True, plots of various intermediate analyses are created.

    Returns
    -------
    Array of shape ([apN,] scanY, scanX, ...), where the first axis is
    removed if singular.

    Notes
    -----
    Aperture positions: The data shift `dyx` may be determined by a number of
    methods, including:
    - fpd.fpd_processing.center_of_mass
    - fpd.fpd_processing.phase_correlation
    while remembering to negate and centre the returned positions. For example,
    for positions comyx, values for dyx may be obtained from:

    >>> import numpy as np
    >>> shifts = -(comyx - np.percentile(comyx, 50, (-2, -1))[..., None, None])

    If the data is to be aligned to a particular scan point, then it should be
    subtracted rather than the percentile in the example above.

    The shifts should be set relative to the apertures provided. For example,
    if the apertures are positioned for the first scan point, then `dyx` for
    that point should be subtracted from all `dyx` points.

    If `nr` or `nc` are None, the entire dimension is processed at once.
    For chunked data, setting `nrnc_are_chunks` to True, and `nr` and `nc`
    to a suitable values can improve performance.

    See Also
    --------
    virtual_apertures, VirtualAnnularImages, center_of_mass

    """

    if nrnc_are_chunks:
        nr, nc = _condition_nrnc_if_chunked(data, nr, nc, True)

    if np.ndim(apertures) == 2:
        apertures = apertures[None]
    apertures = np.rollaxis(apertures, 0, 3)
    apN = apertures.shape[-1]
    # now Y, X, apN

    rebinning = (rebin is not None) and (rebin != 1)

    nondet = data.shape[:-2]
    nonscan = data.shape[2:]
    scanY, scanX = data.shape[:2]
    detY, detX = data.shape[-2:]
    colour = len(nondet) == 3

    r_if, c_if = _block_indices((scanY, scanX), (nr, nc))

    # use aperture limits to efficiently index data
    aps = apertures.sum(-1)
    rii, rif = np.where(aps.sum(axis=1) != 0)[0][[0, -1]]
    cii, cif = np.where(aps.sum(axis=0) != 0)[0][[0, -1]]

    # quick check of combined aperture fill factor (not accounting for dyx)
    if apN >= 2:
        fill_factor = (aps > 0).sum() / ((rif - rii + 1) * (cif - cii + 1))
        fill_factor_thresh = (np.pi / 4) * 0.5
        # threshold depends on many factors and is not optimised here
        if fill_factor <= fill_factor_thresh:
            print(
                "INFO: The combined aperture fill factor is small (%0.1f%%)."
                % (fill_factor * 100)
            )
            print("INFO: Separate runs for each aperture may be more efficient.")

    # prepare apertures
    if rebinning:
        (rii, rif, cii, cif), (fy, fx), (fsy, fsx) = _rebin_crop_parse(
            rebin, rii, rif, cii, cif, detY, detX
        )
        rebina = np.array([fy, fx])
        if (rebina != rebin).any():
            print(
                "Requested rebin (%d) changed to nearest value: (%d, %d)."
                % (rebin, fy, fx)
            )
            print("Possible values are:", (fsy, fsx))

    apertures = apertures[rii : rif + 1, cii : cif + 1]
    if rebinning:
        ns = (
            tuple([int(x / rebinf) for (x, rebinf) in zip(apertures.shape[:2], rebina)])
            + apertures.shape[2:]
        )
        apertures = rebinA(apertures, *ns)
    if debug:
        for i, imi in enumerate(np.moveaxis(apertures, -1, 0)):
            plt.matshow(imi)
            plt.title("aperture %02d" % (i))

    sim = np.empty(nondet + (apN,))
    if progress_bar:
        print("Calculating virtual aperture images.")
    total_ims = np.prod(nondet)
    with tqdm(
        total=total_ims,
        mininterval=0,
        leave=True,
        unit="images",
        disable=(not progress_bar),
    ) as pbar:
        for i, (ri, rf) in enumerate(r_if):
            for j, (ci, cf) in enumerate(c_if):
                if dyx is None:
                    # no shift
                    d = data[ri:rf, ci:cf, ..., rii : rif + 1, cii : cif + 1]
                    d = np.ascontiguousarray(d)
                else:
                    ## shift images
                    d = _shift_image_chunk(
                        data,
                        dyx,
                        dyx_mode,
                        fill_value,
                        ri,
                        rf,
                        ci,
                        cf,
                        rii,
                        rif,
                        cii,
                        cif,
                        debug=debug,
                    )
                # rebin and calculate
                if rebinning:
                    ns = d.shape[:-2] + tuple(
                        [int(x / rebinf) for (x, rebinf) in zip(d.shape[-2:], rebina)]
                    )
                    d = rebinA(d, *ns)
                np.einsum(
                    "ij...kl,klm->ij...m",
                    d,
                    apertures,
                    out=sim[ri:rf, ci:cf],
                    optimize=True,
                )
                pbar.update(np.prod(d.shape[:-2]))
    if progress_bar:
        print("\n")

    if apertures.shape[-1] == 1:
        sim = sim[..., 0]
    else:
        sim = np.rollaxis(sim, -1, 0)
    return sim


# --------------------------------------------------
def find_circ_centre(
    im,
    sigma,
    rmms,
    mask=None,
    plot=True,
    spf=1,
    low_threshold=0.1,
    high_threshold=0.95,
    pct=None,
    max_n=1,
    min_distance=1,
    subpix=True,
    fit_hw=3,
):
    """
    Find centre and radius of circle in image. Sub-pixel accurate in centre
    coordinates with subpix=True. The precision may be improved by setting
    spf>1 for subpixel in radius at the same time.

    Parameters
    ----------
    im : 2-D array
        Image data.
    sigma : scalar
        Smoothing width for canny edge detection .
    rmms : length 3 iterable
        Radius (min, max, step) in pixels.
    mask : array-like or None
        Mask for canny edge detection. False values are ignored.
        If None, no mask is applied.
    plot : bool
        Determines if best matching circle is plotted in matplotlib.
    spf : integer
        Sub-pixel factor used for upscaling images. Use 1 for none.
        If not None, step is forced to 1 and corresponds to 1/spf pixels.
        See also ``subpix``.
    low_threshold : float
        Lower bound for hysteresis thresholding (linking edges) in [0, 1].
    high_threshold : float
        Upper bound for hysteresis thresholding (linking edges) in [0, 1].
    pct : None or scalar
        If not None, values in the image below this percentile are masked.
    max_n : int
        Maximum number of discs to find. When the number of discs exceeds
        max_n, return max_n centres based on highest Hough intensity across
        all radii.
    min_distance : int
        Minimum distance between centre coordinates at each radii. Centres
        that are separated by at least min_distance, are returned. To find
        the maximum number of centres, use min_distance=1.
    subpix : bool
        If True, Hough space is fitted to in the region of the peak to extract
        centres with subpixel precision by fitting a 2-D Gaussian to the region
        defined by `fit_hw`.
    fit_hw : int
        Region used for fitting (2*fit_hw+1) if subpix=True.

    Returns
    -------
    Tuple of arrays of (center_y, center_x), radius.

    Notes
    -----
    Image is first scaled to full range of dtype, then upscaled if chosen.
    Canny edge detection is performed, followed by a Hough transform.
    Linking of edges is set by thresholds. See skimage.feature.canny for
    details. The best matching circle or circles are returned depending
    on the value of `max_n`.

    When multiple discs are present, increasing `high_threshold` reduces the
    number of edges considered  to those which higher connectivity. For bright
    field discs in STEM, values around 0.99 often work well.

    Subpixel resolution is much more efficient through `subpix=True` than is
    image upscalling with `spf`>1. However, more precise values may be obtained
    when the radii are subpixel with the latter setting.

    The plots are always only pixel accurate.

    To gain subpixel accuracy in radii, or for an alternative method of
    estimating the circle centres and radii,
    fpd.fpd_processing.disc_edge_properties may be used.

    Examples
    --------
    If upscaling, two calls can be made to make the calculations more efficient
    by reducing the range over which the Hough transform takes place.

    >>> from merlintools.vendor.fpd import fpd_processing as fpdp
    >>> from fpd.synthetic_data import disk_image

    >>> im = disk_image(intensity=64, radius=32)
    >>> rmms = (10, 100, 2)
    >>> spf = 1
    >>> sigma = 2
    >>> cyx, r = fpdp.find_circ_centre(im, sigma, rmms, mask=None, plot=True, spf=spf)

    >>> rmms = (r-4, r+4, 1)
    >>> spf = 4
    >>> cyx, r = fpdp.find_circ_centre(im, sigma, rmms, mask=None, plot=True, spf=spf)

    """

    # TODO
    # decide on best centre by normalising hough space to circle pixels?
    # use 3-D Hough (r, y, x) and specify min distances, etc?

    ## scale im so default thresholding works appropriately (% of range)
    # im = (im.astype(np.float)/im.max()*np.iinfo(im.dtype).max)
    # im = im.astype(im.dtype)
    im = im.astype(float, copy=False)

    if pct is not None:
        pct = np.percentile(im, pct)
        pct_mask = (im > pct).astype(bool)

        if mask is None:
            mask = pct_mask
        else:
            mask = np.logical_and(pct_mask, mask)

    if spf > 1:
        spf = float(spf)
        im = sp.ndimage.interpolation.zoom(
            im, spf, output=None, order=1, mode="reflect", prefilter=True
        )
        if mask is not None:
            mask = sp.ndimage.interpolation.zoom(
                mask * 1.0, spf, output=None, order=1, mode="reflect", prefilter=True
            )
            mask = mask > 0.5
        rmms = [x * spf for x in rmms[:2]] + [1]

    if plot:
        kwd = dict(adjustable="box-forced", aspect="equal")

        import matplotlib as mpl

        mplv = mpl.__version__
        from distutils.version import LooseVersion

        if LooseVersion(mplv) >= LooseVersion("2.2.0"):
            _ = kwd.pop("adjustable")

        f, (ax1, ax2, ax3) = plt.subplots(
            1, 3, sharex=True, sharey=True, subplot_kw=kwd, figsize=(8, 3)
        )
        # plot image
        ax1.imshow(im, interpolation="nearest", cmap="gray")  # ,
        # norm=mpl.colors.LogNorm())
        ax1.set_title("Image")

    edges = canny(
        im,
        sigma,
        mask=mask,
        use_quantiles=True,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    if plot:
        ax2.imshow(edges, interpolation="nearest", cmap="gray")
        ax2.set_title("Edges")

    # hough transform
    hough_radii = np.arange(rmms[0], rmms[1], rmms[2])
    hough_res = hough_circle(edges, hough_radii)
    if subpix:
        from .tem_tools import gaussian_2d_peak_fit
        import warnings
        from scipy.optimize import OptimizeWarning

    # return (hough_radii, hough_res)

    centers = []
    accums = []
    radii = []
    for radius, h in zip(hough_radii, hough_res):
        peaks_int = peak_local_max(h, num_peaks=max_n, min_distance=min_distance)
        accumi = h[peaks_int[:, 0], peaks_int[:, 1]]

        if subpix:
            # fit to region of Hough space
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=OptimizeWarning)
                popt, perr = gaussian_2d_peak_fit(
                    h,
                    yc=peaks_int[:, 0],
                    xc=peaks_int[:, 1],
                    fit_hw=[fit_hw * spf] * len(peaks_int),
                    plot=False,
                    maxfev=3200 * 2,
                )
                peaks_int = popt[:, 1:3]
                # use digital accum to avoid large values from bad fits of no peaks
                # could use perr for this too
                # accumi = popt[:, 0]

        centers.extend(peaks_int)
        accums.extend(accumi)
        radii.extend([radius] * len(peaks_int))
    centers = np.array(centers)
    radii = np.array(radii)
    accums = np.array(accums)

    if subpix:
        # remove nans
        keep = np.isfinite(centers[:, 0])
        centers = centers[keep]
        radii = radii[keep]
        accums = accums[keep]

    accum_order = np.argsort(accums)[::-1]
    idx = accum_order[:max_n]

    center_y, center_x = centers[idx].T
    radius = radii[idx].round(0).astype(int)

    if plot:
        # Draw the most prominent max_n circles
        imc = color.gray2rgb(im / im.max())
        for cyi, cxi, ri in zip(
            center_y.round(0).astype(int, copy=False),
            center_x.round(0).astype(int, copy=False),
            radius,
        ):
            cy, cx = circle_perimeter(cyi, cxi, ri)
            imc[cy, cx] = (1, 0, 0)
            imc[cyi, cxi] = (0, 1, 0)
        ax3.imshow(imc, interpolation="nearest")
        ax3.set_title("Detected Circle(s)")
        plt.draw()

    if spf > 1:
        center_y, center_x, radius = center_y / spf, center_x / spf, radius / spf

    return np.squeeze(np.array((center_y, center_x))).T, np.squeeze(radius)


# --------------------------------------------------
def radial_profile(data, cyx, mask=None, r_nm_pp=None, plot=False, spf=1.0):
    """
    Returns radial profile(s) by azimuthally averaging each of one or multiple images.
    Sub-pixel accurate with spf>1.

    Parameters
    ----------
    data : ndarray
        Image data of shape (...,y,x).
    cyx : length 2 tuple
        Centre y, x pixel cooridinates.
    mask : None or ndarray
        If not None, True values are retained.
    r_nm_pp : scalar or None
        Value for reciprocal nm per pixel.
        If None, values are in pixels.
    spf : scalar
        Sub-pixel factor for upscaling to give sub-pixel calculations.
        If 1, pixel level calculations.

    Returns
    -------
    r_pix, rms : tuple of ndarrays
        radii, mean intensity. The output dimentionality reflects the input one.

    Notes
    -----
    If `r_nm_pp` is not None, radii is in 1/nm, otherwise in pixels.
    This is convenient when analysing diffraction data.

    `r_pix` starts at zero.

    Examples
    --------
    >>> import numpy as np
    >>> from merlintools.vendor.fpd import fpd_processing as fpdp

    >>> cyx = (128,)*2
    >>> im_shape = (256,)*2
    >>> y, x = np.indices(im_shape)
    >>> r = np.hypot(y - cyx[0], x - cyx[1])
    >>> data = np.dstack((r**0.5, r, r**2))
    >>> data = np.rollaxis(data, 2, 0)

    >>> r_pix, radial_mean = fpdp.radial_profile(data, cyx, plot=True, spf=2)

    See Also
    --------
    radial_profiles

    """

    if r_nm_pp is None:
        r_nm_pp = 1.0
        xlab = "Pixel"
    else:
        xlab = "1/nm"

    multi_ims = True
    if len(data.shape) == 2:
        # single image, reshaped to 1 x Y x X
        data = data[None, ...]
        nr = 1
        nc = 1
        nonim_shape = (1,)
        multi_ims = False
    else:
        # fpd data with images in last 2 dims
        nonim_shape = data.shape[:-2]

    if spf > 1:
        spf = float(spf)
        r_nm_pp /= spf
        cyx = [x * spf for x in cyx]
        data = sp.ndimage.interpolation.zoom(
            data,
            (1,) * len(nonim_shape) + (spf,) * 2,
            output=None,
            order=3,
            mode="reflect",
            prefilter=True,
        )
    im_shape = data.shape[-2:]
    y, x = np.indices(im_shape)
    r = np.hypot(y - cyx[0], x - cyx[1])
    r = r.round(0).astype(np.int)  # need int for bincounting

    if mask is not None:
        if spf > 1:
            mask = sp.ndimage.interpolation.zoom(
                mask, spf, output=None, order=0, mode="reflect", prefilter=True
            )
        r = r[mask]

    for i in range(np.prod(nonim_shape)):
        nd_indx = np.unravel_index(i, nonim_shape)

        di = data[nd_indx]
        if mask is not None:
            di = di[mask]

        with np.errstate(invalid="ignore", divide="ignore"):
            tbin = np.bincount(r.ravel(), di.ravel())
            nr = np.bincount(r.ravel())
            radial_mean = tbin / nr
        if i == 0:
            rms = np.empty(nonim_shape + radial_mean.shape)
        rms[nd_indx] = radial_mean[:]
    r_pix = np.arange(radial_mean.shape[0]) * r_nm_pp

    if plot:
        f, ax = plt.subplots()
        ax.plot(r_pix, rms.T)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(xlab)
        plt.ylabel("Mean Intensity")
        plt.tight_layout()
        plt.draw()

    if multi_ims is False:
        rms = rms[0]
    return (r_pix, rms)


# --------------------------------------------------
def radial_profiles(
    data,
    nr,
    nc,
    cyx,
    dyx=None,
    mask=None,
    r_lim_mode="corner_max",
    parallel=True,
    ncores=None,
    parallel_mode="thread",
    print_stats=True,
    nrnc_are_chunks=False,
    progress_bar=True,
):
    """
    Returns radial profiles by azimuthally averaging data.

    Parameters
    ----------
    data : array_like
        Multidimensional fpd data of shape (scanY, scanX, ..., detY, detX).
    nr : integer or None
        Number of rows to process at once (see Notes).
    nc : integer or None
        Number of columns to process at once (see Notes).
    cyx : length 2 iterable or ndarray
        If a length 2 iterable, the centre (y, x) pixel cooridinates. Otherwise,
        the shape should be of shape (2, scanY, scanX) with (y, x) in the first
        dimension. See Notes.
    dyx : None or ndarray
        If not None, and `cyx` is a single centre coordinate, an ndarray of shape
        (2, scanY, scanX) with (y, x) vector of shifts to be applied to `cyx`. See
        Notes.
    mask : None or ndarray
        If not None, values of the detector are retained where mask==True. The mask
        is fixed in position regardless of `cyx` and `dyx` values.
    r_lim_mode : str
        One of ['corner_max', 'corner_min', 'edge_max', 'edge_min'] defining the
        radii limit mode. The values correspond to the maximum and minimum corner
        distances, and the maximum and minimum edge distances. `r_lim_mode='corner_max'`
        contains the maximum of valid data, while `r_lim_mode='edge_min'` corresponds
        to radii where all angles exist. Missing values are replaced with nans.
    parallel : bool
        If True, the calculations are processed in parallel.
    ncores : None or int
        Number of cores to use for parallel execution. If None, all cores
        are used.
    parallel_mode : str
        The mode to use for parallel processing.
        If 'thread' use multithreading.
        If 'process' use multiprocessing.
        Which is faster depends on the calculations performed.
    print_stats : bool
        If True, calculation progress is printed to stdout.
    nrnc_are_chunks : bool
        If True, `nr` and `nc` are interpreted as the number of chunks to
        process at once. If `data` is not chunked, `nr` and `nc` are used
        directly.
    progress_bar : bool
        If True, progress bars are printed.

    Returns
    -------
    r_pix, rms : tuple of ndarrays
        1-D array of radii (in pixels) of length n, and ndarray of mean intensity of
        shape (scanY, scanX, ..., n).

    Notes
    -----
    The centre positions `cyx` may be determined determined by a number of methods,
    including:
    - fpd.fpd_processing.center_of_mass
    - fpd.fpd_processing.phase_correlation
    The CoM values may be used directly, but the PC values are relative (to the
    reference) and so must be corrected.

    Alternatively, a single `cyx` coordinate may be used and data on how the
    centre coordinate must be moved for each scan point, `dyx`, may be supplied.

    If `nr` or `nc` are None, the entire dimension is processed at once.
    For chunked data, setting `nrnc_are_chunks` to True, and `nr` and `nc`
    to a suitable values can improve performance.

    Examples
    --------
    Compare output to that from the simpler radial_profile:

    >>> import numpy as np
    >>> from merlintools.vendor.fpd import fpd_processing as fpdp

    >>> cyx = (128,)*2
    >>> im_shape = (256,)*2
    >>> y, x = np.indices(im_shape)
    >>> r = np.hypot(y - cyx[0], x - cyx[1])
    >>> data = np.dstack((r**0.5, r, r**2))
    >>> data = np.rollaxis(data, 2, 0)

    >>> r_pix, radial_mean = fpdp.radial_profile(data, cyx)
    >>> r_pix2, radial_mean2 = fpdp.radial_profiles(data[None], 9, 9, cyx, r_lim_mode='edge_max')
    >>> radial_mean = radial_mean.T
    >>> radial_mean2 = radial_mean2[0].T

    >>> import matplotlib.pylab as plt
    >>> plt.ion()
    >>> plt.semilogy(r_pix, radial_mean, '--s')
    >>> plt.semilogy(r_pix2, radial_mean2, '-k')

    See Also
    --------
    radial_profile

    """

    # TODO
    # add spf (sub-pixel factor)?

    nondet = data.shape[:-2]
    nonscan = data.shape[2:]
    scanY, scanX = data.shape[:2]
    detY, detX = data.shape[-2:]
    colour = len(nondet) == 3

    ## condition inputs
    r_lim_modes = ["corner_max", "corner_min", "edge_max", "edge_min"]
    if r_lim_mode not in r_lim_modes:
        raise ValueError("'r_lim_mode' must be one of " + str(r_lim_modes))

    cyx = np.array(cyx, dtype=float)
    if dyx is not None:
        dyx = np.array(dyx, dtype=float)

    cyx_single = cyx.size == 2
    if not cyx_single:
        if dyx is not None:
            raise ValueError(
                "When 'cyx' is specified at each scan point, 'dyx' must be None."
            )
        else:
            # use cyx as is
            pass
    else:
        # expand dims
        cyx = cyx[..., None, None]
        if dyx is not None:
            # combine centre and shift
            cyx = cyx + dyx
        else:
            # tile cyx to correct shape
            cyx = np.tile(cyx, (1, scanY, scanX))

    ## determine limits for 'valid' data
    corners = np.array([[0, 0], [0, detX], [detY, 0], [detY, detX]])
    for i in range(cyx.ndim - 1):
        corners = corners[..., None]
    cyx_detlas = cyx[None] - corners

    rs = np.hypot(cyx_detlas[1], cyx_detlas[2])
    # maximum corner distance
    corner_max = np.floor(rs.max()).astype(int)
    # minimum corner distance
    corner_min = np.floor(rs.min()).astype(int)

    edge_dist = np.abs(cyx_detlas[[0, 3]])
    # maximum square distance
    edge_max = np.floor(edge_dist.max()).astype(int)
    # minimum square distance (complete circle)
    edge_min = np.floor(edge_dist.min()).astype(int)

    # select limit
    r_lim_mode_dict = dict(
        zip(r_lim_modes, [corner_max, corner_min, edge_max, edge_min])
    )
    r_lim = r_lim_mode_dict[r_lim_mode]

    ## function to map
    def _radial_profile_calc(im, detY, detX, cyx, mask, r_lim):
        y, x = np.indices([detY, detX])
        r = np.hypot(y - cyx[0], x - cyx[1])
        r = r.round(0).astype(np.int)

        if mask is not None:
            r = r[mask]
            im = im[mask]

        with np.errstate(invalid="ignore", divide="ignore"):
            tbin = np.bincount(r.ravel(), im.ravel())
            nr = np.bincount(r.ravel())
            radial_mean = tbin / nr

        # clip / expand data
        n = r_lim - radial_mean.size
        if n < 0:
            radial_mean = radial_mean[:r_lim]
        elif n > 0:
            radial_mean = np.pad(radial_mean, [0, n], constant_values=np.nan)

        return radial_mean

    rtn = map_image_function(
        data,
        nr=nr,
        nc=nc,
        func=_radial_profile_calc,
        params={"detY": detY, "detX": detX, "mask": mask, "r_lim": r_lim},
        mapped_params={"cyx": np.moveaxis(cyx, 0, -1)},
        parallel=parallel,
        ncores=ncores,
        parallel_mode=parallel_mode,
        print_stats=print_stats,
        nrnc_are_chunks=nrnc_are_chunks,
        progress_bar=progress_bar,
    )

    # move radial_mean axis to end
    rms = np.moveaxis(np.array(rtn, copy=False), 0, -1)
    r_pix = np.arange(r_lim)
    return r_pix, rms


def _condition_nrnc_if_chunked(data, nr, nc, print_enabled):
    """
    Determine if data is chunked and set nr and nc as multiples if True.

    """
    if nr is None or nc is None:
        return None, None

    isdask = str(type(data)) == "<class 'dask.array.core.Array'>"
    if isdask:
        if print_enabled:
            print(
                "Dask chunk comprehension is not implemented. Leaving (nr, nc) as:",
                (nr, nc),
            )
            print("Performance may improve with larger nr and nc.")
        return nr, nc

    try:
        scan_chunks = data.chunks[:2]
        nr, nc = [x * y for x, y in zip(scan_chunks, (nr, nc))]
        if print_enabled:
            print("Data is chunked, setting (nr, nc) to:", (nr, nc))
            if scan_chunks[0] == 1 or scan_chunks[1] == 1:
                print(
                    "One or more scan chunks are 1, performance may improve with larger nr and nc."
                )
    except AttributeError:
        # not chunked data
        if print_enabled:
            print("Data is not chunked, leaving (nr, nc) as:", (nr, nc))
            print("Performance may improve with larger nr and nc.")
    return nr, nc


class DummyFile(object):
    def flush(self):
        pass

    def write(self, x):
        pass


def _comf(d, aperture, yi0, xi0, thr):
    """
    CoM process data function that operates on single image at a time.

    See 'center_of_mass' for 'thr' and 'aperture' documentation.

    d is 2-D image.
    yixi is index array, as defined in calling function.

    """

    if thr is None:
        pass
    elif isinstance(thr, Number):
        d = d >= thr
    elif isinstance(thr, str):
        if thr.lower() == "otsu":
            try:
                thr_val = threshold_otsu(d)
            except TypeError:
                # TypeError: Cannot cast array data from dtype('uint64') to dtype('int64') according to the rule 'safe'
                thr_val = threshold_otsu(d.astype(float, copy=False))
            except Exception as e:
                print(e)
                # most likely a ValueError, but could be anything
                # ValueError: threshold_otsu is expected to work with images having more than one color. The input image seems to have just one color 0.
                thr_val = np.nan
            d = d >= thr_val
        else:
            # string not understood
            pass
    elif callable(thr):
        # function
        d = thr(d)

    if aperture is not None:
        d = d * aperture

    ds = d.sum().astype(float, copy=False)  # sum_im
    comi = np.array([(d * yi0[:, None]).sum(), (d * xi0[None, :]).sum()]) / ds

    return comi


# --------------------------------------------------
def center_of_mass(
    data,
    nr,
    nc,
    aperture=None,
    pre_func=None,
    thr=None,
    rebin=None,
    parallel=True,
    ncores=None,
    parallel_mode="thread",
    print_stats=True,
    nrnc_are_chunks=False,
    origin="top",
    progress_bar=True,
):
    """
    Calculate a centre of mass image from fpd data. The results are
    naturally sub-pixel resolution.

    Parameters
    ----------
    data : array_like
        Mutidimensional data of shape (scanY, scanX, ..., detY, detX).
    nr : integer or None
        Number of rows to process at once (see Notes).
    nc : integer or None
        Number of columns to process at once (see Notes).
    aperture : array_like or None
        If not None, a mask of shape (detY, detX), applied to diffraction
        data after `pre_func` processing. Note, the data is automatically
        cropped according to the mask for efficiency.
    pre_func : callable
        Function that operates (out-of-place) on data before processing.
        out = pre_func(in), where in is nd_array of shape (n, detY, detX).
    thr : object
        Control thresholding of the diffraction images.
        If None, no thresholding.
        If scalar, threshold value.
        If string, 'otsu' for otsu thresholding.
        If callable, function(2-D array) returns thresholded image.
    rebin : integer or None
        Rebinning factor for detector dimensions. None or 1 for none.
        If the value is incompatible with the cropped array shape, the
        nearest compatible value will be used instead.
    parallel : bool
        If True, the calculations are processed in parallel.
    ncores : None or int
        Number of cores to use for parallel execution. If None, all cores
        are used.
    parallel_mode : str
        The mode to use for parallel processing.
        If 'thread' use multithreading.
        If 'process' use multiprocessing.
        Which is faster depends on the calculations performed.
    print_stats : bool
        If True, statistics on the analysis are printed.
    nrnc_are_chunks : bool
        If True, `nr` and `nc` are interpreted as the number of chunks to
        process at once. If `data` is not chunked, `nr` and `nc` are used
        directly.
    origin : str
        Controls y-origin of returned data. If origin='top', pythonic indexing
        is used. If origin='bottom', increasing y is up.
    progress_bar : bool
        If True, progress bars are printed.

    Returns
    -------
    Array of shape (yx, scanY, scanX, ...).
    Increasing Y, X CoM is disc shift up, right in image.

    Notes
    -----
    The order of operations is rebinning, pre_func, threshold, aperture,
    and CoM.

    If `nr` or `nc` are None, the entire dimension is processed at once.
    For chunked data, setting `nrnc_are_chunks` to True, and `nr` and `nc`
    to a suitable values can improve performance.

    The execution of pre_func is not multithreaded, so it could employ
    paralellisation for cpu intensive calculations.

    Examples
    --------
    Using an aperture and rebinning:

    >>> import numpy as np
    >>> from merlintools.vendor.fpd import fpd_processing as fpdp
    >>> from fpd.synthetic_data import disk_image, fpd_data_view

    >>> radius = 32
    >>> im = disk_image(intensity=1e3, radius=radius, size=256, upscale=8, dtype='u4')
    >>> data = fpd_data_view(im, (32,)*2, colours=0)
    >>> ap = fpdp.virtual_apertures(data.shape[-2:], cyx=(128,)*2, rio=(0, 48), sigma=0, aaf=1)
    >>> com_y, com_x = fpdp.center_of_mass(data, nr=9, nc=9, rebin=3, aperture=ap)


    """

    # Possible alternative was not as fast in tests:
    # from scipy.ndimage.measurements import center_of_mass

    if nrnc_are_chunks:
        nr, nc = _condition_nrnc_if_chunked(data, nr, nc, print_stats)

    nondet = data.shape[:-2]
    nonscan = data.shape[2:]
    scanY, scanX = data.shape[:2]
    detY, detX = data.shape[-2:]

    # crop and rebin
    rtn = _parse_crop_rebin(
        crop_r=None,
        detY=detY,
        detX=detX,
        cyx=None,
        aperture=aperture,
        rebin=rebin,
        print_stats=print_stats,
    )
    cropped_im_shape, rebinf, rebinning, rii, rif, cii, cif = rtn

    if aperture is not None:
        aperture = aperture[rii : rif + 1, cii : cif + 1].astype(np.float, copy=False)
        if rebinning:
            ns = tuple([int(x / rebin) for x in aperture.shape])
            aperture = rebinA(aperture, *ns)

    r_if, c_if = _block_indices((scanY, scanX), (nr, nc))
    com_im = np.zeros(nondet + (2,), dtype=np.float)
    yi, xi = np.indices((detY, detX))
    yi = yi[::-1, ...]  # reverse order so increasing Y is up.

    yixi = np.concatenate((yi[..., None], xi[..., None]), 2)
    yixi = yixi[rii : rif + 1, cii : cif + 1, :].astype(np.float)
    if rebinning:
        ns = tuple([int(x / rebin) for x in yixi.shape[:2]]) + yixi.shape[2:]
        yixi = rebinA(yixi, *ns)
    yi0 = yixi[:, 0, 0]
    xi0 = yixi[0, :, 1]

    if print_stats:
        print("Calculating centre-of-mass")
        tqdm_file = sys.stderr
    else:
        tqdm_file = DummyFile()
    total_nims = np.prod(nondet)
    with tqdm(
        total=total_nims,
        file=tqdm_file,
        mininterval=0,
        leave=True,
        unit="images",
        disable=(not progress_bar),
    ) as pbar:
        for i, (ri, rf) in enumerate(r_if):
            for j, (ci, cf) in enumerate(c_if):
                d = data[
                    ri:rf, ci:cf, ..., rii : rif + 1, cii : cif + 1
                ]  # .astype(np.float)
                d = np.ascontiguousarray(d)
                if rebinning:
                    ns = d.shape[:-2] + tuple([int(x / rebin) for x in d.shape[-2:]])
                    d = rebinA(d, *ns)
                else:
                    ns = d.shape

                # modify with function
                if pre_func is not None:
                    d = pre_func(d.reshape((-1,) + d.shape[-2:]))
                    d.shape = ns

                partial_comf = partial(
                    _comf, aperture=aperture, yi0=yi0, xi0=xi0, thr=thr
                )

                d_shape = d.shape  # scanY, scanX, ..., detY, detX
                d.shape = (np.prod(d_shape[:-2]),) + d_shape[-2:]
                # (scanY, scanX, ...), detY, detX

                rslt = _run(
                    partial_comf,
                    d,
                    parallel=parallel,
                    parallel_mode=parallel_mode,
                    ncores=ncores,
                )
                rslt = np.asarray(rslt)

                # print(d_shape, com_im[ri:rf,ci:cf,...].shape, rslt.shape)
                rslt.shape = d_shape[:-2] + (2,)
                com_im[ri:rf, ci:cf, ...] = rslt
                pbar.update(np.prod(d.shape[:-2]))
    if print_stats:
        print("\n")
    com_im = (com_im) / rebinf**2

    # roll: (scanY, scanX, ..., yx) to (yx, scanY, scanX, ...)
    com_im = np.rollaxis(com_im, -1, 0)

    # default origin implementation is bottom
    if origin.lower() == "top":
        com_im[0] = nonscan[0] - 1 - com_im[0]

    # print some stats
    if print_stats:
        _print_shift_stats(com_im)

    return com_im


# --------------------------------------------------
def _g2d_der(sigma, truncate=4.0):
    """
    Returns tuple (gy, gy) of first partial derivitives of Gaussian.
    Y increasing is up.

    """

    d = int(np.ceil(sigma * truncate))
    dtot = 2 * d + 1
    y, x = np.indices((dtot,) * 2) - d

    gx = -x / (2 * np.pi * sigma**4) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gy = -np.rollaxis(gx, 1, 0)  # -ve to have y increasin upward
    # plt.matshow(gx)
    # plt.matshow(gy)

    return (gy, gx)


# --------------------------------------------------
def _grad(im, gxy):
    """
    Calculate gradient by colvolving 'im' with 'gxy'.

    """
    imf = fft_module.rfft2(im)
    cy = fft_module.irfft2(gxy[1] * imf)
    cx = fft_module.irfft2(gxy[0] * imf)
    img = np.hypot(cy, cx)
    return img


def _process_grad(
    d,
    pre_func,
    mode,
    sigma,
    truncate,
    gxy,
    parallel,
    ncores,
    parallel_mode,
    der_clip_fraction,
    der_clip_max_pct,
    post_func,
):
    """Calculate gradients."""

    if pre_func is not None:
        ns = d.shape
        d = pre_func(d.reshape((-1,) + d.shape[-2:]))
        d.shape = ns

    if gxy is not None:
        if mode == "1d":
            # Now that we have thread control, we could have option of parallel.
            # Might not be much savings.
            df = d.astype(float, copy=False)
            gy = gaussian_filter1d(
                df, sigma=sigma, axis=-2, order=1, mode="reflect", truncate=truncate
            )
            gx = gaussian_filter1d(
                df, sigma=sigma, axis=-1, order=1, mode="reflect", truncate=truncate
            )
            gm = np.hypot(gy, gx)
        elif mode == "2d":
            partial_grad = partial(_grad, gxy=gxy)
            d_shape = d.shape
            d.shape = (np.prod(d_shape[:-2]),) + d_shape[-2:]

            rslt = _run(
                partial_grad,
                d,
                parallel=parallel,
                parallel_mode=parallel_mode,
                ncores=ncores,
            )
            gm = np.asarray(rslt)
        else:
            raise ValueError("Mode value unknown.")

        if der_clip_fraction != 0:
            ref = np.percentile(gm, der_clip_max_pct, axis=(-2, -1))
            clip_low = der_clip_fraction * ref
            gm[gm < clip_low[:, None, None]] = 0
    else:
        gm = d
    if post_func is not None:
        ns = gm.shape
        gm = post_func(gm.reshape((-1,) + gm.shape[-2:]))
        gm.shape = ns

    gm = gm.reshape((-1,) + gm.shape[-2:])
    return gm


# --------------------------------------------------
def phase_correlation(
    data,
    nr,
    nc,
    cyx=None,
    crop_r=None,
    sigma=2.0,
    spf=100,
    pre_func=None,
    post_func=None,
    mode="2d",
    ref_im=None,
    rebin=None,
    der_clip_fraction=0.0,
    der_clip_max_pct=99.9,
    truncate=4.0,
    parallel=True,
    ncores=None,
    parallel_mode="thread",
    print_stats=True,
    nrnc_are_chunks=False,
    origin="top",
    progress_bar=True,
):
    """
    Perform phase correlation on 4-D data using efficient upscaling to
    achieve sub-pixel resolution.

    Parameters
    ----------
    data : array_like
        Mutidimensional data of shape (scanY, scanX, ..., detY, detX).
    nr : integer or None
        Number of rows to process at once (see Notes).
    nc : integer or None
        Number of columns to process at once (see Notes).
    cyx : length 2 iterable or None
        Centre of disk in pixels (cy, cx).
        If None, centre is used.
    crop_r : scalar or None
        Radius of circle about cyx defining square crop limits used for
        cross-corrolation, in pixels.
        If None, the maximum square array about cyx is used.
    sigma : scalar
        Smoothing of Gaussian derivative. If set to 0, no derivative is
        performed. If negative, `ref_im` is modified by subtracting `ref_im`
        smoothed by a Gaussian of this standard deviation.
    spf : integer
        Sub-pixel factor i.e. 1/spf is resolution.
    pre_func : callable
        Function that operates (out-of-place) on data before processing.
        out = pre_func(in), where in is nd_array of shape (n, detY, detX).
    post_func : callable
        Function that operates (out-of-place) on data after derivitive.
        out = post_func(in), where in is nd_array of shape (n, detY, detX).
    mode : string
        Derivative type.
        If '1d', 1d convolution; faster but not so good for high sigma.
        If '2d', 2d convolution; more accurate but slower.
    ref_im : None or ndarray
        2-D image used as reference.
        If None, the first probe position is used.
    rebin : integer or None
        Rebinning factor for detector dimensions. None or 1 for none.
        If the value is incompatible with the cropped array shape, the
        nearest compatible value will be used instead.
        'cyx' and 'crop_r' are for the original image and need not be modified.
        'sigma' and 'spf' are scaled with rebinning factor, as are output shifts.
    der_clip_fraction : float
        Fraction of `der_clip_max_pct` in derivative images below which will be
        to zero.
    der_clip_max_pct : float
        Percentile of derivative image to serve as reference for `der_clip_fraction`.
    truncate : scalar
        Number of sigma to which Gaussians are calculated.
    parallel : bool
        If True, derivative and correlation calculations are processed in parallel.
    ncores : None or int
        Number of cores to use for parallel execution. If None, all cores
        are used.
    parallel_mode : str
        The mode to use for parallel processing.
        If 'thread' use multithreading.
        If 'process' use multiprocessing.
        Which is faster depends on the calculations performed.
    print_stats : bool
        If True, statistics on the analysis are printed.
    nrnc_are_chunks : bool
        If True, `nr` and `nc` are interpreted as the number of chunks to
        process at once. If `data` is not chunked, `nr` and `nc` are used
        directly.
    origin : str
        Controls y-origin of returned data. If origin='top', pythonic indexing
        is used. If origin='bottom', increasing y is up.
    progress_bar : bool
        If True, progress bars are printed.

    Returns
    -------
    Tuple of (shift_yx, shift_err, shift_difp, ref), where:
    shift_yx : array_like
        Shift array in pixels, of shape ((y,x), scanY, scanX, ...).
        Increasing Y, X is disc shift up, right in image.
    shift_err : 2-D array
        Translation invariant normalized RMS error in correlations.
        See skimage.feature.register_translation for details.
    shift_difp : 2-D array
        Global phase difference between the two images.
        (should be zero if images are non-negative).
        See skimage.feature.register_translation for details.
    ref : 2-D array
        Reference image.

    Notes
    -----
    The order of operations is rebinning, pre_func, derivative,
    post_func, and correlation.

    If `nr` or `nc` are None, the entire dimension is processed at once.
    For chunked data, setting `nrnc_are_chunks` to True, and `nr` and `nc`
    to a suitable values can improve performance.

    Specifying 'crop_r' (and appropriate cyx) can speed up calculation significantly.

    The execution of 'pre_func' and 'post_func' are not multithreaded, so
    they could employ paralellisation for cpu intensive calculations.

    Efficient upscaling is based on:
    http://scikit-image.org/docs/stable/auto_examples/transform/plot_register_translation.html
    http://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.register_translation

    """
    from skimage import __version__ as skiv
    from distutils.version import LooseVersion

    if LooseVersion(skiv) < LooseVersion("0.17.0"):
        from skimage.feature import register_translation as trans_func
    else:
        from skimage.registration import phase_cross_correlation as trans_func

    # der_clip_max_pct=99.9 'ignores' (256**2)*0.001 ~ 65 pixels.
    # (256**2)*0.001 / (2*3.14) / 2 ~ 5. == ignoring of 5 pix radius, 2 pix width torus

    if nrnc_are_chunks:
        nr, nc = _condition_nrnc_if_chunked(data, nr, nc, print_stats)

    nondet = data.shape[:-2]
    nonscan = data.shape[2:]
    scanY, scanX = data.shape[:2]
    detY, detX = data.shape[-2:]

    r_if, c_if = _block_indices((scanY, scanX), (nr, nc))

    rtn = _parse_crop_rebin(
        crop_r=crop_r,
        detY=detY,
        detX=detX,
        cyx=cyx,
        aperture=None,
        rebin=rebin,
        print_stats=print_stats,
    )
    cropped_im_shape, rebinf, rebinning, rii, rif, cii, cif = rtn

    # gradient of gaussian
    if sigma <= 0:
        gxy = None
    else:
        gy, gx = _g2d_der(sigma, truncate=truncate)

        # cut / pad
        nny, nnx = gy.shape
        if nny > detY:
            dy = nny - detY
            dy1 = dy // 2
            dy2 = dy - dy1
            gy = gy[dy1:-dy2, :]
            gx = gx[dy1:-dy2, :]
        elif nny < detY:
            dy = detY - nny
            dy1 = dy // 2
            dy2 = dy - dy1
            gy = np.pad(gy, [(dy1, dy2), (0, 0)], mode="constant", constant_values=0)
            gx = np.pad(gx, [(dy1, dy2), (0, 0)], mode="constant", constant_values=0)
        if nnx > detX:
            dx = nnx - detX
            dx1 = dx // 2
            dx2 = dx - dx1
            gy = gy[:, dx1:-dx2]
            gx = gx[:, dx1:-dx2]
        elif nnx < detX:
            dx = detX - nnx
            dx1 = dx // 2
            dx2 = dx - dx1
            gy = np.pad(gy, [(0, 0), (dx1, dx2)], mode="constant", constant_values=0)
            gx = np.pad(gx, [(0, 0), (dx1, dx2)], mode="constant", constant_values=0)

        # shift to centre of crop
        gxy = np.array([gx, gy])
        ry = detY // 2 - (rii + rif) // 2
        rx = detX // 2 - (cii + cif) // 2
        gyx = np.roll(gxy, -ry, 1)
        gyx = np.roll(gxy, -rx, 2)

        # crop
        gxy = gxy[:, rii : rif + 1, cii : cif + 1]
        # could pad to rii:rif+1 etc to start with, rather than centre and shift.

    # rebinning
    rebinf = 1
    rebinning = rebin is not None and rebin != 1
    if rebinning:
        f, fs = nearest_int_factor(cropped_im_shape[0], rebin)
        if rebin != f:
            print("Image data cropped to:", cropped_im_shape)
            print(
                "Requested rebin (%d) changed to nearest value: %d. Possible values are:"
                % (rebin, f),
                fs,
            )
            rebin = f
        rebinf = rebin
        sigma = float(sigma) / rebinf
        spf = int(float(spf) * rebinf)

        if gxy is not None:
            ns = gxy.shape[:-2] + tuple([int(x / rebin) for x in gxy.shape[-2:]])
            gxy = rebinA(gxy, *ns)
    rebinned_im_shape = tuple([x // rebinf for x in cropped_im_shape])

    # der fft
    if gxy is not None:
        gxy = fft_module.rfft2(gxy, axes=(-2, -1)).conjugate()

    ### ref im
    if ref_im is None:
        # use first point
        ref_im = data[0, 0, ...]
        for i in range(len(nondet) - 2):
            ref_im = ref_im[0]
    else:
        # provided option
        ref_im = ref_im

    if sigma < 0:
        ref_im = ref_im.astype(float, copy=False)
        ref_im = ref_im - gaussian_filter(ref_im, np.abs(sigma), truncate=truncate)

    ref = ref_im[rii : rif + 1, cii : cif + 1]
    for t in range(len(nondet)):
        ref = np.expand_dims(ref, 0)  # ref[None, None, None, ...]
    if rebinning:
        ns = ref.shape[:-2] + tuple([int(x / rebin) for x in ref.shape[-2:]])
        ref = rebinA(ref, *ns)
    ref = _process_grad(
        ref,
        pre_func,
        mode,
        sigma,
        truncate,
        gxy,
        parallel,
        ncores,
        parallel_mode,
        der_clip_fraction,
        der_clip_max_pct,
        post_func,
    )[0]
    ref_f = fft_module.fft2(ref)

    shift_yx = np.empty(nondet + (2,))
    shift_err = np.empty(nondet)
    shift_difp = np.empty_like(shift_err)

    if print_stats:
        print("\nPerforming phase correlation")
        tqdm_file = sys.stderr
    else:
        tqdm_file = DummyFile()
    total_nims = np.prod(nondet)
    with tqdm(
        total=total_nims,
        file=tqdm_file,
        mininterval=0,
        leave=True,
        unit="images",
        disable=(not progress_bar),
    ) as pbar:
        for i, (ri, rf) in enumerate(r_if):
            for j, (ci, cf) in enumerate(c_if):
                # read selected data (into memory if hdf5)
                d = data[ri:rf, ci:cf, ..., rii : rif + 1, cii : cif + 1]
                d = np.ascontiguousarray(d)
                if rebinning:
                    ns = d.shape[:-2] + tuple([int(x / rebinf) for x in d.shape[-2:]])
                    d = rebinA(d, *ns)

                # calc grad
                gm = _process_grad(
                    d,
                    pre_func,
                    mode,
                    sigma,
                    truncate,
                    gxy,
                    parallel,
                    ncores,
                    parallel_mode,
                    der_clip_fraction,
                    der_clip_max_pct,
                    post_func,
                )
                # gm is (n, detY, detX), with last 2 rebinned

                # fft of processed grad
                if parallel and LooseVersion(sp.__version__) >= LooseVersion("1.4.0"):
                    gm = fft_module.fft2(gm, axes=(-2, -1), workers=ncores)
                else:
                    gm = fft_module.fft2(gm, axes=(-2, -1))

                # do correlation
                partial_reg = partial(
                    trans_func, ref_f, upsample_factor=spf, space="fourier"
                )
                rslt = _run(
                    partial_reg,
                    gm,
                    parallel=parallel,
                    parallel_mode=parallel_mode,
                    ncores=ncores,
                )

                shift, error, phasediff = np.asarray(rslt).T
                shift = np.array(shift.tolist())
                # -ve shift to swap source/ref coords to be consistent with
                # other phase analyses
                shift *= -1.0

                shift_yx[ri:rf, ci:cf].flat = shift
                shift_err[ri:rf, ci:cf].flat = error
                shift_difp[ri:rf, ci:cf].flat = phasediff

                pbar.update(np.prod(d.shape[:-2]))
    if print_stats:
        print("")
        sys.stdout.flush()
    shift_yx = np.rollaxis(shift_yx, -1, 0)

    # reverse y for shift up being positive
    flp = np.array([-1, 1])
    for i in range(len(nonscan)):
        flp = np.expand_dims(flp, -1)
    shift_yx = shift_yx * flp

    # default origin implementation is bottom
    if origin.lower() == "top":
        shift_yx[0] = -shift_yx[0]

    # scale shifts for rebinning
    if rebinning:
        shift_yx *= rebinf

    if gxy is not None:
        ref = fft_module.fftshift(ref)

    # print stats
    if print_stats:
        _print_shift_stats(shift_yx)

    return shift_yx, shift_err, shift_difp, ref


def _print_shift_stats(shift_yx):
    """
    Prints statistics of 'shift_yx' array

    shift_yx is of shapeyx, scanY, scanX, ...
    """

    # handle nans
    n_vals = np.prod(
        shift_yx.shape[1:]
    )  # if >1 colour is present, then n_vals > scan points
    shift_yx = shift_yx.reshape([2, -1])  # 2 x n_vals
    b_nan = np.isnan(shift_yx[0])  # if one of y or x are nan, the other is too
    n_nans = b_nan.sum()
    if n_nans != 0:
        print("WARNING: %d / %d values are nan!" % (n_nans, n_vals))
        shift_yx = shift_yx[:, b_nan == False]

    shift_yx_mag = (shift_yx**2).sum(0) ** 0.5
    shift_yxm = np.concatenate((shift_yx, shift_yx_mag[None, ...]), axis=0)

    non_yx_axes = tuple(range(1, len(shift_yxm.shape)))
    yxm_mn, yxm_std = shift_yxm.mean(non_yx_axes), shift_yxm.std(non_yx_axes)
    yxm_min, yxm_max = shift_yxm.min(non_yx_axes), shift_yxm.max(non_yx_axes)
    yxm_ptp = yxm_max - yxm_min

    print("{:10s}{:>8s}{:>11s}{:>11s}".format("Statistics", "y", "x", "m"))
    print("{:s}".format("-" * 40))
    print("{:6s}: {:10.3f} {:10.3f} {:10.3f}".format(*(("Mean",) + tuple(yxm_mn))))
    print("{:6s}: {:10.3f} {:10.3f} {:10.3f}".format(*(("Min",) + tuple(yxm_min))))
    print("{:6s}: {:10.3f} {:10.3f} {:10.3f}".format(*(("Max",) + tuple(yxm_max))))
    print("{:6s}: {:10.3f} {:10.3f} {:10.3f}".format(*(("Std",) + tuple(yxm_std))))
    print("{:6s}: {:10.3f} {:10.3f} {:10.3f}".format(*(("Range",) + tuple(yxm_ptp))))
    print()


def disc_edge_properties(
    im, sigma=2, cyx=None, r=None, n_angles=90, r_res_pix=0.25, plot=True
):
    """
    Calculates disc edge properties by fitting Erfs to an unwrapped disc image.

    Parameters
    ----------
    im : 2-D array
        Image of disc.
    sigma : scalar
        Estimate of edge stdev.
    cyx : length 2 iterable or None
        Centre coordinates of disc. If None, these are calculated.
    r : scalar or None
        Disc radius in pixels. If None, the value is calculated.
    n_angles : int
        The number of angles used to subdivide the full circle.
    r_res_pix : scalar
        The number of pixels used to interpolate the radius.
    plot : bool
        Determines if images are plotted.

    Returns
    -------
    dp : named tuple with the following elements:

    sigma_wt_avg : scalar
        Average sigma value, weighted if possible by fit error.
    sigma_wt_std : scalar
        Average sigma standard deviation, weighted if possible by fit error.
        Nan if no weighting is possible.
    sigma_vals : 1-D array
        Sigma values from fit.
    sigma_avg, sigma_std : scalars
        Mean and standard deviation of all sigma values.
    r_vals, r_stds : 1-D arrays
        Radius values and standard deviations from fit.
    d_vals : 1-D array
        Diameter values from fit.
    d_avg, d_std : scalars
        Mean and standard deviation of all diameter values.
    radial_axis : 1-D array
        Radial axis used to generate polar data, in pixels.
    angles : 1-D array
        Angles along which polar data is analysed, in radians.
    cyx_opt : length two 1-D array
        Disc centre value optimised by circle fit to disc edge radii.
    r_opt : scalar
        Disc radius extracted from a circle fit to disc edge radii.
    cyx_opt_err : length two 1-D array
        Disc centre error.
    r_opt_err : scalar
        Disc radius error.

    Notes
    -----
    `sigma` is used for initial value and for setting range of fit.
    Increasing value widens region fitted to.

    The angle starts at east (positive x-axis) and rotates clockwise
    towards the south (positive y-axis) when viewed with the origin at
    the top left corner.

    The disc diameter is the mean of the radii at angles pi apart, and so
    accounts for `cyx` being slightly off. If `n_angles` is odd, the last
    value is ignored when calculating the disc diameter.

    Examples
    --------
    >>> import fpd
    >>> import matplotlib.pylab as plt
    >>>
    >>> plt.ion()
    >>>
    >>> im = fpd.synthetic_data.disk_image(intensity=64, radius=32, sigma=5.0, size=256, noise=True)
    >>> cyx, r = fpd.fpd_processing.find_circ_centre(im, 2, (22, int(256/2.0), 1), spf=1, plot=False)
    >>>
    >>> dp = fpd.fpd_processing.disc_edge_properties(im, sigma=6, cyx=cyx, r=r, plot=True)
    >>> sigma_wt_avg = dp.sigma_wt_avg # etc

    """

    detY, detX = im.shape

    if cyx is None or r is None:
        cyx_, r_ = find_circ_centre(im, 2, (3, int(detY / 2.0), 1), spf=1, plot=plot)
    if cyx is None:
        cyx = cyx_
    if r is None:
        r = r_
    cy, cx = cyx

    # set up coordinated
    yi, xi = np.indices((detY, detX), dtype=float)
    yi -= cy
    xi -= cx
    ri2d = np.hypot(yi, xi)
    ti2d = np.arctan2(yi, xi)
    ti2d[ti2d < 0] += 2 * np.pi

    angles = np.linspace(0, 360, n_angles, endpoint=False) / 180.0 * np.pi
    radial_axis = np.arange(0, 2.5 * r, r_res_pix)
    rr, tt = np.meshgrid(radial_axis, angles, indexing="ij")
    xx = rr * np.cos(tt) + cx
    yy = rr * np.sin(tt) + cy

    # MAP TO RT
    rt_val = sp.ndimage.interpolation.map_coordinates(
        im.astype(float, copy=False), np.vstack([yy.flatten(), xx.flatten()])
    )
    rt_val = rt_val.reshape(rr.shape)

    if plot:
        f = plt.figure(figsize=(5, 4))
        plt.imshow(
            rt_val, extent=[tt.min(), tt.max(), rr.max(), rr.min()], aspect="auto"
        )
        plt.minorticks_on()
        plt.ylabel("Radius (pixels)")
        plt.xlabel("Angle (rad)")

        plt.figure()
        plt.plot(rt_val[:, ::18])
        plt.xlabel("Interp pixels")
        plt.ylabel("Intensity")

    # Fit edge
    der = -np.diff(rt_val, axis=0)

    # fit range
    ri2d_edge_min = np.concatenate((ri2d[[0, -1], :], ri2d[:, [0, -1]].T), axis=1).min()
    rmin = max((r - 3 * sigma), 0)
    rmax = min((r + 3 * sigma), ri2d_edge_min)

    from scipy.optimize import curve_fit

    def function(x, p1, p2, p3, p4):
        # p1, p2, p3, p4 = origin, A, sigma, offset
        y = p2 * (sp.special.erf((x - p1) / (np.sqrt(2) * p3)) + 1.0) / 2.0 + p4
        return y

    # fit range
    x = np.arange(len(rt_val))
    xmin, xmax = rmin / r_res_pix, rmax / r_res_pix
    b = np.logical_and(x >= xmin, x <= xmax)

    p0 = (r / r_res_pix, -np.percentile(rt_val, 90), sigma / r_res_pix, 0)
    popts = []
    perrs = []
    for rt_vali in rt_val.T:
        yi = rt_vali[b]
        xi = x[b]
        popt, pcov = curve_fit(f=function, xdata=xi, ydata=yi, p0=p0, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))

        popts.append(popt)
        perrs.append(perr)
    popts = np.array(popts)
    perrs = np.array(perrs)

    sigma_vals = np.abs(popts[:, 2]) * r_res_pix
    sigma_stds = np.abs(perrs[:, 2]) * r_res_pix

    r_vals = np.abs(popts[:, 0]) * r_res_pix
    r_stds = np.abs(perrs[:, 0]) * r_res_pix

    if plot:
        A = np.percentile(popts[:, 1], 50)
        fits = np.array([function(x, *pi) for pi in popts])

        inds = np.arange(len(sigma_vals))[::10]
        f, ax = plt.subplots(1, 1, figsize=(6, 8))
        pad = 0.2 * A
        for j, i in enumerate(inds):
            ax.plot(x, rt_val[:, i] + pad * j, "x")
            ax.plot(x[b], fits[i][b] + pad * j, "b-")

    # calculate averages
    sigma_std = sigma_vals.std()
    sigma_avg = sigma_vals.mean()

    # disc diameter
    if n_angles % 2:
        # odd
        r_vals_sel = r_vals[:-1]
    else:
        # even
        r_vals_sel = r_vals
    d_vals = r_vals_sel.reshape((2, -1)).sum(0)
    d_avg = d_vals.mean()
    d_std = d_vals.std()
    d_median = np.median(d_vals)

    err_is = np.where(np.isfinite(sigma_stds))[0]
    if err_is.size > 1:
        print("Calculating weighted average sigma...")
        vs = sigma_vals[err_is]
        ws = 1.0 / sigma_stds[err_is] ** 2
        sigma_wt_avg = (vs * ws).sum() / ws.sum()
        sigma_wt_std = (1.0 / ws.sum()) ** 0.5
    else:
        print("Calculating unweighted average sigma...")
        sigma_wt_avg = sigma_vals.mean()
        sigma_wt_std = np.nan
    print("Avg: %0.3f +/- %0.3f" % (sigma_wt_avg, sigma_wt_std))
    print("Std: %0.3f" % (sigma_std))

    sigma_pcts = np.percentile(sigma_vals, [10, 50, 90])
    print("Sigma percentiles (10, 50, 90): %0.3f, %0.3f, %0.3f" % tuple(sigma_pcts))

    try:
        # optimise centre
        def circ_function(x, p1, p2, p3):
            # p1, p2, p3 = A, phase, offset
            return p3 + np.abs(p1) * np.sin((x - p2))

        a0 = np.diff(np.percentile(r_vals, (2, 98)))[0] / 2.0
        a0 = max([a0, 0.5])
        ph0 = (np.argmax(r_vals) - np.argmin(r_vals)) / 2
        ph0 = angles[int(ph0)]
        o0 = d_median
        p0 = (a0, ph0, o0)
        popt, pcov = curve_fit(
            f=circ_function, xdata=angles, ydata=r_vals, p0=p0, maxfev=10000
        )
        popt[0] = np.abs(popt[0])
        perr = np.sqrt(np.diag(pcov))

        dcyx = -popt[0] * np.array([-np.cos(popt[1]), np.sin(popt[1])])
        cyx_opt = cyx + dcyx
        r_opt = popt[2]
        cy_err = np.hypot(
            perr[0] * np.cos(popt[1]), popt[0] * np.sin(popt[1]) * perr[1]
        )
        cx_err = np.hypot(
            perr[0] * np.sin(popt[1]), popt[0] * np.cos(popt[1]) * perr[1]
        )
        cyx_opt_err = np.array([cy_err, cx_err])
        r_opt_err = perr[2]
    except:
        cyx_opt = np.array(
            [
                np.nan,
            ]
            * 2
        )
        r_opt = np.nan
        cyx_opt_err = np.array(
            [
                np.nan,
            ]
            * 2
        )
        r_opt_err = np.nan

    if plot:
        plt.figure()
        plt.plot(angles, r_vals, label="data")
        plt.plot(angles, circ_function(angles, *popt), "--", label="fit")
        plt.legend()
        plt.xlabel("angle (rad)")
        plt.ylabel("r_val (pix)")
        plt.title(
            "cyx: (%0.4f, %0.4f) -> (%0.4f +/- %0.4f, %0.4f +/- %0.4f)\ncr = %0.4f -> %0.4f +/- %0.4f"
            % (
                cyx[0],
                cyx[1],
                cyx_opt[0],
                cyx_opt_err[0],
                cyx_opt[1],
                cyx_opt_err[1],
                r,
                r_opt,
                r_opt_err,
            )
        )
        plt.tight_layout()

    names = "sigma_wt_avg, sigma_wt_std, sigma_vals, sigma_avg, sigma_std, sigma_stds, r_vals, r_stds, d_vals, d_avg, d_std, d_median, radial_axis, angles, cyx_opt, r_opt, cyx_opt_err, r_opt_err"
    dp = namedtuple("dp", [ni.strip() for ni in names.split(",")])
    return dp(
        sigma_wt_avg,
        sigma_wt_std,
        sigma_vals,
        sigma_avg,
        sigma_std,
        sigma_stds,
        r_vals,
        r_stds,
        d_vals,
        d_avg,
        d_std,
        d_median,
        radial_axis,
        angles,
        cyx_opt,
        r_opt,
        cyx_opt_err=cyx_opt_err,
        r_opt_err=r_opt_err,
    )


def nrmse(ref_im, test_ims, allow_nans=False):
    """
    Euclidean normalised mean square error.

    Parameters
    ----------
    ref_im : 2-D array
        Reference image.
    test_ims : ndarray
        Images to compare.
    allow_nans : bool
        If True, any nan values are masked.

    Returns
    -------
    n : ndarrray
        Euclidean normalised mean square error.
    """

    im1 = test_ims.astype(float, copy=False)
    im2 = ref_im.astype(float, copy=False)

    if allow_nans is False:
        # n = (((im1-im2)**2).mean((-2, -1)) / ((im1**2+im2**2).mean((-2, -1))/2))**0.5
        n = (
            ((im1 - im2) ** 2).mean((-2, -1)) / (im1**2 + im2**2).mean((-2, -1)) / 2
        ) ** 0.5
    else:
        num = np.nanmean((im1 - im2) ** 2, axis=(-2, -1))
        den = np.nanmean(im1**2 + im2**2, axis=(-2, -1)) / 2
        n = (num / den) ** 0.5
    return n


def find_matching_images(
    images, aperture=None, avg_nims=3, cut_len=20, plot=True, progress_bar=True
):
    """
    Finds matching images using euclidean normalised mean square error through
    all combinations of a given number of images.

    Parameters
    ----------
    images : ndarray
        Array of images with image axes in last 2 dimensions.
    aperture : 2D array
        An aperture to apply to the images.
    avg_nims : int
        The number of images in a combination.
    cut_len : int
        The number of combinations in which to look for common images.
    progress_bar : bool
        If True, progress bars are printed.

    Returns
    -------
    named tuple 'matching' containing:

    yxi_combos : tuple of two 2-D arrays
        y- and x-indices of combinations, sorted by match quality.
    yxi_common :
        y- and x-indices of most common image in ``cut_len`` combinations.
    ims_common : 3-D array
        All images in in ``cut_len`` combinations matched with most common image.
    ims_best : 3-D array
        Best matching ``avg_nims`` images.

    Notes
    -----
    The number of combinations increases very rapidly with ``avg_nims`` and
    the number of images. Using around 100 or so images runs relatively quickly.

    Examples
    --------
    >>> from fpd.synthetic_data import disk_image, shift_array, shift_images
    >>> from merlintools.vendor.fpd import fpd_processing as fpdp

    # Generate synthetic data.
    >>> disc = disk_image(radius=32, intensity=64)
    >>> shift_array = shift_array(6, shift_min=-1, shift_max=1)

    # Set shifts on diagonal to zero.
    >>> diag_inds = [np.diag(x) for x in np.indices(shift_array[0].shape)]
    >>> shift_array[0][diag_inds] = 0
    >>> shift_array[1][diag_inds] = 0

    # Generate shifted images.
    >>> images = shift_images(shift_array, disc, noise=False)
    >>> aperture = fpdp.virtual_apertures(images.shape[-2:], cyx=(128,)*2, rio=(0, 48), sigma=0, aaf=1)

    # Find matching images.
    >>> matching = fpdp.find_matching_images(images, aperture, plot=True)
    >>> ims_best = matching.ims_best.mean(0)

    """

    # convert dask and other out-of-core to numpy
    ims_orig = np.ascontiguousarray(images)

    # flatten original images
    ims_orig_shape = ims_orig.shape
    ims_orig.shape = (-1,) + ims_orig.shape[-2:]
    n_ims = ims_orig.shape[0]

    # apply aperture and crop
    if aperture is not None:
        ri, rf = np.where(aperture.sum(0))[0][[0, -1]]
        ci, cf = np.where(aperture.sum(1))[0][[0, -1]]
        sr = slice(ri, rf + 1)
        sc = slice(ci, cf + 1)
        aperture = aperture[sr, sc]
        ims = ims_orig[:, sr, sc] * (aperture[None, ...].astype(int))
    else:
        ims = ims_orig

    # calculate nrsme for all combinations in one half diagonal
    err = np.ones((n_ims, n_ims), dtype=float)
    err[:] = np.nan
    print("Calculating NRSME for all image combinations")
    for ri, ref_im in enumerate(tqdm(ims, disable=(not progress_bar))):
        test_ims = ims[:ri]
        err_col = nrmse(ref_im, test_ims)
        err[:ri, ri] = err_col
    if plot:
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(8, 4))
        ax1.imshow(err, interpolation="nearest")
        ax1.set_xlabel("Flattened image index")
        ax1.set_ylabel("Flattened image index")
        ax1.set_title("NRSME")

    # loop over all combinations
    print("Calculating combined NRSME for all combinations of %d images" % (avg_nims))
    combs_tot = int(
        np.math.factorial(n_ims)
        / (np.math.factorial(avg_nims) * np.math.factorial(n_ims - avg_nims))
    )
    comb_vals = np.empty(combs_tot, dtype=float)
    comb_inds = np.empty((combs_tot, avg_nims), dtype=int)
    for i, inds in enumerate(
        tqdm(
            combinations(range(n_ims), avg_nims),
            total=combs_tot,
            disable=(not progress_bar),
        )
    ):
        # calculate rmse from values at intercepts of row and column slices
        ind_perms = np.array(list(combinations(inds, 2))).T
        intercept_vals = err[ind_perms[0], ind_perms[1]]
        comb_vals[i] = np.nansum(intercept_vals**2).sum() ** 0.5
        comb_inds[i] = inds

    # sort perms by rmse
    si = np.argsort(comb_vals)
    comb_vals = comb_vals[si]
    comb_inds = comb_inds[si]

    if plot:
        # Combined NRSME
        ax2.semilogx(comb_vals)
        ax2.set_xlabel("Combination index")
        ax2.set_ylabel("Combined NRSME")
        ax2.set_title("%d combinations of %d images" % (combs_tot, avg_nims))
        plt.tight_layout()

        # map of scan locations
        gri, gci = np.unravel_index(comb_inds, ims_orig_shape[:2])
        map_im = np.zeros(ims_orig_shape[:2])
        for i in range(cut_len):
            map_im[gri[i], gci[i]] += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 4))
        ax1.imshow(map_im)
        ax1.set_xlabel("Scan X index")
        ax1.set_ylabel("Scan Y index")
        ax1.set_title("First %d combinations of %d images" % (cut_len, avg_nims))

    # find most common scan index within cut
    common_im_ind = np.bincount(comb_inds[:cut_len].flat).argmax()
    print(
        "Most common scan index in 1st %d combinations of %d images:"
        % (cut_len, avg_nims),
        np.unravel_index(common_im_ind, ims_orig_shape[:2]),
    )
    contains_common_im = (comb_inds[:cut_len] == common_im_ind).sum(1) > 0

    # unique image indices within cut with most popular image in common
    common_im_inds = np.unique(comb_inds[:cut_len][contains_common_im].flatten())
    print(
        "Number of unique images in these combinations sharing this index: %d"
        % (len(common_im_inds))
    )
    if plot:
        # plot unique points
        sel_im = np.zeros(ims_orig_shape[:2])
        sel_im.flat[common_im_inds] = 1
        ax2.imshow(sel_im)
        ax2.set_xlabel("Scan X index")
        # plt.ylabel('Scan Y index')
        plt.title(
            "Unique images in 1st %d combinations of %d images\nsharing most common image"
            % (cut_len, avg_nims)
        )

    # calculate means and stds with mask if specified
    if plot:
        f, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(5, 8))
        ax1, ax2, ax3, ax4, ax5, ax6 = axs.flatten()

        im_common = ims[common_im_inds]
        im_common_mean = im_common.mean(0)
        im_common_std = im_common.std(0)
        ax1.imshow(im_common_mean)
        ax2.imshow(im_common_std)
        ax1.set_title("Most common %d best" % (len(common_im_inds)))

        im_best = ims[comb_inds[0]]
        im_best_mean = im_best.mean(0)
        im_best_std = im_best.std(0)
        ax3.imshow(im_best_mean)
        ax4.imshow(im_best_std)
        ax3.set_title("Best combination of %d" % (avg_nims))

        im_worst = ims[comb_inds[-1]]
        im_worst_mean = im_worst.mean(0)
        im_worst_std = im_worst.std(0)
        ax5.imshow(im_worst_mean)
        ax6.imshow(im_worst_std)
        ax5.set_title("Worst combination of %d" % (avg_nims))
    print("")

    # return data (without masks)
    yxi_combos = np.unravel_index(comb_inds, ims_orig_shape[:2])
    yxi_common = np.unravel_index(common_im_inds, ims_orig_shape[:2])

    ims_common = ims_orig[common_im_inds]
    ims_common_mean = ims_common.mean(0)
    ims_common_std = ims_common.std(0)

    ims_best = ims_orig[comb_inds[0]]
    ims_best_mean = ims_best.mean(0)
    ims_best_std = ims_best.std(0)

    # reshape original, in case ascontiguousarray returns view
    ims_orig.shape = ims_orig_shape

    rtn = namedtuple("matching", ["yxi_combos", "yxi_common", "ims_common", "ims_best"])
    return rtn(yxi_combos, yxi_common, ims_common, ims_best)


def make_ref_im(
    image,
    edge_sigma,
    aperture=None,
    upscale=4,
    bin_opening=None,
    bin_closing=None,
    crop_pad=False,
    threshold=None,
    plot=True,
):
    """
    Generate a cleaned version of the image supplied for use as a reference.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    edge_sigma : float
        Edge width in pixels.
    aperture : None or 2-D array
        If not None, the data will be multiplied by the aperture mask.
    upscale : int
        Upscaling factor.
    bin_opening : None or int
        Circular element radius used for binary opening.
    bin_closing : None or int
        Circular element radius used for binary closing.
    crop_pad : bool
        If True and ``aperture`` is not None, the image is cropped before
        upscaling and padded in returned image for efficiency.
    threshold : scalar or None
        Image threshold. If None, Otsu's method is used. Otherwise, the scalar
        value is used.
    plot : bool
        If True, the images are plotted.

    Notes
    -----
    The sequence of operation is:
        apply aperture
        upscale
        threshold
        bin_opening
        bin_closing
        edge_sigma
        downscale
        scale magnitude

    Examples
    --------
    >>> from fpd.synthetic_data import disk_image
    >>> from merlintools.vendor.fpd import fpd_processing as fpdp

    # Generate synthetic image
    >>> image = disk_image(radius=32, intensity=64)

    # Get centre and edge, and make aperture
    >>> cyx, cr = fpdp.find_circ_centre(image, sigma=6, rmms=(2, int(image.shape[0]/2.0), 1), plot=False)
    >>> edge_sigma = fpdp.disc_edge_properties(image, sigma=2, cyx=cyx, r=cr, plot=False).sigma_wt_avg
    >>> aperture = fpdp.virtual_apertures(image.shape[-2:], cyx=cyx, rio=(0, cr+16), sigma=0, aaf=1)

    # Make reference image
    >>> ref_im = fpdp.make_ref_im(image, edge_sigma, aperture)

    """

    # float
    im = image.astype(float, copy=False)
    im_shape = image.shape

    # mask
    if aperture is not None:
        im = im * aperture
        if crop_pad:
            # crop and pad for efficiency
            ci, cf = np.where((aperture > 0.5).sum(0) > 0)[0][[0, -1]]
            ri, rf = np.where((aperture > 0.5).sum(0) > 0)[0][[0, -1]]
            im = im[ri : rf + 1, ci : cf + 1]

    # upscale and threshold
    ref_imu = sp.ndimage.interpolation.zoom(
        im, zoom=4, output=None, order=3, mode="constant", cval=0.0, prefilter=True
    )
    if threshold is None:
        thresh = threshold_otsu(ref_imu)
    else:
        thresh = float(threshold)
    processed = ref_imu >= thresh

    # binary opening / closing
    if bin_opening is not None:
        el = disk(bin_opening * upscale)
        processed = binary_opening(processed, el)
    if bin_closing is not None:
        el = disk(bin_closing * upscale)
        processed = binary_closing(processed, el)

    # smooth and downscale
    processed = sp.ndimage.filters.gaussian_filter(
        processed * 1.0, edge_sigma * upscale
    )
    processed = sp.ndimage.interpolation.zoom(
        processed,
        zoom=1.0 / upscale,
        output=None,
        order=3,
        mode="constant",
        cval=0.0,
        prefilter=True,
    )

    # scale mag
    mag_scale = np.percentile(im[processed > 0.5], 50)
    processed = processed * mag_scale

    if aperture is not None and crop_pad:
        im_pad = np.zeros_like(image, dtype=float)
        im_pad[ri : rf + 1, ci : cf + 1] = im
        im = im_pad

        im_pad = np.zeros_like(image, dtype=float)
        im_pad[ri : rf + 1, ci : cf + 1] = processed
        processed = im_pad

    # plot
    if plot:
        err = processed - im
        pct = 0.1
        vmin_max = np.percentile(err, [pct, 100 - pct])
        vmin, vmax = np.abs(vmin_max).max() * np.array([-1, 1])

        f, (ax1, ax2, ax3) = plt.subplots(
            1, 3, sharex=True, sharey=True, figsize=(9, 3)
        )
        ax1.imshow(im)
        ax2.imshow(processed)
        ax3.imshow(err, vmin=vmin, vmax=vmax, cmap="bwr")
        ax1.set_title("Original")
        ax2.set_title("Processed")
        ax3.set_title(
            "Processed : Original\n%0.3f - %0.3f" % (vmin_max[0], vmin_max[1])
        )

    return processed


def _parse_rebin(detY, detX, cropped_im_shape, rebin, rii, rif, cii, cif, print_stats):
    # TODO integrate better with crop

    rebinf = 1
    rebinning = rebin is not None and rebin != 1
    if rebinning:
        # change crop
        extra_pixels = (
            int(np.ceil(cropped_im_shape[0] / float(rebin)) * rebin)
            - cropped_im_shape[0]
        )
        ext_pix_pads = extra_pixels // 2

        # this is where the decision on if extra pixels can be added and where
        # they should go could be made
        if extra_pixels % 2:
            # odd
            ext_pix_pads = (-ext_pix_pads, ext_pix_pads + 1)
        else:
            # even
            ext_pix_pads = (-ext_pix_pads, ext_pix_pads)
        riic, rifc = rii + ext_pix_pads[0], rif + ext_pix_pads[1]
        ciic, cifc = cii + ext_pix_pads[0], cif + ext_pix_pads[1]
        if riic < 0 or rifc > detY - 1 or ciic < 0 or cifc > detX - 1:
            # change rebin
            f, fs = nearest_int_factor(cropped_im_shape[0], rebin)
            if rebin != f:
                if print_stats:
                    print("Image data cropped to:", cropped_im_shape)
                    print(
                        "Requested rebin (%d) changed to nearest value: %d. Possible values are:"
                        % (rebin, f),
                        fs,
                    )
                rebin = f
        else:
            rii, rif = riic, rifc
            cii, cif = ciic, cifc
            cropped_im_shape = (rif + 1 - rii, cif + 1 - cii)
            if print_stats:
                print("Image data cropped to:", cropped_im_shape)
        rebinf = rebin

    return cropped_im_shape, rebinf, rebinning, rii, rif, cii, cif


def _parse_crop_rebin(crop_r, detY, detX, cyx, aperture, rebin, print_stats):
    # TODO: crop and rebin could be better intergrated

    if crop_r and aperture:
        raise Exception("only one of 'crop_r' and 'aperture' may be specified")

    # determine crop
    if aperture is not None:
        # determine limits to index array for efficiency
        rii, rif = np.where(aperture.sum(axis=1) > 0)[0][[0, -1]]
        cii, cif = np.where(aperture.sum(axis=0) > 0)[0][[0, -1]]
    elif crop_r:
        crop_r = np.round([crop_r]).astype(int)[0]
        if cyx is None:
            cyx = [(detY - 1) / 2.0, (detX - 1) / 2.0]
        cy, cx = np.round(cyx).astype(int)

        crop_r_max = int(min(cx, detX - 1 - cx, cy, detY - 1 - cy))  # L R T B
        if crop_r > crop_r_max:
            if print_stats:
                print(
                    "WARNING: 'crop_r' (%d) is being set to max. value (%d)."
                    % (crop_r, crop_r_max)
                )
            crop_r = crop_r_max
        # indices
        rii, rif = (cy - crop_r, cy + crop_r - 1)
        cii, cif = (cx - crop_r, cx + crop_r - 1)
    else:
        rii, rif = 0, detY - 1
        cii, cif = 0, detX - 1
    cropped_im_shape = (rif + 1 - rii, cif + 1 - cii)

    rtn = _parse_rebin(
        detY, detX, cropped_im_shape, rebin, rii, rif, cii, cif, print_stats
    )
    return rtn


def _run(func, d, parallel=True, parallel_mode="thread", ncores=None):
    if parallel:
        if ncores is None:
            ncores = mp.cpu_count()

        parallel_modes = ["thread", "process"]
        if parallel_mode not in parallel_modes:
            print("'parallel_mode' must be one of: %s" % (str(parallel_modes)))

        with threadpool_limits(limits=1):
            if parallel_mode == "thread":
                from multiprocessing.pool import ThreadPool

                p = ThreadPool
            elif parallel_mode == "process":
                p = mp.Pool
            pool = p(processes=ncores)
            rslt = pool.map(func, d)
            pool.close()
    else:
        rslt = list(map(func, d))
    return rslt


def map_image_function(
    data,
    nr,
    nc,
    cyx=None,
    crop_r=None,
    func=None,
    params=None,
    mapped_params=None,
    rebin=None,
    parallel=True,
    ncores=None,
    parallel_mode="thread",
    print_stats=True,
    nrnc_are_chunks=False,
    progress_bar=True,
):
    """
    Map an arbitrary function over a multidimensional dataset.

    Parameters
    ----------
    data : array_like
        Mutidimensional data of shape (scanY, scanX, ..., detY, detX).
    nr : integer or None
        Number of rows to process at once (see Notes).
    nc : integer or None
        Number of columns to process at once (see Notes).
    cyx : length 2 iterable or None
        Centre of disk in pixels (cy, cx).
        If None, the centre is used.
    crop_r : scalar or None
        Radius of circle about `cyx` defining square crop limits used for
        cross-corrolation, in pixels.
        If None, the maximum square array about cyx is used.
    func : callable
        Function that operates (out-of-place) on an image: out = pre_func(im),
        where `im` is an ndarray of shape (detY, detX).
    params : None or dictionary
        If not None, a dictionary of parameters passed to the function.
    mapped_params : None or dictionary
        If not None, a dictionary of spatially resolved parameters passed to
        the function, and of shape (scanY, scanX, ...,).
    rebin : integer or None
        Rebinning factor for detector dimensions. None or 1 for none.
        If the value is incompatible with the cropped array shape, the
        nearest compatible value will be used instead.
        'cyx' and 'crop_r' are for the original image and need not be modified.
    parallel : bool
        If True, the calculations are processed in parallel.
    ncores : None or int
        Number of cores to use for parallel execution. If None, all cores
        are used.
    parallel_mode : str
        The mode to use for parallel processing.
        If 'thread' use multithreading.
        If 'process' use multiprocessing.
        Which is faster depends on the calculations performed.
    print_stats : bool
        If True, calculation progress is printed to stdout.
    nrnc_are_chunks : bool
        If True, `nr` and `nc` are interpreted as the number of chunks to
        process at once. If `data` is not chunked, `nr` and `nc` are used
        directly.
    progress_bar : bool
        If True, progress bars are printed.

    Returns
    -------
    rtn : ndarray
        The result of mapping the function over the dataset. If the output of
        the function is non-uniform, the dimensions are those of the nondet axes
        and the dtype is object. If the function output is uniform, the first
        axis is of the length of the function return, unless it is singular, in
        which case it is removed.

    Notes
    -----
    If `nr` or `nc` are None, the entire dimension is processed at once.
    For chunked data, setting `nrnc_are_chunks` to True, and `nr` and `nc`
    to a suitable values can improve performance.

    Specifying 'crop_r' (and appropriate cyx) can speed up calculation significantly.

    Examples
    --------
    Center of mass:
    >>> import scipy as sp
    >>> import numpy as np
    >>> from merlintools.vendor.fpd import fpd_processing as fpdp
    >>> from fpd.synthetic_data import disk_image, fpd_data_view

    >>> radius = 32
    >>> im = disk_image(intensity=1e3, radius=radius, size=256, upscale=8, dtype='u4')
    >>> data = fpd_data_view(im, (32,)*2, colours=0)
    >>> func = sp.ndimage.center_of_mass
    >>> com_y, com_x = fpdp.map_image_function(data, nr=9, nc=9, func=func)

    Non-uniform return:
    >>> def f(image):
    ...    l = np.random.randint(4)+1
    ...    return np.arange(l)
    >>> r = fpdp.map_image_function(data, nr=9, nc=9, func=f)

    Parameter passing:
    >>> def f(image, v):
    ...    return (image >= v).sum()
    >>> r = fpdp.map_image_function(data, nr=9, nc=9, func=f, params={'v' : 2})

    Mapped parameter passing:
    >>> mvals = np.arange(np.prod(data.shape[:-2])).reshape(data.shape[:-2])
    >>> def f(image, v, w):
    >>>     # we can operate on any of the inputs, but we simply return
    >>>     # the mapped parameter here as to demonstrate the feature.
    >>>     return w

    >>> r = fpdp.map_image_function(data, nr=9, nc=9, func=f, params={'v' : 2}, mapped_params={'w' : mvals})
    >>> np.all(r == mvals)

    Doing very little (when reading from file, this is a measure of access
    and decompression overhead):
    >>> def f(image):
    ...    return None
    >>> data_chunk = data[:16, :16]
    >>> r = fpdp.map_image_function(data_chunk, nr=None, nc=None, func=f)

    """

    if params is None:
        params = {}

    if nrnc_are_chunks:
        nr, nc = _condition_nrnc_if_chunked(data, nr, nc, print_stats)

    nondet = data.shape[:-2]
    nonscan = data.shape[2:]
    scanY, scanX = data.shape[:2]
    detY, detX = data.shape[-2:]

    r_if, c_if = _block_indices((scanY, scanX), (nr, nc))

    rtn = _parse_crop_rebin(
        crop_r=crop_r,
        detY=detY,
        detX=detX,
        cyx=cyx,
        aperture=None,
        rebin=rebin,
        print_stats=print_stats,
    )
    cropped_im_shape, rebinf, rebinning, rii, rif, cii, cif = rtn

    rebinned_im_shape = tuple([x // rebinf for x in cropped_im_shape])
    # print('Cropped shape: ', cropped_im_shape)
    # if rebinning:
    # print('Rebinned cropped shape: ', rebinned_im_shape)

    rd = np.empty(nondet, dtype=object)

    if print_stats:
        print("\nMapping image function")
        tqdm_file = sys.stderr
    else:
        tqdm_file = DummyFile()
    total_nims = np.prod(nondet)
    with tqdm(
        total=total_nims,
        file=tqdm_file,
        mininterval=0,
        leave=True,
        unit="images",
        disable=(not progress_bar),
    ) as pbar:
        for i, (ri, rf) in enumerate(r_if):
            for j, (ci, cf) in enumerate(c_if):
                # read selected data (into memory if hdf5)
                d = data[ri:rf, ci:cf, ..., rii : rif + 1, cii : cif + 1]
                d = np.ascontiguousarray(d)

                # rebin
                if rebinning:
                    ns = d.shape[:-2] + tuple([int(x / rebinf) for x in d.shape[-2:]])
                    d = rebinA(d, *ns)

                # flatten non-detector axes
                d_shape = d.shape
                d.shape = (np.prod(d_shape[:-2]),) + d_shape[-2:]

                # prepare functions
                if mapped_params is None:
                    partial_func = partial(func, **params)
                else:
                    # process mapped params
                    mapped_params_chk = {}
                    for k, v in mapped_params.items():
                        # select chunk of mapped params
                        v_chk = v[ri:rf, ci:cf]
                        v_chk = np.ascontiguousarray(v_chk)

                        # no rebinning since rebin here is only for detector axes

                        # flatten scan (+colour) axes
                        nd_to_flatten = len(d_shape[:-2])
                        v_chk_shape = v_chk.shape
                        v_chk.shape = (
                            np.prod(v_chk_shape[:nd_to_flatten]),
                        ) + v_chk_shape[nd_to_flatten:]

                        # add to dict
                        mapped_params_chk[k] = v_chk

                    # enumerate data for mapped param indexing
                    d = list(enumerate(d))

                    # wrapper function to loop over data and index, with mapped params chunk passed in
                    original_func = func

                    def func_wrap(index_d, params, mapped_params_chk):
                        i, di = index_d

                        # index mapped params chunk and add to the params of the original function
                        mapped_params_chk_i = dict(
                            [(k, v[i]) for k, v in mapped_params_chk.items()]
                        )
                        params.update(mapped_params_chk_i)
                        return func(di, **params)

                    # partial
                    partial_func = partial(
                        func_wrap, params=params, mapped_params_chk=mapped_params_chk
                    )

                rslt = _run(
                    partial_func,
                    d,
                    parallel=parallel,
                    parallel_mode=parallel_mode,
                    ncores=ncores,
                )
                t = np.empty(len(rslt), dtype=object)
                t[:] = rslt
                rd[ri:rf, ci:cf].flat = t

                pbar.update(np.prod(d_shape[:-2]))
    if print_stats:
        print("")
        sys.stdout.flush()

    # convert dtype from object to more appropriate type if possible
    try:
        rdf = rd.ravel()
        rdfa = np.vstack(rdf).reshape(rd.shape + (-1,))
        rtn = np.rollaxis(rdfa, -1, 0)
    except ValueError:
        rtn = rd

    # remove return axis if singular
    if rtn.ndim > rd.ndim:
        if rtn.shape[0] == 1:
            rtn = rtn[0]

    return rtn


def rotate_vector(yx_array, theta, axis=0):
    """
    Rotate a vector by an angle.

    Parameters
    ----------
    yx_array : ndarray
        Array or iterable of arrays of vectors, one dimension of which has [y,x].
    theta : scalar
        Rotation angle in degrees (anticlockwise).
    axis : scalar
        Axis of yx_array with [y, x] values.

    """

    yx_array = np.array(yx_array, copy=False)
    single_yx = False
    if yx_array.ndim == 1:
        yx_array = yx_array[..., None]
        single_yx = True

    yx = np.rollaxis(yx_array, axis, 0)
    yx_shape = yx.shape
    yx = yx.reshape((yx.shape[0], np.prod(yx.shape[1:])))

    t = np.deg2rad(theta)
    rot_mat = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]).T
    # transpose for anticlockwise with yx in 1st axis

    yx_rot = rot_mat.dot(yx)
    yx_rot.shape = yx_shape
    ims_rot = np.rollaxis(yx_rot, 0, axis + 1)

    if single_yx:
        ims_rot = ims_rot[..., 0]
    return ims_rot


class VirtualAnnularImages(object):
    def __init__(
        self,
        data,
        nr=16,
        nc=16,
        cyx=None,
        dyx=None,
        dyx_mode="pixel",
        parallel=True,
        ncores=None,
        parallel_mode="thread",
        nrnc_are_chunks=False,
        print_stats=True,
        mask=None,
        spf=1,
        progress_bar=True,
    ):
        """
        Fast virtual annular aperture image class using cumulative sums to
        calculate all data only once, with interactive plotting.

        To do this it uses: `fpd.fpd_processing.radial_profile` and
        `fpd.fpd_processing.map_image_function`. See those functions for details
        not documented below.

        This method is very fast and so useful for exploring data, but is not as
        flexible as using `fpd.fpd_processing.virtual_images` directly.

        The pixel-level accuracy can be improved at the expense of computation time
        by increasing the subpixel evaluation of the radial distribution through the
        `spf` parameter.

        Parameters
        ----------
        data : ndarray or string or dict
            If ndarray, `data` is the data to be processed, as defined in the
            fpd.fpd_processing.map_image_function. If a string, it should be the
            filename of a npz file with the parameters saved from the `save_data`
            method. If a dictionary, it must contain the same parameters.
        cyx : length 2 iterable or None
            The centre y and x coordinates of the direct beam in pixels.
            This value must be specified unless `data` is an object to be loaded.
        dyx : ndarray or None
            If not None, (y, x) vector of shifts to be applied to the data, of
            shape (2, scanY, scanX). See Notes.
        dyx_mode : str
            If 'pixel', `dyx` is converted to integer type for pixel resolution.
            If 'linear', `dyx` is used with sub-pixel resolution in linear interpolation.
            If 'fourier', `dyx` is used with sub-pixel resolution in Fourier shifting.
            The 'pixel' mode is very fast. The 'linear' mode is slowest, but avoids
            possible ringing in the intermediate speed 'fourier' mode.
        progress_bar : bool
            If True, progress bars are printed.

        Notes
        -----
        The data shift `dyx` may be determined by a number of methods, including:
        - fpd.fpd_processing.center_of_mass
        - fpd.fpd_processing.phase_correlation
        while remembering to negate and centre the returned positions. For example,
        for positions comyx, values for dyx may be obtained from:

        >>> import numpy as np
        >>> shifts = -(comyx - np.percentile(comyx, 50, (-2, -1))[..., None, None])

        If the data is to be aligned to a particular scan point, then it should be
        subtracted rather than the percentile in the example above.

        See Also
        --------
        virtual_images

        """

        self.r1 = None
        self.r2 = None
        self.virtual_image = None

        if isinstance(data, str):
            # add data filename attribute and load data as dict
            self._source_filename = data
            data = dict(np.load(data))
        if isinstance(data, dict):
            # add attributes
            for k, v in data.items():
                setattr(self, k, v)
        else:
            # process data to generate attributes
            if cyx is None:
                raise TypeError("cyx must be specified")
            self.data_shape = np.array(data.shape)
            self.cyx = np.array(cyx)
            self._calc_rdf(
                data,
                nr,
                nc,
                cyx,
                dyx,
                dyx_mode,
                mask,
                spf,
                parallel,
                ncores,
                parallel_mode,
                nrnc_are_chunks,
                print_stats,
                progress_bar,
            )

        # cummulative sums
        self.rms_cs = np.cumsum(
            self.rms * 2 * np.pi * self.r_pix[:, None, None], axis=0
        )
        self.a_cs = np.cumsum(2 * np.pi * (self.r_pix), axis=0)

    def save_data(self, filename=None):
        """
        Save the calculated parameters to file for later reloading through the `data`
        initialisation parameter.

        Parameters
        ----------
        filename : None or string
            File name to save data under. If None a date stamped filename is generated.
            If the file name does not end in '.npz', it is automatically added.

        Returns
        -------
        full_fn : string
            The absolute path of the output file.

        """

        version = 1

        if filename is None:
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = "VirtualAnnularImages_" + now
        if filename.endswith(".npz") is False:
            filename = filename + ".npz"

        np.savez(
            filename,
            data_shape=self.data_shape,
            cyx=self.cyx,
            r_pix=self.r_pix,
            rms=self.rms,
            version=version,
        )
        print("Data saved to: %s" % (filename))
        full_fn = os.path.abspath(filename)
        return full_fn

    def _calc_rdf(
        self,
        data,
        nr,
        nc,
        cyx,
        dyx,
        dyx_mode,
        mask,
        spf,
        parallel,
        ncores,
        parallel_mode,
        nrnc_are_chunks,
        print_stats,
        progress_bar,
    ):
        if dyx is None:
            func = radial_profile
            mapped_params = None
        else:
            dyx, dyx_mode = _condition_dyx(dyx, dyx_mode)

            from fpd.synthetic_data import shift_im

            def wrapped_radial_profile(image, dyx, **kwdargs):
                shifted_image = shift_im(image, dyx, method="pixel")
                rtn = radial_profile(shifted_image, **kwdargs)
                return rtn

            func = wrapped_radial_profile
            mapped_params = {"dyx": np.moveaxis(dyx, 0, -1)}

        rtn = map_image_function(
            data,
            nr,
            nc,
            cyx=cyx,
            crop_r=None,
            func=func,
            params={"cyx": cyx, "mask": mask, "spf": spf},
            mapped_params=mapped_params,
            rebin=None,
            parallel=parallel,
            ncores=ncores,
            parallel_mode=parallel_mode,
            nrnc_are_chunks=nrnc_are_chunks,
            print_stats=print_stats,
            progress_bar=progress_bar,
        )

        r_pix, rms = rtn.reshape((2, -1) + rtn.shape[1:])

        # 1-D
        self.r_pix = np.squeeze(r_pix[:, 0, 0])
        # rdf, scanY, scanX
        if rms.ndim == 4:
            # colour
            rms = rms[..., 0]

        self.rms = rms
        del rtn

    def annular_slice(self, r1, r2):
        """
        Calculate an annular virtual image.

        Parameters
        ----------
        r1 : scalar
            Inner radius of aperture in pixels.
        r2 : scalar
            Inner radius of aperture in pixels.

        Returns
        -------
        virtual_image : ndarray
            The virtual image.

        """
        self.r1 = r1
        self.r2 = r2

        r1i = np.argmax(self.r_pix >= r1)
        r2i = np.argmin(self.r_pix <= r2) - 1
        v = self.rms_cs[r2i] - self.rms_cs[r1i]
        va = self.a_cs[r2i] - self.a_cs[r1i]
        n = np.pi * (r2**2 - r1**2) / va
        self.virtual_image = v * n

        return self.virtual_image

    def plot(
        self,
        r1=None,
        r2=None,
        nav_im=None,
        norm="log",
        scroll_step=1,
        alpha=0.3,
        cmap=None,
        pct=0.1,
        mradpp=None,
    ):
        """
        Interactive plotting of the virtual aperture images.

        The sliders control the parameters and may be clicked, dragged or scrolled.
        Clicking on inner (r1) and outer (r2) slider labels sets the radii values
        to the minimum and maximum, respectively.

        Parameters
        ----------
        r1 : scalar
            Inner radius of aperture in pixels.
        r2 : scalar
            Inner radius of aperture in pixels.
        nav_im : None or ndarray
            Image used for the navigation plot. If None, a blank image is used.
            See Notes.
        norm : None or string:
            If not None and norm='log', a logarithmic cmap normalisation is used.
        scroll_step : int
            Step in pixels used for each scroll event.
        alpha : float
            Alpha for aperture plot in [0, 1].
        cmap : None or a matplotlib colormap
            If not None, the colormap used for both plots.
        pct : scalar
            Slice image percentile in [0, 50).
        mradpp : None or scalar
            mrad per pixel.

        Notes
        -----
        For data that is aligned using `dyx`, `nav_im` may be generated by:
        >>> nav_im = fpd.fpd_processing.sum_dif(data, nr, nc, dyx=dyx, dyx_mode=dyx_mode)

        """

        from matplotlib.widgets import Slider

        self._scroll_step = max([1, int(scroll_step)])
        self._pct = pct

        if norm is not None:
            if norm.lower() == "log":
                from matplotlib.colors import LogNorm

                norm = LogNorm()

        # condition rs
        if r1 is not None:
            self.r1 = r1
        else:
            if self.r1 is None:
                self.r1 = 0
        if r2 is not None:
            self.r2 = r2
        else:
            if self.r2 is None:
                self.r2 = int((self.data_shape[-2:] / 4).mean())
        self.rc = (self.r2 + self.r1) / 2.0

        if nav_im is None:
            nav_im = np.zeros(self.data_shape[-2:])

        # calculate data
        virtual_image = self.annular_slice(self.r1, self.r2)

        # prepare plots
        if mradpp is None:
            self._f_nav, (ax_nav, ax_cntrst) = plt.subplots(1, 2, figsize=(8.4, 4.8))
        else:
            # add 2nd x-axis
            # https://matplotlib.org/examples/axes_grid/parasite_simple2.html
            from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost
            import matplotlib.transforms as mtransforms

            self._f_nav = plt.figure(figsize=(8.4, 4.8))
            ax_nav = SubplotHost(self._f_nav, 1, 2, 1)
            ax_cntrst = SubplotHost(self._f_nav, 1, 2, 2)

            aux_trans = mtransforms.Affine2D().scale(1.0 / mradpp, 1.0)
            ax_mrad = ax_cntrst.twin(aux_trans)
            ax_mrad.set_viewlim_mode("transform")

            self._f_nav.add_subplot(ax_nav)
            self._f_nav.add_subplot(ax_cntrst)

            ax_mrad.axis["top"].set_label("mrad")
            ax_mrad.axis["top"].label.set_visible(True)
            ax_mrad.axis["right"].major_ticklabels.set_visible(False)

        self._f_nav.subplots_adjust(bottom=0.3, wspace=0.3)
        axr1 = plt.axes([0.10, 0.05, 0.80, 0.03])
        axr2 = plt.axes([0.10, 0.10, 0.80, 0.03])
        axr3 = plt.axes([0.10, 0.15, 0.80, 0.03])

        val_max = self.r_pix.max()
        try:
            self._sr1 = Slider(
                axr1, "r1", 0, val_max - 1, valinit=self.r1, valfmt="%0.0f", valstep=1
            )
            self._sr2 = Slider(
                axr2, "r2", 1, val_max, valinit=self.r2, valfmt="%0.0f", valstep=1
            )
        except AttributeError:
            self._sr1 = Slider(
                axr1, "r1", 0, val_max - 1, valinit=self.r1, valfmt="%0.0f"
            )
            self._sr2 = Slider(axr2, "r2", 1, val_max, valinit=self.r2, valfmt="%0.0f")
        self._sr3 = Slider(axr3, "rc", 1, val_max, valinit=self.rc, valfmt="%0.1f")

        # these don't seem to work
        # self._sr1.slider_max = self._sr2
        # self._sr2.slider_min = self._sr1

        self._sr1.on_changed(self._update_r_from_slider)
        self._sr2.on_changed(self._update_r_from_slider)
        self._sr3.on_changed(self._update_rc_from_slider)

        ax_nav.imshow(nav_im, norm=norm, cmap=cmap)
        ax_nav.set_xlabel("Detector X (pixels)")
        ax_nav.set_ylabel("Detector Y (pixels)")

        # line plot
        r_cntrst_max = int(np.abs(self.data_shape[-2:] - self.cyx).max())
        dw = 1
        rs = np.arange(dw, r_cntrst_max)

        r1, r2 = self.r1, self.r2
        sls = np.array([self.annular_slice(r - dw, r) for r in rs])
        self.r1, self.r2 = r1, r2

        self._contrast_y = np.std(sls, (1, 2)) ** 2 / np.mean(sls, (1, 2))
        self._contrast_x = rs - dw / 2.0
        ax_cntrst.plot(self._contrast_x, self._contrast_y)
        ax_cntrst.minorticks_on()
        ax_cntrst.set_xlabel("Radius (pixels)")
        ax_cntrst.set_ylabel("Contrast (std^2/mean)")
        self._span = ax_cntrst.axvspan(self.r1, self.r2, color=[1, 0, 0, 0.1], ec="r")

        # wedges
        fc = [0, 0, 0, alpha]
        ec = "r"
        from matplotlib.patches import Wedge

        self._rmax = val_max + 1
        self._w2 = Wedge(
            self.cyx[::-1], self._rmax, 0, 360, width=self._rmax - self.r2, fc=fc, ec=ec
        )
        self._w1 = Wedge(self.cyx[::-1], self.r1, 0, 360, width=self.r1, fc=fc, ec=ec)
        ax_nav.add_artist(self._w2)
        ax_nav.add_artist(self._w1)

        self._f_im, ax_im = plt.subplots(1, 1)
        vmin, vmax = np.percentile(virtual_image, [self._pct, 100 - self._pct])
        self._vim = ax_im.imshow(virtual_image, cmap=cmap, vmin=vmin, vmax=vmax)
        self._cb = plt.colorbar(self._vim)
        self._cb.set_label("Counts")
        ax_im.set_xlabel("Scan X (pixels)")
        ax_im.set_ylabel("Scan Y (pixels)")

        cid = self._f_nav.canvas.mpl_connect("scroll_event", self._onscroll)

        self._sr1.label.set_picker(True)
        self._sr2.label.set_picker(True)
        cid_pick = self._f_nav.canvas.mpl_connect("pick_event", self._onpick)

    def _onpick(self, event):
        if event.artist == self._sr1.label:
            self.r1 = self._sr1.valmin
            self._update_plot_r_from_val()
        if event.artist == self._sr2.label:
            self.r2 = self._sr2.valmax
            self._update_plot_r_from_val()

    def _update_r_from_slider(self, val):
        self.r1 = int(self._sr1.val)
        self.r2 = int(self._sr2.val)
        self.rc = (self.r2 + self.r1) / 2.0

        self._sr3.eventson = False
        self._sr3.set_val(self.rc)
        self._sr3.eventson = True

        _ = self.annular_slice(self.r1, self.r2)

        self._w1.set_radius(self.r1)
        self._w1.set_width(self.r1)
        self._w2.set_width(self._rmax - self.r2)

        xy = self._span.xy
        xy[:, 0] = [self.r1, self.r1, self.r2, self.r2, self.r1]
        self._span.set_xy(xy)

        self._vim.set_data(self.virtual_image)
        # vmin, vmax = self.virtual_image.min(), self.virtual_image.max()
        vmin, vmax = np.percentile(self.virtual_image, [self._pct, 100 - self._pct])
        self._vim.set_clim(vmin, vmax)

        self._f_im.canvas.draw_idle()
        self._f_nav.canvas.draw_idle()

    def _update_rc_from_slider(self, val):
        rc_prev = (self.r2 + self.r1) / 2.0

        drc = self._sr3.val - rc_prev

        self._sr1.eventson = False
        self._sr1.set_val(self._sr1.val + drc)
        self._sr1.eventson = True

        self._sr2.eventson = False
        self._sr2.set_val(self._sr2.val + drc)
        self._sr2.eventson = True

        self._update_r_from_slider(None)

    def _update_plot_r_from_val(self):
        self._sr1.eventson = False
        self._sr1.set_val(self.r1)
        self._sr1.eventson = True

        self._sr2.eventson = False
        self._sr2.set_val(self.r2)
        self._sr2.eventson = True

        self._update_r_from_slider(None)

    def _onscroll(self, event):
        if event.inaxes not in [self._sr1.ax, self._sr2.ax, self._sr3.ax]:
            return
        if event.button == "up":
            dr = self._scroll_step
        else:
            dr = -self._scroll_step

        if event.inaxes == self._sr1.ax:
            self.r1 += dr
        elif event.inaxes == self._sr2.ax:
            self.r2 += dr
        else:
            self.r1 += dr
            self.r2 += dr
        self._update_plot_r_from_val()
