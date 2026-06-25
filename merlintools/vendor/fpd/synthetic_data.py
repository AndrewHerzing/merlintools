
import matplotlib.pyplot as plt
import skimage
from skimage import morphology
from skimage import transform

import scipy as sp
import numpy as np

from scipy.ndimage import fourier_shift
import multiprocessing as mp
from functools import partial, reduce
from threadpoolctl import threadpool_limits

from scipy.ndimage import shift

from numbers import Number

from .fpd_processing import rebinA

plt.ion()


#--------------------------------------------------
def disk_image(intensity=None, dose=None, radius=32, sigma=1.0, size=256, upscale=8,
               noise=False, dtype='u2', truncate=4.0, ds_method='interp'):
    '''
    Generate disk image.
    
    Parameters
    ----------
    intensity : scalar
        Counts per pixel.
    dose : scalar
        Total counts.
    radius : scalar
        Radius in pixels.
    upscale : integer
        Up-scaling factor used to reduce anti-aliasing.
    sigma : scalar
        Sigma of Gaussian smoothing.
    size : integer
        Square image edge length.
    noise : bool
        If True, disk data has Poissonian noise.
    dtype : numpy dtype
        Any valid numpy dtype, e.g. 'u2', np.float, float, etc.
    truncate : scalar
        Number of sigma to which Gaussians are calculated.
    ds_method : str
        String controlling the downsampling method. Methods are:
        'rebin'  for rebinning the data.
        'interp' for interpolation.
    
    Returns
    -------
    disk : ndarray
        2-D numpy array of specified `dtype`.
    
    Notes
    -----
    One and only one of `intensity` and `dose` must be specified.
    
    `dose` will set correct counts for all values of `sigma`.
    
    Gaussian convolution uses sp.ndimage.filters.gaussian_filter
    which is two 1-D convolutions, and so is poor for large sigma.
    
    Examples
    --------
    Create disk with noise:
    
    >>> from fpd import synthetic_data
    >>> import matplotlib.pyplot as plt
    >>> plt.ion()
    
    >>> disk = synthetic_data.disk_image(intensity=64, noise=True)
    >>> f = plt.figure() 
    >>> im = plt.imshow(disk, interpolation='nearest')
    >>> plt.set_cmap('gray')

    '''
    
    assert type(upscale) == int
    
    ds_methods = ['rebin', 'interp']
    ds_method = ds_method.lower()
    if ds_method not in ds_methods:
        erm = "'ds_method' must be one of: " + ', '.join(ds_methods)
        raise NotImplementedError(erm)
    
    
    if (intensity == None and dose == None) or (intensity != None and dose != None):
        raise Exception("One of `intensity` or `dose` must be specified.")
    if intensity != None:
        pass
    elif dose != None:
        intensity = dose / (np.pi * radius**2)
    
    # generate up-scaled disk
    disk = skimage.morphology.disk(radius*upscale, dtype='float')
    
    # pad for blur
    pw = [int(x) for x in (np.ceil(sigma*4)*upscale,)*2]
    disk = np.pad(disk, pad_width=pw, mode='constant', constant_values=0)
    
    # blur
    if sigma != 0:
        disk = sp.ndimage.filters.gaussian_filter(disk, 
                                                  sigma=sigma*upscale, 
                                                  order=0, 
                                                  mode='constant', 
                                                  cval=0.0, 
                                                  truncate=truncate)
    
    if upscale !=1:
        if ds_method == 'rebin':
            # pad image so it can be rebinned reliably
            ns = np.array([np.ceil(t/float(upscale))*upscale for t in disk.shape], dtype=int)
            pyx = ns - np.array(disk.shape)
            pyx1 = np.fix(pyx / 2.0).astype(int)
            pyx2 = pyx - pyx1
            pw = [(pyx1[0], pyx2[0]), (pyx1[1], pyx2[1])]
            disk = np.pad(disk, pad_width=pw, mode='constant', constant_values=0)
            disk = rebinA(disk, int(disk.shape[0]/upscale), int(disk.shape[1]/upscale)) / float(upscale**2)
        if ds_method == 'interp':
            # Bi-cubic down scale
            kwd = {'order': 3, 
                'mode': 'constant', 
                'cval': 0, 
                'clip': True, 
                'preserve_range': False,
                'multichannel' : False,
                'anti_aliasing': True}
            try:
                disk = skimage.transform.rescale(disk, 1.0/upscale, **kwd)
            except TypeError:
                try:
                    _ = kwd.pop('multichannel')
                    disk = skimage.transform.rescale(disk, 1.0/upscale, **kwd)
                except TypeError:
                    _ = kwd.pop('anti_aliasing')
                    disk = skimage.transform.rescale(disk, 1.0/upscale, **kwd)
    
    
    # pad disk to desired shape
    pad_pre = [np.ceil((size-x)/2.0) for x in disk.shape]
    pad_post = [size-x-y for x, y in zip(disk.shape, pad_pre)]
    pw = list(zip(pad_pre, pad_post))
    pw = np.asarray(pw, dtype=int)
    disk = np.pad(disk, pad_width=pw, mode='constant', constant_values=0)
    assert disk.shape == (size,)*2
    
    # set intensity
    disk *= intensity
    if dose == None:
        pass
    elif dose != None:
        disk_sum = disk.sum()
        scale = float(dose) / disk_sum
        disk *= scale
    
    # poissonian noise
    if noise:
        disk = np.random.poisson(disk)
    
    # convert dtype
    if np.issubdtype(dtype, np.integer):
        dtype_max = np.iinfo(dtype).max
    elif np.issubdtype(dtype, np.float):
        dtype_max = np.finfo(dtype).max
    assert(disk.max() <= dtype_max)
    disk = disk.astype(dtype, copy=False)
    return disk


#--------------------------------------------------
def poisson_noise(ims, samples):
    '''
    Returns `samples` of `ims` with Poissonian noise.
    
    Parameters
    ----------
    ims : ndarray
        Images from which noisy images are made.
    samples : int
        Number of samples of each image.
    
    Returns
    -------
    noisy_ims : ndarray
        Noisy images of shape (samples,) + ims.shape.
    
    Examples
    --------
    Create 3 images with Poissonian noise.
    
    >>> from fpd import synthetic_data
    >>> import numpy as np
    >>> ims = np.ones((4,5))
    >>> noisy_ims = synthetic_data.poisson_noise(ims, 3)
    >>> print(noisy_ims.shape)
    (3, 4, 5)
    
    '''
    
    size = (samples,) + ims.shape
    noisy_ims = np.random.poisson(ims, size)
    return noisy_ims
    

#--------------------------------------------------
def shift_array(scan_len=32, shift_min=-8.0, shift_max=8.0, shift_type=0):
    '''
    Generate 2-D shift arrays of different texture.
    
    Parameters
    ----------
    scan_len : integer
        Square edge scan length in pixels.
    shift_min : scalar
        Minimum shift in pixels.
    shift_max : scalar
        Maximum shift in pixels.
    shift_type : integer
        Type of shift array.
        If 0, a slope array is returned, starting from minimum at top
        left and going to maximum at bottom right.
        If 1, white noise is used, centred on mean of 'shift_min` and
        `shift_max`.
        If 2, polar shifts are used, with a magnitude of `shift_max`,
        and angle range of [0, pi/2).
        
    Returns
    -------
    shiftyy, shiftxx : tuple of ndarrays
        2-D numpy arrays of y and x shifts.
    
    Examples
    --------
    Create slope shift profiles and plot them.
    
    >>> from fpd import synthetic_data
    >>> import matplotlib.pylab as plt
    >>> import numpy as np
    
    >>> shiftyy, shiftxx = synthetic_data.shift_array(shift_type=0)
    >>> shift_mag = np.hypot(shiftyy, shiftxx)
    >>> shift_deg = np.rad2deg(np.arctan2(shiftyy, shiftxx))

    >>> f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8,8))
    >>> im = ax1.matshow(shiftyy, cmap='gray')
    >>> im = ax2.matshow(shiftxx, cmap='gray')
    >>> im = ax3.matshow(shift_mag, cmap='gray')
    >>> im = ax4.matshow(shift_deg, cmap='gray')
    
    '''
    
    if shift_type == 0:
        #slope
        #slope_shift_max = shift_max/np.sqrt(2)      # for same in corner
        shifty = np.linspace(shift_min, shift_max, scan_len)
        shiftx = shifty
        shiftyy, shiftxx = np.meshgrid(shifty, shiftx, indexing='ij')
    elif shift_type == 1:
        # white noise
        scan_im_shape = (scan_len,)*2
        shiftyy = (np.random.random_sample(scan_im_shape)*(shift_max-shift_min)
                   -(shift_max+shift_min)/2.0)
        shiftxx = (np.random.random_sample(scan_im_shape)*(shift_max-shift_min)
                   -(shift_max+shift_min)/2.0)
    elif shift_type == 2:
        # r-theta
        r = np.linspace(0, shift_max, scan_len)
        t = np.linspace(0, np.pi/2.0, scan_len, endpoint=False)
        rr, tt = np.meshgrid(r, t, indexing='ij')
        shiftyy = rr*np.cos(tt)
        shiftxx = rr*np.sin(tt)
        #plt.matshow(rr, cmap='gray')
        #plt.matshow(tt, cmap='gray')
    
    return(shiftyy, shiftxx)


def _fill_shifted_image(im, dyx, fill_value):
    '''
    Fill missing pixels of shifted image in-place.
    
    dyx can be a single set of value or multiple values ([n,] y, x).
    
    Always only 1 image.
    
    '''
    
    if dyx is not None:
        dyx = np.asarray(dyx)
        if dyx.ndim > 1:
            # multiple dyx values 
            dyx_min = dyx.min((-2, -1))
            dyx_max = dyx.max((-2, -1))
        else:
            dyx_min = dyx_max = dyx
        dyx_min = np.floor(dyx_min).astype(int)
        dyx_max = np.ceil(dyx_max).astype(int)
        
        dy2, dx2 = dyx_min
        dy1, dx1 = dyx_max
        
        # handle -ve / +ve here (coerce to 0 or axis size)
        dx1 = max(dx1, 0)
        dy1 = max(dy1, 0)
        if dx2 >= 0:
            dx2 = im.shape[1]
        if dy2 >= 0:
            dy2 = im.shape[0]
        
        im[..., :dy1, :] = fill_value
        im[..., dy2:, :] = fill_value
        im[..., :dx1] = fill_value
        im[..., dx2:] = fill_value
    return None


def shift_im(im, dyx, noise=False, method='linear', fill_value=0):
    '''
    Shift image `im` by amount in `dyx`, a tuple of (dy, dx).
    If `noise` is true, data has Poisson noise.
    
    Parameters
    ----------
    im : 2-d array
        Image to be shifted
    dyx : length 2 iterrable
        Shift vector, in direction of axis index.
    noise : bool
        If True, shifted image has Poissonian noise.
    method : string
        Method for image shifting. One of ['linear', 'fourier', 'pixel'].
        If 'linear', a bi-linear method is used.
        If 'fourier', the image is shifted cyclically.
        If 'pixel', the data is shifted with pixel resolution.
        In all cases, `fill_value` is used to replace extrapolated points.
    fill_value : scalar or None
        The value to use for points outside of the interpolation domain.
        If None and `method='linear'`, values outside the domain are
        extrapolated. Otherwise, zero is used.
    
    Returns
    -------
    im_new : 2-d array
        Shifted image.
    
    '''
    
    methods = ['linear', 'fourier', 'pixel']
    method = method.lower()
    if method not in methods:
        raise Exception("'method' (%s) must be one of: " %(method), methods)
    
    if method != 'linear' and fill_value is None:
        fill_value = 0
    
    # add processing of multiple dyxs and parallel processing?
    if method == 'fourier':
        im_new = fourier_shift(np.fft.fftn(im), dyx)
        im_new = np.abs(np.fft.ifftn(im_new))
    elif  method == 'linear':
        im_new = shift(im, shift=dyx, order=1)
    elif method == 'pixel':
        dyx = np.round(dyx, 0).astype(int)
        im_new = np.roll(im, dyx, axis=(-2, -1))
    
    _fill_shifted_image(im_new, dyx, fill_value)
    
    if noise:
        im_new = np.random.poisson(im_new).astype('u2', copy=False)
    
    return im_new


def array_image(image, yxg, amps=None, method='linear', fill_value=0):
    '''
    Create an image by summing `image` shifted by values in `yxg`.
    
    Parameters
    ----------
    image : ndarray
        2-D image to be arrayed.
    yxg : ndarray
        Shift coordinated of shape N x (yi, xi).
    amps : None, scalar, or ndarray
        If not None, the value(s) by which the values in `image` are
        scaled. If a scalar, the same scaling is used for all `yxg`
        points. Otherwise, an array of the same length as `yxg` with
        a separate intensity for each `yxg` point.
    method : string
        Method of shifting images. See fpd.synthetic_data.shift_im
        for details.
    fill_value : scalar or None
        The value to use for points outside of the interpolation
        domain. See fpd.synthetic_data.shift_im for details.
    
    Return
    ------
    im : ndarray
        The composite image.
    
    Examples
    --------
    import matplotlib.pylab as plt
    plt.ion()
    import numpy as np
    import fpd
    
    im_shape = (256,)*2
    cyx = (np.array(im_shape) -1) / 2
    d0 = fpd.synthetic_data.disk_image(intensity=100, radius=10, size=im_shape[0], sigma=0.5)
    yxg = fpd.tem_tools.synthetic_lattice(cyx=cyx, ab=(50,)*2, angles=(0, np.pi/2), shape=im_shape, plot=True)
    yxg -= cyx

    im = im = fpd.synthetic_data.array_image(d0, yxg)
    plt.matshow(im)
    plt.colorbar()
    
    See also
    --------
    fpd.synthetic_data.disk_image, fpd.tem_tools.synthetic_lattice
    
    '''
    
    image_i = image
    if amps is not None:
        if isinstance(amps, Number):
            image_i = image_i * amps
            amps = None
        else:
            amps = np.array(amps, dtype=np.float)
            assert len(amps) == len(yxg)
    
    for i, yxgi in enumerate(yxg):
        if amps is not None:
            image_i = image * amps[i]
        imi = shift_im(image_i, dyx=yxgi, method=method, fill_value=fill_value)
        if i == 0:
            im = imi.copy()
        else:
            im += imi
    return im


#--------------------------------------------------
def shift_images(shifts, image, noise=False, dtype=None, parallel=True, ncores=None,
                 parallel_mode='thread', origin='top', method='linear', fill_value=0):
    '''
    Generate array of `image` shifted by `shifts` with sub-pixel
    precision and, optionally, with Poissonian noise.
    
    Parameters
    ----------
    shifts : array_like
        Shift y, shift x in pixels.
    image : array_like
        Image to be shifted.
    noise : bool
        If True, returned data has Poissonian noise.
    dtype : numpy dtype
        If not None, `dtype` determines dtype of returned array.
        If None, dtype matches that of image.
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
    origin : str
        Controls y-origin of returned data. If origin='top', pythonic indexing 
        is used. If origin='bottom', increasing y is up.
    method : string
        Method for image shifting. One of ['linear', 'fourier', 'pixel'].
        If 'linear', a bi-linear method is used.
        If 'fourier', the image is shifted cyclically.
        If 'pixel', the data is shifted with pixel resolution.
        In all cases, `fill_value` is used to replace extrapolated points.
    fill_value : scalar or None
        The value to use for points outside of the interpolation domain.
        If None and `method='linear'`, values outside the domain are
        extrapolated. Otherwise, zero is used.
    
    Returns
    -------
    shifted_ims : ndarray
        Array with first n dimensions those of `shifts`, and last two those
        of `image`.
    
    Examples
    --------
    Generate disk images, a shift array, and shift the images by the shift array:
    
    >>> import matplotlib.pylab as plt
    >>> plt.ion()
    >>> from fpd import synthetic_data

    >>> sa = synthetic_data.shift_array(scan_len=8, shift_min=-16.0, shift_max=16.0)
    >>> disk_im = synthetic_data.disk_image(intensity=64)
    >>> sim = synthetic_data.shift_images(sa, disk_im)

    >>> f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(6, 3))
    >>> im = ax1.matshow(sa[0], cmap='gray')
    >>> im = ax2.matshow(sa[1], cmap='gray')

    >>> f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(8, 3))
    >>> im = ax1.matshow(disk_im, cmap='gray')
    >>> im = ax2.matshow(sim[0,0], cmap='gray')
    >>> im = ax3.matshow(sim[-1,-1], cmap='gray')
    
    '''
    
    from .fpd_processing import _run
    
    image_dtype = image.dtype    
    
    shiftyy, shiftxx = shifts
    # default origin implementation is top
    if origin.lower() == 'bottom':
        shiftyy = -shiftyy
    dyx_flat = np.column_stack([shiftyy.flatten(), shiftxx.flatten()])
    
    partial_f = partial(shift_im, image, noise=noise,
                        method=method, fill_value=fill_value)
    
    rslt = _run(partial_f, dyx_flat, parallel=parallel, parallel_mode=parallel_mode, ncores=ncores)
        
    shifted_ims = np.asarray(rslt)
    shifted_ims.shape = shiftyy.shape + image.shape # (scanY, scanX, detY, detX)
    
    # convert float to desired dtype
    target_dtype = image_dtype
    if dtype is not None:
        target_dtype = dtype
    # check range dtype
    if np.issubdtype(target_dtype, np.integer):
        dtype_max = np.iinfo(target_dtype).max
    elif np.issubdtype(target_dtype, np.float):
        dtype_max = np.finfo(target_dtype).max
    assert shifted_ims.max() <= dtype_max
    shifted_ims = shifted_ims.astype(target_dtype, copy=False)
    return shifted_ims


#--------------------------------------------------
def segmented_detectors(im_shape=(256,)*2, rio=(28, 64), cyx=None, 
                        ac_det_roll=0, dtype='u2'):
    '''
    Generate 8-segment detector images for use in synthetic segmented
    DPC analysis.
    
    Parameters
    ----------
    im_shape : length 2 tuple
        Detector image shape.
    rio : length 2 tuple
        Radius of inner and outer detector edges.
    cyx : length 2 tuple
        Centre of detector in pixels.
        If None, the centre is used. Use (128,)*2 for a (256,)*2 image.
    ac_det_roll : integer
        Anticlockwise roll of detector ordering.
        If 0, no change on order.
    dtype : numpy dtype
        Datatype of returned array.
    
    Returns
    -------
    detectors : ndarray
        Array of shape (8,)+`im_shape`, with values 0 or 1. The first 
        dimension is the detectors, ordered clockwise from top left. The
        first four are the inner detectors, the last four are the outer
        detectors.
    
        Detector layout:
            0 1
            3 2
    
    Examples
    --------
    
    >>> from fpd import synthetic_data
    >>> import matplotlib.pylab as plt
    >>> plt.ion()
    
    >>> detectors = synthetic_data.segmented_detectors(rio=(28, 128))
    
    >>> weights = np.arange(1, detectors.shape[0]+1, dtype='u2')
    >>> det_ims = detectors * weights[..., None, None]
    >>> det_im = det_ims.sum(0)
    >>> im = plt.matshow(det_im, cmap='gray')
    
    For detector layout:
        2 1
        3 0
    
    >>> detectors = synthetic_data.segmented_detectors(rio=(28, 128), ac_det_roll=2)
    
    '''
    
    if cyx is None:
        cyx = [t/2.0 for t in im_shape]
    cyx = [t-0.5 for t in cyx] # subtract 0.5 for indexing from 0
    cy, cx = cyx
    
    rin, rout = rio
    
    yi, xi = np.indices(im_shape)
    ri = np.hypot(yi-cy, xi-cx)

    tr = np.logical_and(yi <= cy, xi > cx)
    tl = np.logical_and(yi <= cy, xi <= cx)
    bl = np.logical_and(yi > cy, xi <= cx)
    br = np.logical_and(yi > cy, xi > cx)
    
    det_masks = np.array([tl, tr, br, bl])
    qds = reduce(lambda x, y: np.logical_or(x, y), det_masks)
    assert np.all(qds)
    qim = det_masks * (np.arange(len(det_masks))+1)[:, None, None]
    qim = qim.sum(0)
    #qim = tr*1 + tl*2 + bl*3 + br*4     
    # plt.matshow(qim, cmap='gray')

    si = ri <= rin
    so = np.logical_and(~si, ri <= rout) 
    #rim = si*1 + so*2; plt.matshow(rim, cmap='gray')

    # all segment images
    sim = qim * (si*1 + so*10)
    # plt.matshow(sim, cmap='gray')

    uv = np.unique(sim)
    uv = uv[np.where(uv != 0)]    # remove 0

    detectors = np.zeros((len(uv),) + im_shape, dtype=dtype)
    for i, u in enumerate(uv):
        yw, xw = np.where(sim == u)
        detectors[i, yw, xw] = 1
    
    # reorder
    if ac_det_roll != 0:
        # row 0 : inner, 1 : outer
        detectors.shape = (2, 4)+im_shape
        #detectors = np.reshape(detectors, (2,4)+im_shape)
        
        # rotate anticlockwise
        detectors = np.roll(detectors, ac_det_roll, 1)
        
        # flatten 1st dim
        #detectors = np.reshape(detectors, (-1,)+im_shape)
        detectors.shape = (8,)+im_shape
   
    return detectors


#--------------------------------------------------
def segmented_dpc_signals(fp_ims, detectors):
    '''
    Returns DPC signals from focal plane images, `fp_ims`, and 
    segmented detector images, `detectors`.
    
    Parameters
    ----------
    fp_ims : ndarray
        Focal plane images of shape (..., detY, detX).
    detectors : ndarray
        Segmented detector images of shape (n, detY, detX), where n
        is the number of detectors.
    
    Returns
    -------
    det_sigs : ndarray
        Array of shape (n, ...), where n is the number of detectors, and
        the ellipsis is the non-detector dimensions of `fp_ims`.
    
    Examples
    --------
    Create shifted image array, segmented detectors, and pass it in to
    a SegmentedDPC class.
    
    >>> from fpd.synthetic_data import disk_image, shift_array, shift_images
    >>> from fpd.synthetic_data import segmented_detectors, segmented_dpc_signals
    >>> from fpd import SegmentedDPC
    
    >>> radius = 32
    >>> im = disk_image(intensity=1e3, radius=radius, size=256, upscale=8, dtype=float)
    >>> sa = shift_array(scan_len=9, shift_min=-2.0, shift_max=2.0)
    >>> sa = np.asarray(sa)
    >>> data = shift_images(sa, im, noise=False)
    
    >>> detectors = segmented_detectors(im_shape=(256, 256), rio=(24, 128), ac_det_roll=2)
    >>> det_sigs = segmented_dpc_signals(data, detectors)
    >>> d = SegmentedDPC(det_sigs, alpha=radius)
    
    '''
    
    d = fp_ims[None, ...]
    scand = len(d.shape)-len(detectors.shape)

    for i in range(scand): 
        detectors = np.expand_dims(detectors, 1)

    det_sig_ims = d*detectors
    det_sigs = det_sig_ims.sum((-2, -1))
    return det_sigs



def fpd_data_view(im, scan_shape, colours=0):
    '''
    Return a view of an image broadcast to `scan_shape` with optional
    colour axis.
    
    Parameters
    ----------
    im : ndarray
        Image to be viewed.
    scan_shape : tuple
        Scan shape in y, x order.
    colours : integer
        Length of colours. Use 0 for no colour axis.
    
    Returns
    -------
    data : ndarray
        A view of the image `im` of shape `scan_shape` + [(colours,) +] im.shape.
        The colour axis is present if singular (or greater), and omitted
        if `colours` is 0.  
    
    Examples
    --------
    Create a data view of a disk image:
    
    >>> from fpd import synthetic_data
    >>> im = synthetic_data.disk_image(intensity=54, radius=32, size=256)
    >>> data = synthetic_data.fpd_data_view(im, (32,)*2)

    '''
    
    shape = scan_shape
    if colours !=0:
        shape += (colours,)
    shape += im.shape
    data = np.broadcast_to(im, shape)
    #data.__array_interface__['data']
    return data
    






