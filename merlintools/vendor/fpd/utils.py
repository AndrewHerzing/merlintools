
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import warnings
import time
from skimage.restoration import estimate_sigma
from numbers import Integral
from collections.abc import Iterable
from collections import namedtuple

import matplotlib.pylab as plt
plt.ion()


def seq_image_array(a, axes):
    '''
    Creates a view of an ndarray `a` with `axes` at the end and all other
    axes flattened and located at the start.
    
    Parameters
    ----------
    a : ndarray
        Multidimentional array.
    axes : length-2 iterable
        Axes of array containing images.
    
    Returns
    -------
    av : ndarray
        A view of the array `a` with the image `axes` at the end
        and all other axes flattened and located in the first axis.
    unflat_shape : tuple
        The shape of the array before flattening.
    
    See also
    --------
    unseq_image_array
    
    '''
    
    # make axes all +ve
    axes = np.array(axes)
    axes_neg = axes < 0
    if axes_neg.any():
        axes[axes_neg] += a.ndim

    # create view where images are last
    av = np.moveaxis(a, axes, (-2, -1))
    unflat_shape = av.shape
    
    # reshape to flatten other axes
    ns = (np.prod(av.shape[:-2]),) + av.shape[-2:]
    av = np.reshape(av, ns)
    
    return (av, unflat_shape)


def unseq_image_array(a, axes, unflat_shape):
    '''
    Parameters
    ----------
    a : ndarray
        Multidimentional array.
    axes : length-2 iterable
        Axes of array containg images.
    
    Returns
    -------
    arsv : ndarray
        The unflattened and reshaped array matching the original array.
    
    Notes
    -----
    Only 1-D or 2-D data generated from each image is currently supported. 
    
    See also
    --------
    seq_image_array
    
    '''
    
    # reshape
    ns = unflat_shape[:-2] + a.shape[1:]
    auf = np.reshape(a, ns)
    
    # move data to correct axes
    im_dims = a.ndim - 1
    # this is the dimensions replacing the image axes (may be anything >=1)
    # just handle 1 or 2 for now
    if im_dims == 1:
        a_rs = dvufrs = np.moveaxis(auf, -1, axes[0])
    elif im_dims == 2:
        a_rs = np.moveaxis(auf, (-2, -1), axes)
    else:
        raise NotImplementedError
    
    return a_rs


def gaus_im_div1d(image, sigma, mode='nearest', cval=0.0, truncate=4.0):
    '''
    Derivative of Gaussian image derivative using two 1-D derivatives. 
    
    Parameters
    ----------
    image : ndarray
        2-D array of which the derivative is calculated.
    sigma : scalar
        Width of the Gaussian used in the derivative.   
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The `mode` parameter determines how the array borders are
        handled, where `cval` is the value when mode is equal to
        'constant'. Default is 'reflect'
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0
    truncate : float, optional
        Truncate the filter at this many standard deviations.
        Default is 4.0.
    
    Returns
    -------
    im_div : ndarray
        2-D array of the `image` derivative.

    '''
    
    im_div_y = gaussian_filter1d(image, sigma, 0, 1, mode=mode, truncate=truncate)
    im_div_x = gaussian_filter1d(image, sigma, 1, 1, mode=mode, truncate=truncate)
    im_div = np.hypot(im_div_y, im_div_x)
    return im_div


def median_lc(a, axis=1, sigma=1):
    '''
    Median line correction of 2-D images.
    
    Parameters
    ----------
    a : ndarray
        2-D image array.
    axis : int in [0, 1]
        Axis of lines to correct (fast scan).
    sigma : scalar
        1-D Gaussian convolution width, used to avoid resolution
        issues in digitised data. 
    
    Returns
    -------
    ac : ndarray
        Corrected 2-D image array.
    
    '''
    
    dt = a.dtype
    af = a.astype(float, copy=False)
    
    # to avoid digitisation issues
    if sigma !=0:
        from scipy.ndimage.filters import gaussian_filter1d
        ag = gaussian_filter1d(af, sigma, axis=axis)
    else:
        ag = af
    med = np.median(ag, axis)
    med -= np.mean(med)
    
    if axis == 1:
        med = med[:, None]
    else:
        med = med[None, :]
    
    ac = af - med
    ac = ac.astype(dt, copy=False)
    return ac


def median_of_difference_lc(a, axis=1, sigma=1):
    '''
    Median of difference line correction of 2-D images.
    
    Parameters
    ----------
    a : ndarray
        2-D image array.
    axis : int in [0, 1]
        Axis of lines to correct (fast scan).
    sigma : scalar
        1-D Gaussian convolution width, use
    
    Returns
    -------
    ac : ndarray
        Corrected 2-D image array.
    
    '''
    
    dt = a.dtype
    af = a.astype(float, copy=False)
    
    # gradient
    ag = np.diff(af, 1, 1-axis)
    if sigma !=0:
        from scipy.ndimage.filters import gaussian_filter1d
        ag = gaussian_filter1d(ag, sigma, axis=axis)

    # median of differences
    med = np.median(ag, axis)
    
    if axis == 1:
        med = med[:, None]
        med = np.row_stack((np.zeros_like(med[:1, :]), med))
    else:
        med = med[None, :]
        med = np.column_stack((np.zeros_like(med[:, :1]), med))
    
    # restore 2-D
    med = np.cumsum(med, 1-axis)
    med -= np.mean(med)
        
    ac = af - med
    ac = ac.astype(dt, copy=False)
    return ac


def gaussian_2d(yx, amplitude, y0, x0, sigma_y, sigma_x, theta, offset):
    '''
    
    Parameters
    ----------
    yx : ndarray
        (2, N) array of ((y, x), N) values.
    amplitude : scalar
        Peak intensity.
    y0 : scalar
        y-axis centre.
    x0 : scalar
        x-axis centre.
    sigma_y : scalar
        y-axis stdev.
    sigma_x : scalar
        x-axis stdev.
    theta : scalar
        Rotation in degrees, anticlockwise when viewed with the
        origin at the top.
    offset : scalar
        Constant offset.
        
    Returns
    -------
    g : ndarray
        1-D raveled array of Gaussian values.
    
    Examples
    --------
    Unflattened 2-D Gaussian plot.
    
    >>> import numpy as np
    >>> import matplotlib.pylab as plt
    >>> plt.ion()
    >>> from fpd.utils import gaussian_2d
    
    >>> fit_hw = 7
    >>> y, x = np.indices((fit_hw*2+1,)*2)-fit_hw

    # amplitude, y0, x0, sigma_y, sigma_x, theta, offset
    >>> g = gaussian_2d((y, x), 5, 0, 0, 2, 1, np.deg2rad(10), 1)
    >>> g = g.reshape(y.shape)
    >>> plt.matshow(g)  
    
    '''
    
    y, x = yx
    x0 = float(x0)
    y0 = float(y0)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)))
    return g.ravel()


def gaussian_2d_peak_fit(image, yc, xc, fit_hw=None, smoothing=0, plot=False, 
                         plot_mode='individual', plot_log=True, **kwds):
    '''
    2-D peak fitting based on known coordinate estimates.

    Parameters
    ----------
    image : ndarray
        2-D image array.
    yc : scalar or ndarray
        y-axis peak centre.
    xc : scalar or ndarray
        x-axis peak centre.
    fit_hw : integer or ndarray
        Half width of area to fit to.
    smoothing : scalar
        Stdev of Gaussian smoothing applied to image.
    plot : boolean
        If True, the fitted region and centre is plotted.
    plot_mode : string
        One of ['individual', 'all']. If 'individual', each valid region
        is plotted separately. If 'all', all centres are plotted on the
        entire image.
    plot_log : bool
        If True and `plot_mode` == 'all', `image` is plotted with a logarithmic
        intensity scale.
    
    Additional kwds are passed to `scipy.optimize.curve_fit`.
    
    Returns
    -------
    popt : ndarray
        Optimised parameters. See `gaussian_2d` for parameter details.
    perr : ndarray
        Parameter error from the covariance matrix. See `gaussian_2d` for
        parameter details. `perr = np.sqrt(np.diag(pcov)).` See 
        `sp.optimize.curve_fit` for details.
    
    Notes
    -----
    Edge cases and fits raising errors return nans.
    
    
    '''
    
    plot_mode = plot_mode.lower()
    plot_modes = ['individual', 'all']
    if plot_mode not in plot_modes:
        raise Exception("`plot_mode` must be one of [`individual`, `all`]")
    
    plot_all = False
    if plot_mode == plot_modes[1]:
        plot_all = True
    
    sy, sx = image.shape
    image = np.array(image, dtype=float)
    if smoothing != 0:
        image = gaussian_filter(image, smoothing)
    
    yc_it = isinstance(yc, Iterable)
    xc_it = isinstance(xc, Iterable)
    fit_hw_it = isinstance(fit_hw, Iterable)
    
    all_it = np.all([yc_it, xc_it, fit_hw_it])
    not_all_it = np.all([not t for t in [yc_it, xc_it, fit_hw_it]])
    if not all_it and not not_all_it:
        raise Exception('`yc`, `xc` and `fit_hw` must be all scalars or iterables.')
    
    if not_all_it:
        yc = np.array([yc])
        xc = np.array([xc])
        fit_hw = np.array([fit_hw])
    if all_it:
        if len(yc) != len(xc) != len(fit_hw):
            raise Exception('`yc`, `xc` and `fit_hw` must be of the same length.')

    popts = np.repeat(np.array((np.nan,)*7)[None], len(yc), 0)
    perrs = popts.copy() 
    for i, (yci, xci, fit_hwi) in enumerate(zip(yc, xc, fit_hw)):
        yc_int, xc_int = int(yci), int(xci)
        edge_case = ((yc_int < fit_hwi) or (xc_int < fit_hwi)
                    or ((sy - yc_int -1) < fit_hwi) or ((sx - xc_int -1) < fit_hwi))
        
        if not edge_case:
            # fitting area width in pixels, tot = (2*fit_hwi+1)
            fit_hwi = int(max([2, fit_hwi]))

            # x and y are centred at peak point, so fitted values are deltas
            y, x = np.indices((fit_hwi*2+1,)*2)-fit_hwi
            
            region = image[yc_int-fit_hwi:yc_int+fit_hwi+1, xc_int-fit_hwi:xc_int+fit_hwi+1]
            if plot and not plot_all:
                plt.matshow(region)
                ax = plt.gca()
            
            # amplitude, x0, y0, sigma_x, sigma_y, theta, offset
            v = image[yc_int, xc_int]
            initial_guess = (v/fit_hwi, 0, 0, fit_hwi, fit_hwi, 0, region.min())
            try:
                kw_dict = {'p0':initial_guess, 'maxfev':1600*2}
                kw_dict.update(kwds)        
                popt, pcov = curve_fit(gaussian_2d, (y, x), region.ravel(), **kw_dict)
                # rectify sigmas
                popt[3:5] = np.abs(popt[3:5])
                if plot and not plot_all:
                    ax.plot(popt[2]+fit_hwi, popt[1]+fit_hwi, '+g')
                
                # convert to absolute values
                popt[1] += yc_int
                popt[2] += xc_int
                
                popts[i] = popt
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    perr = np.sqrt(np.diag(pcov))
                    perrs[i] = perr
            except Exception as e:
                print(e)
    
    if plot and plot_all:
        norm = None
        if plot_log == True:
            import matplotlib as mpl
            norm = mpl.colors.LogNorm()
        plt.matshow(image, norm=norm)
        plt.plot(popts[:, 2], popts[:, 1], 'rx')
    
    return popts, perrs


def snr_single_image(image, s=1, w=2, order=1, hw=5, mode='poly', pad_mode='median',
                     plot=True, signal_from='contrast', per_pixel=False):
    '''
    Single image SNR calculation using auto-correlation and
    polynomial extrapolation or Gaussian interpolation, based on [1].
    
    Parameters
    ----------
    image : 2-D array
        Input image used to calculate SNR.
    s : int
        Start index of polynomial fit.
    w : int
        Width of region used in the polynomial fit. If mode is `gaussian`, this sets
        the fit half-width size.
    order : int
        Order of the polynomial fit.
    hw : int
        Half width of the cross correlation calculation space.
    mode : string
        Extrapolation method. One of: ['poly', 'gaussian'].
    pad_mode : string
        The pad mode passed to `np.pad`.
    plot : bool
        If True, the results are plotted using Matplotlib.
    signal_from : str
        Controls what quantity is used for the signal. One of
        ['contrast', 'zero']. 
        If 'contrast', only variation in the image is taken as signal.
        If 'zero', the signal is taken from 0.
    per_pixel : bool
        If True, the values returned are per pixel. Otherwise, the values
        represent those for the entire image.
    
    Returns
    -------
    snr_res: namedtuple
        Named tuple of `snrt snry snrx st nt sy ny sx nx`, where:
            snrX is the SNR, sX is the signal**2, and nX is the noise**2.
            X is `t` for total, `y` for y-axis and `x` for x-axis.
    
    Notes
    -----
    The algorithm assumes strong noise-free image correlations across a few pixels.
    
    For smoothly varying blob-like images, such as atomic resolution STEM images, the 
    Gaussian fit may be more appropriate than the other functions implemented. 
    
    References
    ----------
    1. J.T.L. Thong et. al, Single-image signal-to-noise ratio estimation, Scanning, 23, 328 (2001). 
    https://onlinelibrary.wiley.com/doi/epdf/10.1002/sca.4950230506
    
    See Also
    --------
    snr_single_image_wl, snr_two_image, decibel
    
    '''
    
    signal_froms = ['contrast', 'zero']
    if signal_from not in signal_froms:
        raise Exception("'signal_from' (%s) must be one of: " %(signal_from), str(signal_froms))        
    
    mode = mode.lower()
    known_modes = ['poly', 'gaussian']
    if mode not in known_modes:
        raise NotImplementedError('known modes:', known_modes)
    
    from scipy import signal
    
    # pad image for valid region
    im_pad = np.pad(image, hw, mode=pad_mode)#, constant_values=image.mean())
    # auto-correlation
    ac = signal.convolve2d(image, im_pad[::-1, ::-1], mode='valid')  
    
    if mode == 'poly':
        # dummy x for poly fits   
        # high side
        xh = np.arange(s, s+w)
        px1 = np.polyfit(xh, ac[hw, hw+s:hw+s+w], order)
        py1 = np.polyfit(xh, ac[hw+s:hw+s+w, hw], order)
        # low side
        xl = -xh[::-1]
        px0 = np.polyfit(xl, ac[hw, hw-(s+w-1):hw-(s-1)], order)
        py0 = np.polyfit(xl, ac[hw-(s+w-1):hw-(s-1), hw], order)

        xi0 = np.polyval(px0, 0)
        yi0 = np.polyval(py0, 0)
        xi1 = np.polyval(px1, 0)
        yi1 = np.polyval(py1, 0)

        #print(xi0, xi1)
        #print(yi0, yi1)
        
        # ac(0,0)_interpolated
        nfx = np.mean([xi0, xi1])
        nfy = np.mean([yi0, yi1])
        nf = np.mean([nfx, nfy])
    elif mode == 'gaussian':
        xg = np.arange(2*w+1) - w
        x_sc = np.delete(xg, w)
        
        acx = ac[hw, hw-w:hw+w+1]
        acy = ac[hw-w:hw+w+1, hw]
        acx_sc = np.delete(acx, w)
        acy_sc = np.delete(acy, w)
        
        def gaussian_1d(x, amplitude, x0, sigma, offset):
            g = offset + amplitude * np.exp(-(x - x0)**2 / (2.0*sigma**2))
            return g
        
        p0x = (acx.ptp(), 0.0, 2.0, acx.min())
        p0y = (acy.ptp(), 0.0, 2.0, acy.min())
        
        popt_x, pcov_x = curve_fit(gaussian_1d, x_sc, acx_sc, p0=p0x)
        popt_y, pcov_y = curve_fit(gaussian_1d, x_sc, acy_sc, p0=p0y)
        
        nfx = popt_x[0] + popt_x[3]
        nfy = popt_y[0] + popt_y[3]
        nf = np.mean([nfx, nfy])
    
    # mode dependent: nf, nfy, nfx
    # ac(0,0)
    n0 = ac[hw, hw]
    
    # background
    if signal_from == 'contrast':
        m = (image.mean())**2 * image.size
    if signal_from == 'zero':
        m = 0
    
    # n: noise, s: signal
    nt = n0 - nf
    st = nf - m
    snrt = st / nt
    
    ny = n0 - nfy
    sy = nfy - m
    snry = sy / ny

    nx = n0 - nfx
    sx = nfx - m
    snrx = sx / nx
    
    
    if plot:
        x = np.arange(2*hw+1)-hw
    
        plt.figure(figsize=(7, 7))
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        ax2 = plt.subplot2grid((3, 3), (0, 2), colspan=1, rowspan=2) # right, y
        ax3 = plt.subplot2grid((3, 3), (2, 0), colspan=2, rowspan=1) # bottom, x
        ax4 = plt.subplot2grid((3, 3), (2, 2), colspan=1, rowspan=1) # image
        ax2.invert_yaxis()
        ax1.xaxis.set_visible(False)
        ax1.yaxis.set_visible(False)
        ax4.xaxis.set_visible(False)
        ax4.yaxis.set_visible(False)
        
        ax3.set_xlabel('x')
        ax2.set_ylabel('y')
        
        # data
        ax1.imshow(ac, extent=(-hw, hw, -hw, hw), interpolation='nearest')
        ax2.plot(ac[:, hw], x, '-o', alpha=0.5)
        ax3.plot(x, ac[hw, :], '-o', alpha=0.5)
        
        ax4.imshow(image, interpolation='nearest')
        
        if mode == 'poly':
            # fits
            xh = np.arange(0, s+w)
            xl = -xh[::-1]
            
            # high side
            xi1 = np.polyval(px1, xh)
            yi1 = np.polyval(py1, xh)
            ax2.plot(yi1, xh, '--k')
            ax3.plot(xh, xi1, '--k')
            
            # low side
            xi0 = np.polyval(px0, xl)
            yi0 = np.polyval(py0, xl)
            ax2.plot(yi0, xl, '--k')
            ax3.plot(xl, xi0, '--k')
        
        if mode == 'gaussian':
            xg = np.linspace(xg.min(), xg.max(), 21)
            x_fit = gaussian_1d(xg, *popt_x)
            y_fit = gaussian_1d(xg, *popt_y)
            ax2.plot(y_fit, xg, '--k')
            ax3.plot(xg, x_fit, '--k')
            
        ax2.set_ylim(hw, -hw)
        ax3.set_xlim(-hw, hw)
        
        ax1.set_title('SNR: %0.3f' %(snrt))
        ax2.set_title('SNRy: %0.3f' %(snry))
        ax3.set_title('SNRx: %0.3f' %(snrx))
        
        plt.tight_layout()
    
    precision = snrt ** -0.5
    
    if per_pixel:
        st /= image.size
        nt /= image.size
        sy /= image.size
        ny /= image.size
        sx /= image.size
        nx /= image.size
    
    SNR_Result = namedtuple('SNR_Result', 'snrt snry snrx st nt sy ny sx nx precision')
    snr_res = SNR_Result(snrt=snrt, snry=snry, snrx=snrx,
                         st=st, nt=nt, sy=sy, ny=ny, sx=sx, nx=nx,
                         precision=precision)
    return snr_res


def snr_single_image_wl(image, signal_from='contrast', per_pixel=False):
    '''
    Single image SNR using a noise estimated from the wavelet analysis [1]
    implemented in skimage.restoration.estimate_sigma.
    
    Parameters
    ----------
    image : 2-D array
        Input image used to calculate SNR.
    signal_from : str
        Controls what quantity is used for the signal. One of
        ['contrast', 'zero']. 
        If 'contrast', only variation in the image is taken as signal.
        If 'zero', the signal is taken from 0.
    per_pixel : bool
        If True, the values returned are per pixel. Otherwise, the values
        represent those for the entire image.
    
    Returns
    -------
    snr_res: namedtuple
        Named tuple of `snrt st nt precision`, where:
            snrt is the SNR
            st is the signal**2
            nt is the noise**2
            precision is snrt ** -0.5
    
    References
    ----------
    1. D. L. Donoho and I. M. Johnstone. “Ideal spatial adaptation by wavelet shrinkage.” Biometrika 81.3 (1994): 425-455. DOI:10.1093/biomet/81.3.425
    
    See Also
    --------
    snr_single_image, snr_two_image, decibel
    
    '''
    
    signal_froms = ['contrast', 'zero']
    if signal_from not in signal_froms:
        raise Exception("'signal_from' (%s) must be one of: " %(signal_from), str(signal_froms))        
    
    sigma = estimate_sigma(image, average_sigmas=False, multichannel=False)
    nt = (sigma**2) * image.size
    st = (image**2).sum() - nt
    if signal_from == 'contrast':
        st -= (image.mean())**2 * image.size
    snrt = st / nt
    
    precision = snrt ** -0.5
    
    if per_pixel:
        st /= image.size
        nt /= image.size
    
    SNR_Result = namedtuple('SNR_Result', 'snrt st nt precision')
    snr_res = SNR_Result(snrt=snrt, st=st, nt=nt, precision=precision)
    return snr_res


def snr_two_image(image, ref):
    '''
    Two image signal to noise ratio, after [1].
    
    Parameters
    ----------
    image : 2-D array
        Input image used to calculate SNR.
    ref : 2-D array
        Reference image used to calculate SNR.
    
    Returns
    -------
    snr : float
        The signal to noise ratio.
    
    Notes
    -----
    This implementation assumes the images are aligned.
    
    The equivalet precision may be calculated as snr **-0.5. 
    
    References
    ----------
    1. J. Frank, The Role of Correlation Techniques in Computer Image Processing. In Computer Processing of Electron Microscope Images (Ed. Peter W. Hawkes), Springer Heidelberg, Berlin (1980).
    https://link.springer.com/chapter/10.1007/978-3-642-81381-8_5
    
    See Also
    --------
    snr_single_image, snr_single_image_wl, decibel
    
    '''
    
    cc = (image * ref).sum() / image.size
    
    im, iv = image.mean(), image.std()
    rm, rv = ref.mean(), ref.std()
    
    rho = (cc - im*rm) / (iv*rv)
    snr = rho / (1 - rho)
    
    return snr


def decibel(power_ratio):
    '''
    Calculate decibel conversion of power_ratio.
    
    Parameters
    ----------
    power_ratio : scalar or iterable
        The power ratio to be converted.
    
    Returns
    -------
    dB : scalar or ndarray
        The decibel conversion.
    
    See Also
    --------
    snr_single_image, snr_single_image_wl, snr_two_image
    
    '''
    
    return 10 * np.log10(power_ratio)


def smooth7(min_val, order=8, base_two=False, even=True):
    '''
    Returns 7-smooth number.
    
    Parameters
    ----------
    min_val : int
        Minimum value to return factors for. If not a 7-smooth number, the
        returned number will be larger then this.
    order : int
        Sets the scale of the range of factors explored. See notes.
    base_two : bool
        If True, the returned 7-smooth number is equal to 2**n.
    even : bool
        If True, the returned 7-smooth number is even.
    
    Returns
    -------
    n : int
        The 7-smooth number.
    
    Notes
    -----
    A 7-smooth number is one whose factors are all prime numbers in the
    range [1 7]. FFTs often run with improved efficiency when they are of
    this size.
    
    The smallest number is not guaranteed when `base_two` is False and
    `min_val` is large compared to `order` (see below). Increasing the value
    of `order` improves the reliability of returning the minimum factor at
    the expense of run time.
    
    The algorithm simply calculates the products of all combinations of four
    values from np.array([[2, 3, 5, 7]])**np.arange(order) and returns the
    lowest value.
    
    '''
    
    if base_two:
        n = 2**np.ceil(np.log2(min_val)).astype(int)
    else:
        from itertools import combinations, combinations_with_replacement
        a = np.array([[2, 3, 5, 7]])**np.arange(order)[:, None]
        c = combinations_with_replacement(a.flatten(), 4)
        ct = np.array([np.prod(ci) for ci in c])
        ct = np.unique(ct)
        ct = ct[ct>0]
        #plt.loglog(ct)
        if even:
            ct = ct[ct%2 == 0]
        i = np.searchsorted(ct, min_val)
        n = ct[i]
    return n


def gaussian_fwhm(sigma):
    '''
    Full width at half maximum of a Gaussian distribution of width sigma.
    
    Parameters
    ----------
    sigma : scalar
        Standard deviation of Gaussian distribution.
    
    Returns
    -------
    fwhm : scalar
        The full width at half maximum.
    
    '''
    
    fwhm = 2.0 * (2.0 * np.log(2.0))**0.5 * sigma
    return fwhm

    
class Timer(object):
    def __init__(self, name=None, print=True):
        '''
        Timer class for use as a with statement context manager.
        
        Parameters
        ----------
        name : str or None
            If not None, a name used for the print statement.
        print : bool
            If False, do nothing.
        
        Examples
        --------
        with Timer('my_timer'):
            print('hello world')
        
        '''
        
        self.name = name
        self.print = print

    def __enter__(self):
        if self.print:
            self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.print:
            self.stop = time.perf_counter()
            
            if self.name:
                s = "'%s' duration: " %(str(self.name))
            else:
                s = 'Duration: '
            s += '%0.3g s' %(self.stop - self.start)
            print(s)


def int_factors(n, memory_efficient=False):
    '''
    Calculate factors of positive integer, n, in ascending order.
    
    Parameters
    ----------
    n : int
        Positive integer for which to calculate factors.
    memory_efficient : bool
        If True, a slower memory efficient algorithm is used.
    
    Returns
    -------
    factors : ndarray
        A 1-D array of factors of `n`, in ascending order.
    
    '''
    
    if not isinstance(n, Integral):
        raise ValueError("'n' (%f) must be an int" %(n))
    
    if n <= 0:
        raise ValueError("'n' (%s) must be > 0" %(str(n)))
    
    if memory_efficient:
        factors = []
        for i in range(1, n+1):
            if n % i == 0:
                factors.append(i)
        factors = np.array(factors)
    else:
        nat_nums = np.arange(1, n+1)
        rems = np.remainder(n, nat_nums)
        inds = np.where(rems == 0)
        factors = nat_nums[inds]
    
    return factors


def nearest_int_factor(n, f, memory_efficient=False):
    '''
    Calculate nearest positive integer factor of n to f.
    
    Parameters
    ----------
    n : int
        Positive integer for which to calculate factors.
    f : scalar
        Positive value used to select from `n` by the nearest value.
    memory_efficient : bool
        If True, a slower memory efficient algorithm is used.
    
    Returns
    -------
    Tuple of factor, factors
    factor : scalar
        The nearest factor of `n` to `f` in `factors`. If two factors are
        equidistant from `f`, the smaller value is returned.
    factors : ndarray
        A 1-D array of factors of `n`, in ascending order.
    
    '''
    
    if not isinstance(n, Integral):
        raise ValueError("'n' (%f) must be an int" %(n))
    
    if n <= 0:
        raise ValueError("'n' (%s) must be > 0" %(str(n)))
    
    factors = int_factors(n)
    i = np.abs(factors - f).argmin()
    factor = factors[i]
    
    return factor, factors


