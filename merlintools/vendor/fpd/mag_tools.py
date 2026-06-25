
import numpy as np
from scipy.constants import mu_0, h, e, pi
from numpy.fft import fft2, ifft2, fftfreq, fftshift
from collections import namedtuple
import matplotlib.pylab as plt
plt.ion()

from .tem_tools import e_lambda


################### Patterns
def landau(shape=(128,)*2, pad_width=0, origin='top', plot=False):
    '''
    Generates an anticlockwise landau pattern.
    
    Parameters
    ----------
    shape : length 2 tuple
        y, x size in pixels.
    pad_width : {None, sequence, array_like, int} 
        If not None, the magnetisation arrays are padded using to np.pad.    
    origin : string in ['top', 'bottom']
        Indicates the direction positive pixel values represent. If 'top', the
        positive, values correspond to increases in y-axis position when plotted
        with (0,0) at the top. If 'bottom', positive pixels represent the movement
        along negative y-axis, when plotted with (0,0) at the top. Note that the
        results are always plotted with origin='top'.
    plot : bool
        If True, the returned data are also plotted.
    
    Returns
    -------
    my, mx : 3-D array
        y- and x- magnetisation in [-1, 1]. 
    
    '''
    
    bottom = origin.lower() == 'bottom'
    
    y, x = np.indices(shape)
    regions = (x>y)[::-1]*1 + (x>y)*1
    
    my = np.zeros(shape, dtype=float)
    my[regions==0] = +1
    my[regions==2] = -1
    mx = (my == 0)*1.0
    mx[:mx.shape[0]//2] *= -1
    
    if pad_width is not None:
        my = np.pad(my, pad_width, mode='constant', constant_values=0)
        mx = np.pad(mx, pad_width, mode='constant', constant_values=0)
    
    if bottom:
        my *= -1
    
    if plot:
        plt.matshow(my)
        plt.title('My')
        plt.matshow(mx)
        plt.title('Mx')
    
    return np.array([my, mx])


def stripes(nstripes=4, h2h=False, shape=(128,)*2, pad_width=0, origin='top', plot=False):
    '''
    Generates a stripe pattern.
    
    Parameters
    ----------
    nstripes : int
        Number of stripes.
    h2h : bool
        If True, stripes are head to head.    
    shape : length 2 tuple
        y, x size in pixels.
    pad_width : {None, sequence, array_like, int} 
        If not None, the magnetisation arrays are padded using to np.pad.    
    origin : string in ['top', 'bottom']
        Indicates the direction positive pixel values represent. If 'top', the
        positive, values correspond to increases in y-axis position when plotted
        with (0,0) at the top. If 'bottom', positive pixels represent the movement
        along negative y-axis, when plotted with (0,0) at the top. Note that the
        results are always plotted with origin='top'.
    plot : bool
        If True, the returned data are also plotted.
    
    Returns
    -------
    my, mx : 3-D array
        y- and x- magnetisation in [-1, 1]. 
    
    Examples
    --------
    Infinitely long antiparallel stripes.
    >>> my, mx = stripes(nstripes=2, shape=(256,128), pad_width=[(0,)*2, (50,)*2], plot=True)
    
    Infinitely long head-to-head stripes (note that only my and mx have swapped below):
    >>> mx, my = stripes(nstripes=2, shape=(256,128), pad_width=[(0,)*2, (50,)*2])
    
    '''
    
    bottom = origin.lower() == 'bottom'
    
    y, x = np.indices(shape)
    regions = x[0] // (shape[1] // nstripes)
    labels = np.unique(regions)

    for i, lab in enumerate(labels):
        inds = np.where(regions == lab)
        odd = i % 2
        if odd:
            factor = +1
        else:
            factor = -1
        y[:, inds] = factor

    mx = y*0.0
    my = y

    if h2h:
        my[:shape[0]//2] *= -1
    
    if pad_width is not None:
        my = np.pad(my, pad_width, mode='constant', constant_values=0)
        mx = np.pad(mx, pad_width, mode='constant', constant_values=0)
    
    if bottom:
        my *= -1
    
    if plot:
        plt.matshow(my)
        plt.title('My')
        plt.matshow(mx)
        plt.title('Mx')
    
    return np.array([my, mx])


def uniform(shape=(128,)*2, pad_width=0, origin='top', plot=False):
    '''
    Generates a uniform pattern.
    
    Parameters
    ----------
    shape : length 2 tuple
        y, x size in pixels.
    pad_width : {None, sequence, array_like, int} 
        If not None, the magnetisation arrays are padded using to np.pad.    
    origin : string in ['top', 'bottom']
        Indicates the direction positive pixel values represent. If 'top', the
        positive, values correspond to increases in y-axis position when plotted
        with (0,0) at the top. If 'bottom', positive pixels represent the movement
        along negative y-axis, when plotted with (0,0) at the top. Note that the
        results are always plotted with origin='top'.
    plot : bool
        If True, the returned data are also plotted.
    
    Returns
    -------
    my, mx : 3-D array
        y- and x- magnetisation in [-1, 1]. 
    
    '''
    
    bottom = origin.lower() == 'bottom'
    
    y, x = np.indices(shape)
    my = x*0.0
    mx = my + 1.0
    
    if pad_width is not None:
        my = np.pad(my, pad_width, mode='constant', constant_values=0)
        mx = np.pad(mx, pad_width, mode='constant', constant_values=0)
    
    if bottom:
        my *= -1
    
    if plot:
        plt.matshow(my)
        plt.title('My')
        plt.matshow(mx)
        plt.title('Mx')
    
    return np.array([my, mx])


def grad(shape=(128,)*2, pad_width=0, origin='top', plot=False):
    '''
    Generates a gradient pattern.
    
    Parameters
    ----------
    shape : length 2 tuple
        y, x size in pixels.
    pad_width : {None, sequence, array_like, int} 
        If not None, the magnetisation arrays are padded using to np.pad.    
    origin : string in ['top', 'bottom']
        Indicates the direction positive pixel values represent. If 'top', the
        positive, values correspond to increases in y-axis position when plotted
        with (0,0) at the top. If 'bottom', positive pixels represent the movement
        along negative y-axis, when plotted with (0,0) at the top. Note that the
        results are always plotted with origin='top'.
    plot : bool
        If True, the returned data are also plotted.
    
    Returns
    -------
    my, mx : 3-D array
        y- and x- magnetisation in [-1, 1]. 
    
    '''
    
    bottom = origin.lower() == 'bottom'
    
    y, x = np.indices(shape)
    my = y/float(y.max())
    mx = x * 0.0
    
    if pad_width is not None:
        my = np.pad(my, pad_width, mode='constant', constant_values=0)
        mx = np.pad(mx, pad_width, mode='constant', constant_values=0)
    
    if bottom:
        my *= -1
    
    if plot:
        plt.matshow(my)
        plt.title('My')
        plt.matshow(mx)
        plt.title('Mx')
    
    return np.array([my, mx])


def divergent(shape=(128,)*2, pad_width=0, origin='top', plot=False):
    '''
    Generates a divergent pattern.
    
    Parameters
    ----------
    shape : length 2 tuple
        y, x size in pixels.
    pad_width : {None, sequence, array_like, int} 
        If not None, the magnetisation arrays are padded using to np.pad.    
    origin : string in ['top', 'bottom']
        Indicates the direction positive pixel values represent. If 'top', the
        positive, values correspond to increases in y-axis position when plotted
        with (0,0) at the top. If 'bottom', positive pixels represent the movement
        along negative y-axis, when plotted with (0,0) at the top. Note that the
        results are always plotted with origin='top'.
    plot : bool
        If True, the returned data are also plotted.
    
    Returns
    -------
    my, mx : 3-D array
        y- and x- magnetisation in [-1, 1]. 

    '''
    
    bottom = origin.lower() == 'bottom'
    
    y, x = np.indices(shape)
    y = y - y.mean()
    x = x - x.mean()
    
    my = y / y.max()
    mx = x / x.max()
    
    if pad_width is not None:
        my = np.pad(my, pad_width, mode='constant', constant_values=0)
        mx = np.pad(mx, pad_width, mode='constant', constant_values=0)
    
    if bottom:
        my *= -1
    
    if plot:
        plt.matshow(my)
        plt.title('My')
        plt.matshow(mx)
        plt.title('Mx')
        
        plt.figure()
        n=8
        plt.quiver(mx[::n,::n], -my[::n, ::n], scale=16)
        plt.gca().set_aspect(1)
        plt.gca().invert_yaxis()
    
    return np.array([my, mx])


def neel(width=8.0, shape=(128,)*2, pad_width=0, origin='top', plot=False):
    '''
    Generates a Neel domain wall pattern.
    
    Parameters
    ----------
    width : scalar
        Full width of the tanh profile.
    shape : length 2 tuple
        y, x size in pixels.
    pad_width : {None, sequence, array_like, int} 
        If not None, the magnetisation arrays are padded using to np.pad.    
    origin : string in ['top', 'bottom']
        Indicates the direction positive pixel values represent. If 'top', the
        positive, values correspond to increases in y-axis position when plotted
        with (0,0) at the top. If 'bottom', positive pixels represent the movement
        along negative y-axis, when plotted with (0,0) at the top. Note that the
        results are always plotted with origin='top'.
    plot : bool
        If True, the returned data are also plotted.
    
    Returns
    -------
    my, mx : 3-D array
        y- and x- magnetisation in [-1, 1]. 

    '''
    
    bottom = origin.lower() == 'bottom'
    
    y, x = np.indices(shape)    
    
    xc1, xc2 = np.percentile(x[0], [100/3.0, 100/3.0*2])
    
    my = np.tanh((x-xc1)/(width*2.0)*np.pi) - np.tanh(-(x-xc2)/(width*2.0)*np.pi)
    my /= 2.0
    mx = (1-my**2)**0.5
    
    if pad_width is not None:
        my = np.pad(my, pad_width, mode='constant', constant_values=0)
        mx = np.pad(mx, pad_width, mode='constant', constant_values=0)
    
    if bottom:
        my *= -1
    
    if plot:
        plt.matshow(my)
        plt.title('My')
        plt.matshow(mx)
        plt.title('Mx')
        
        plt.figure()
        n=8
        plt.quiver(mx[::n,::n], -my[::n, ::n], scale=16)
        plt.gca().set_aspect(1)
        plt.gca().invert_yaxis()
        
        plt.figure()
        plt.plot(my[0], label='My')
        plt.plot(mx[0], label='Mx')
        plt.legend()
    
    return np.array([my, mx])


def vortex(lambda_c=5, r_edge=None, shape=(128,)*2, pad_width=0, origin='top', plot=False):
    '''
    Generates an anticlockwise vortex pattern. The core is modelled
    with sech.
    
    Parameters
    ----------
    lambda_c : scalar
        Core width. If zero, there is no core.
        The core FWHM (pix) = 2.633918 * `lambda_c`.
    shape : length 2 tuple
        y, x size in pixels.
    r_edge : scalar or None
        Radius of the outer edge. 
    pad_width : {None, sequence, array_like, int} 
        If not None, the magnetisation arrays are padded using to np.pad.    
    origin : string in ['top', 'bottom']
        Indicates the direction positive pixel values represent. If 'top', the
        positive, values correspond to increases in y-axis position when plotted
        with (0,0) at the top. If 'bottom', positive pixels represent the movement
        along negative y-axis, when plotted with (0,0) at the top. Note that the
        results are always plotted with origin='top'.
    plot : bool
        If True, the returned data are also plotted.
    
    Returns
    -------
    my, mx : 3-D array
        y- and x- magnetisation in [-1, 1]. 
    
    '''
    
    bottom = origin.lower() == 'bottom'
    
    y, x = np.indices(shape)    

    y = y - y.mean()
    x = x - x.mean()
    r = (x**2 + y**2)**0.5
    t = np.arctan2(y, x)
    
    # core
    if lambda_c == 0:
        mz = r*0 + 1
    else:
        mz = 1 - 1 / np.cosh(r / lambda_c)
    
    # outer
    rm = r * 0 + 1
    if r_edge is not None:
        rm[r>=r_edge] = 0
    
    # anticlockwise
    sign = -1
    my = np.cos(t) * mz * sign * rm
    mx = -np.sin(t) * mz * sign * rm
    
    # pad
    if pad_width is not None:
        my = np.pad(my, pad_width, mode='constant', constant_values=0)
        mx = np.pad(mx, pad_width, mode='constant', constant_values=0)
    
    if bottom:
        my *= -1
    
    if plot:
        plt.matshow(my)
        plt.title('My')
        plt.matshow(mx)
        plt.title('Mx')
    
    return np.array([my, mx])


def cross_tie(lambda_c=128, shape=(128, 256), pad_width=0, origin='top', plot=False):
    '''
    Generates a horizontal cross-tie pattern after [1], with a central anticlockwise
    vortex.
    
    [1] K. L. Metlov, Appl. Phys. Lett. 79 (16), 2609 {2001).
        https://doi.org/10.1063/1.1409946
    
    Parameters
    ----------
    lambda_c : scalar
        The spacing between vortex and antivortex cores in pixels.
        The core FWHM (pix) = 0.349699 * `lambda_c`.
    shape : length 2 tuple
        y, x size in pixels.
    pad_width : {None, sequence, array_like, int} 
        If not None, the magnetisation arrays are padded using to np.pad.    
    origin : string in ['top', 'bottom']
        Indicates the direction positive pixel values represent. If 'top', the
        positive, values correspond to increases in y-axis position when plotted
        with (0,0) at the top. If 'bottom', positive pixels represent the movement
        along negative y-axis, when plotted with (0,0) at the top. Note that the
        results are always plotted with origin='top'.
    plot : bool
        If True, the returned data are also plotted.
    
    Returns
    -------
    my, mx : 3-D array
        y- and x- magnetisation in [-1, 1].
    
    Notes
    -----
    Choosing the ratio of the image width, `shape[1]`, to `lambda_c` to be 2n,
    where n = 1, 2, 3, ... results in a periodic structure in x.
    
    '''
    
    bottom = origin.lower() == 'bottom'
    
    y, x = np.indices(shape)  

    y = (y - y.mean()) * np.pi
    x = (x - x.mean()) * np.pi
    
    n = np.cosh(y / lambda_c)
    mx =  np.sinh(y / lambda_c) / n
    my = -np.sin(x / lambda_c) / n
    mz =  np.cos(x / lambda_c) / n
    # for clockwise, negate mx and my
    # mz is polarity (not used)
    
    # pad
    if pad_width is not None:
        my = np.pad(my, pad_width, mode='constant', constant_values=0)
        mx = np.pad(mx, pad_width, mode='constant', constant_values=0)
    
    if bottom:
        my *= -1
    
    if plot:
        plt.matshow(my)
        plt.title('My')
        plt.matshow(mx)
        plt.title('Mx')
    
    return np.array([my, mx])



################### Conversions
def bt2phasegrad(bt):
    '''
    Phase gradient from integrated induction.
    
    Parameters
    ----------
    bt : ndarray or scalar
        Integrated induction in Tesla*m.
    
    Returns
    -------
    phase_grad : ndarray or scalar
        Phase gradient in radians / m.
    
    '''
    
    phase_grad = 2*pi*e/h * bt
    return phase_grad


def phasegrad2bt(phase_grad):
    '''
    Local integrated induction from phase gradient.
    
    Parameters
    ----------
    phase_grad : ndarray or scalar
        Phase gradient in radians / m.
    
    Returns
    -------
    bt : ndarray or scalar
        Integrated induction in Tesla*m.
    
    '''
    
    bt = phase_grad * h/(2*pi*e)
    return bt


def bt2beta(bt, kV=200.0):
    '''
    Beta from local integrated induction.
    
    Parameters
    ----------
    bt : ndarray or scalar
        Integrated induction in Tesla*m.
    kV : scalar
        Beam energy in kV. The wavelength calculation is relativistic.
    
    Returns
    -------
    beta : ndarray or scalar
        Deflection (semi-) angle in radians.
    
    '''
    
    lamb = e_lambda(kV)
    beta = lamb*e/h*bt
    return beta


def beta2bt(beta, kV=200.0):
    '''
    Local integrated induction from beta.
    
    Parameters
    ----------
    beta : ndarray or scalar
        Deflection (semi-) angle in radians.
    kV : scalar
        Beam energy in kV. The wavelength calculation is relativistic.
    
    Returns
    -------
    bt : ndarray or scalar
        Integrated induction in Tesla*m.
    
    '''
    
    lamb = e_lambda(kV)
    bt = beta / (lamb*e/h)
    return bt


def beta2phasegrad(beta, kV=200.0):
    '''
    Phase gradient from beta.
    
    Parameters
    ----------
    beta : ndarray  or scalar
        Deflection (semi-) angle in radians.
    kV : scalar
        Beam energy in kV. The wavelength calculation is relativistic.
    
    Returns
    -------
    phase_grad : ndarray  or scalar
        Phase gradient in radians / m.
    
    '''
    
    lamb = e_lambda(kV)
    phase_grad = 2*pi/lamb * beta
    return phase_grad


def phasegrad2period(phase_grad):
    '''
    Equivalent periodicity from phase gradient.
    
    Parameters
    ----------
    phase_grad : ndarray or scalar
        Phase gradient in radians / m.
    
    Returns
    -------
    period : ndarray or scalar
        Equivalent periodicity in m.
    
    '''
    
    period = 2*pi/phase_grad
    return period


def tesla2mag(tesla):
    '''
    Magnetisation (A/m) from induction (T).
    
    Parameters
    ----------
    tesla : ndarray or scalar
        Induction in Tesla.
    
    Returns
    -------
    m : ndarray or scalar
        Magnetisation in A/m.
    
    '''
    
    m = tesla * 1e7/(4*np.pi)
    return m


def mag2tesla(m):
    '''
    Induction (T) from magnetisation (A/m).
    
    Parameters
    ----------
    m : ndarray or scalar
        Magnetisation in A/m.
    
    Returns
    -------
    tesla : ndarray or scalar
        Induction in Tesla.
    
    '''
    
    tesla = m / (1e7/(4*np.pi))
    return tesla


def tesla2Oe(tesla):
    '''
    Field (Oe) from induction (T).
    
    Parameters
    ----------
    tesla : ndarray or scalar
        Induction in Tesla.
    
    Returns
    -------
    o : ndarray or scalar
        field in Oe.
    
    '''
    
    o = tesla * 10000
    return o


def tesla2G(tesla):
    '''
    Field (G) from induction (T).
    
    Parameters
    ----------
    tesla : ndarray or scalar
        Induction in Tesla.
    
    Returns
    -------
    g : ndarray or scalar
        field in Oe.
    
    '''
    
    g = tesla * 10000
    return g



################### parameters
def exchange_length(ms, A):
    '''
    Magnetic exchange length [1].
    
    [1] Abo et al., IEEE Trans. Magn., 49(8), 4937 (2013). 
        https://doi.org/10.1109/TMAG.2013.2258028
    
    Parameters
    ----------
    ms :  scalar
        Saturation magnetisation in A/m.
    A : scalar
        Exchange constant in J/m.
    
    Returns
    -------
    lex : scalar
        Exchange length in m.
    
    '''
    
    from scipy.constants import mu_0
    lex = ( A / (0.5*mu_0*ms**2) )**0.5
    return lex



################### Phase, DPC, and Fresnel calculations
def mag_phase(my, mx, ypix=1e-9, xpix=1e-9, thickness=10e-9, pad_width=None,
               phase_amp=10, origin='top', plot=False):
    '''
    Electromagnetic phase calculations from magnetisation [1].
    `fresnel` and `fresnel_paraxial` may be used to calculate Fresnel images
    from the output.
    
    [1] Beleggia et al., APL 83 (2003), DOI: 10.1063/1.1603355.
    
    Parameters
    ----------
    my : 2-D array
        Magnetisation in y-axis in [0, 1], where 1 is taken as Bs of 1 Tesla.
    mx : 2-D array
        Magnetisation in x-axis in [0, 1], where 1 is taken as Bs of 1 Tesla.
    ypix : scalar
        y-axis pixel spacing in metres.
    xpix : scalar
        x-axis pixel spacing in metres.
    thickness : scalar
        Material thicknes metres.
    pad_width : {None, sequence, array_like, int} 
        If not None, the magnetisation arrays are padded using to np.pad.
    phase_amp : scalar
        Factor by which the phase is scalled for cosine plot.
    origin : string in ['top', 'bottom']
        Indicates the direction positive pixel values represent. If 'top', the
        positive, values correspond to increases in y-axis position when plotted
        with (0,0) at the top. If 'bottom', positive pixels represent the movement
        along negative y-axis, when plotted with (0,0) at the top. Note that the
        results are always plotted with origin='top'.
    plot : bool
        If True, the returned data are also plotted.
    
    Returns
    -------
    p : named_tuple
        Contains the following parameters:
    phase : 2-D array
        Phase change in radians.
    phase_grady, phase_gradx : 2-D array
        Phase gradient in radians / metre.
    phase_laplacian : 2-D array
        Phase gradient in radians / metre**2.
    my, mx : 2-D array
        Optionally padded 'magnetisation' in Tesla.
    localBy, localBx : 2-D array
        Local induction. Note this number only relates to saturation induction,
        Bs, under centain conditions.
    
    Examples
    --------
    Generate a Landau pattern, calculate phase properties, and plot DPC signals.
    
    >>> import fpd
    >>> import matplotlib.pylab as plt
    >>> plt.ion()
    
    >>> my, mx = fpd.mag_tools.landau()
    >>> p = fpd.mag_tools.mag_phase(my, mx, plot=True)
    >>> byx = [p.localBy, p.localBx]
    >>> b = fpd.DPC_Explorer(byx, vectrot=270, cmap=2, cyx=(0,0), pct=0, r_max_pct=100)
    
    See also
    --------
    fresnel, fresnel_paraxial, tie
    
    '''
    # TODO
    # add tilt angles
    # add mean inner potential?
    
    
    # implementation is top, meaning +ve y points down (plotted with 0,0 at top,
    # the pixel value indicates the delta on the x-axis.
    bottom = origin.lower() == 'bottom'
    if bottom:
        my = my*-1.0
    
    if pad_width is not None:
        my = np.pad(my, pad_width, mode='constant', constant_values=0)
        mx = np.pad(mx, pad_width, mode='constant', constant_values=0)
    
    phi0 = h/(2*e)
    constant = 1j * pi * mu_0 / phi0

    # convert T to A/m^2
    mx_ft = fft2(mx/mu_0)
    my_ft = fft2(my/mu_0)

    # reciprocal vectors
    ky = fftfreq(mx.shape[0], ypix) * (2*np.pi)
    kx = fftfreq(mx.shape[1], xpix) * (2*np.pi)
    kyy, kxx = np.meshgrid(ky, kx, indexing='ij')

    # eq 3 in Beleggia et al., APL 83 (2003), DOI: 10.1063/1.1603355.
    denom = (kxx**2 + kyy**2)
    # avoide runtime warning
    denom[0, 0] = 1.0
    c = (mx_ft * kyy - my_ft * kxx) / denom
    
    phase_ft = constant * thickness * c
    phase_ft[0, 0] = 0j
    phase = ifft2(phase_ft).real 

    # fft derivative
    phase_gradx = ifft2(phase_ft * kxx * 1j).real
    phase_grady = ifft2(phase_ft * kyy * 1j).real

    # 2nd derivative and laplacian
    phase_gradx2 = ifft2(phase_ft * (kxx * 1j)**2).real
    phase_grady2 = ifft2(phase_ft * (kyy * 1j)**2).real
    phase_laplacian = -(phase_gradx2 + phase_grady2)
    
    # local induction
    localBy = phasegrad2bt(phase_grady) / thickness
    localBx = phasegrad2bt(phase_gradx) / thickness
    
    if plot:
        #s = (slice(pw, -pw),  slice(pw, -pw))
        s = (slice(None, None),  slice(None, None))
        
        plt.matshow(my[s])
        plt.title('My')
        plt.matshow(mx[s])
        plt.title('Mx')
        
        plt.matshow(phase[s])
        plt.title('phase')
        plt.matshow(np.cos(phase[s]*phase_amp))
        plt.title('phase cosine')
        
        plt.matshow(phase_laplacian[s])
        plt.title('phase_laplacian')
        
        plt.matshow(phase_grady[s])
        plt.title('phase_grady')
        plt.matshow(phase_gradx[s])
        plt.title('phase_gradx')
        
        plt.matshow(localBy[s])
        plt.title('Local By')
        plt.matshow(localBx[s])
        plt.title('Local Bx')
        
    PhaseResults = namedtuple('PhaseResults',
                              ['phase', 'phase_grady', 'phase_gradx', 'phase_laplacian', 'my', 'mx', 'localBy', 'localBx'])
    p = PhaseResults(phase, phase_grady, phase_gradx, phase_laplacian, my, mx, localBy, localBx)
    return p


def fresnel_paraxial(a, phase, df=0, ypix=1e-9, xpix=1e-9, theta_c=0, kV=200.0):
    '''
    Fresnel image intensity at low defocus from wavefunction amplitude and phase [1].
    `mag_phase` may be used to generate `phase' from a known magnetisation. 
    
    [1] M. De Graef and Y. Zhu, Magnetic Imaging and Its Applications to Materials,
        in Experimental Methods in the Physical Sciences, 36, 27 (2001).
        https://doi.org/10.1016/S1079-4042(01)80036-9
    
    Parameters
    ----------
    a : ndarray
        Wavefunction amplitude.
    phase : ndarray
        Wavefunction phase in radians.
    df : scalar
        Defocus in m. Positive is underfocus.
    ypix : scalar
        Y-axis pixel spacing in m.
    xpix : scalar
        X-axis pixel spacing in m.
    theta_c : scalar
        Beam convergence in radians.
    kV : scalar
        Beam energy in kV. The wavelength calculation is relativistic.
    
    Returns
    -------
    im : ndarray
        Fresnel image.
    
    See also
    --------
    mag_phase, fresnel, tie
    
    '''
    
    lamb = e_lambda(kV)
    
    i1 = a**2
    
    
    # reciprocal vectors
    ky = fftfreq(phase.shape[0], ypix) * (2*np.pi)
    kx = fftfreq(phase.shape[1], xpix) * (2*np.pi)
    kyy, kxx = np.meshgrid(ky, kx, indexing='ij')
    
    # fft derivative of phase
    phase_ft = fft2(phase)
    #phase_ft[0, 0] = 0j

    # first derivative
    phase_gradx = ifft2(phase_ft * kxx * 1j).real
    phase_grady = ifft2(phase_ft * kyy * 1j).real
    # 2nd derivative
    phase_gradx2 = ifft2(phase_ft * (kxx * 1j)**2).real
    phase_grady2 = ifft2(phase_ft * (kyy * 1j)**2).real
    
    
    # fft derivative of amplitude
    a_ft = fft2(a)
    #a_ft[0, 0] = 0j

    # first derivative
    a_gradx = ifft2(a_ft * kxx * 1j).real
    a_grady = ifft2(a_ft * kyy * 1j).real
    # 2nd derivative
    a_gradx2 = ifft2(a_ft * (kxx * 1j)**2).real
    a_grady2 = ifft2(a_ft * (kyy * 1j)**2).real
    
    
    phase_grady = np.gradient(phase, ypix, axis=0)
    phase_gradx = np.gradient(phase, xpix, axis=1)
    pgrad = np.array([phase_grady, phase_gradx])
    d0 = a[None]**2 * pgrad
    d1 = np.gradient( d0[0], ypix, axis=0 )
    d2 = np.gradient( d0[1], xpix, axis=1 )
    i2 = -lamb * df / (2 * np.pi) * (d1 + d2)
    
    a_grady = np.gradient(a, ypix, axis=0)
    a_gradx = np.gradient(a, xpix, axis=1)
    a_grady2 = np.gradient( a_grady, ypix, axis=0 )
    a_gradx2 = np.gradient( a_gradx, xpix, axis=1 )
    i3 = (theta_c * df)**2 / (2 * np.log(2)) * ( a * (a_grady2 + a_gradx2) - (a * pgrad.sum(0))**2 )

    im = i1 + i2 + i3
    return im

def fresnel(amp, phase, df=0, ypix=1e-9, xpix=1e-9, aperture=None,
            c_s=0, theta_c=0, kV=200.0, plot=True):
    '''
    Numerical Fresnel image calculation from wavefunction amplitude and phase [1].
    `mag_phase` may be used to generate `phase' from a known magnetisation.
    
    [1] M. De Graef, Lorentz microscopy: Theoretical basis and image simulations, in Magnetic Imaging and Its Applications to Materials, 36, 27 (2001).
    https://doi.org/10.1016/S1079-4042(01)80036-9.
    
    Parameters
    ----------
    amp : ndarray
        Wavefunction amplitude.
    phase : ndarray
        Wavefunction phase in radians.
    df : scalar
        Defocus in m. Positive is underfocus. See Notes.
    ypix : scalar
        Y-axis pixel spacing in m.
    xpix : scalar
        X-axis pixel spacing in m.
    aperture : ndarray or None
        If not None, the aperture in the BFP.
    c_s : scalar
        Spherical aberration in m.
    theta_c : scalar
        Beam convergence in radians.
    kV : scalar
        Beam energy in kV. The wavelength calculation is relativistic.
    plot : bool
        If True, the main results are plotted.

    Returns
    -------
    fren : namedtuple
        In addition to all input parameters, this contains the following
        parameters:    
    im : ndarray
        Fresnel image.
    chi : ndarray
        Phase contrast transfer function.
    dp : ndarray
        Diffraction pattern image.
    qy, qx : ndarray
        1-D reciprocal space frequency vectors.
    qyy, qxx : ndarray
        2-D reciprocal space frequency vectors.
    tf_im : ndarray
        Transfer function image: np.sin(np.imag(transfer_function))
    
    Notes
    -----
    The electron may be considered to be traveling into the page / screen.
    When underfocus, the image is formed before the cross-over and the
    contrast matches that expected from the classical Lorentz force, and
    that from Huygens' principal.
    
    See also
    --------
    mag_phase, fresnel_paraxial, tie
    
    '''

    lamb = e_lambda(kV)
    
    # grab locals for inclusion in output
    input_dict = locals()
    
    if aperture is None:
        aperture = 1

    # wavefunction
    wf = amp * np.exp(1j * phase)
    
    # reciprocal vectors
    qy = fftfreq(phase.shape[0], ypix)
    qx = fftfreq(phase.shape[1], xpix)
    qyy, qxx = np.meshgrid(qy, qx, indexing='ij')
    qsq = (qxx**2 + qyy**2)

    # phase transfer function
    chi2 = (np.pi * lamb * df) * qsq
    chi4 = (np.pi/2.0 * c_s * lamb**3) * qsq**2
    chi = chi2 + chi4
    
    # damping
    g2 = ((np.pi * theta_c * df)**2 / np.log(2)) * qsq
    tf_F = aperture * np.exp(-1j * chi) * np.exp(-g2)
    
    # microscope transfer function
    tf_im = fftshift( np.sin(np.imag(tf_F)) )
    
    # psf
    wf_F = fft2(wf)
    wf_tr_F = wf_F * tf_F
    im = np.abs(ifft2(wf_tr_F))**2
    
    # diff pattern
    dp = np.abs(fftshift(wf_tr_F))**2
    
    # return
    fkeys = ['im', 'chi', 'dp', 'qy', 'qx', 'qyy', 'qxx', 'tf_im']
    ld = locals()
    fdict = {k: ld[k] for k in fkeys}
    fdict.update(input_dict)
    
    FresnelResults = namedtuple('FresnelResults', sorted(fdict))
    fren = FresnelResults(**fdict)
    
    if plot:
        # TODO: add y- and x-scales to plots
        plt.matshow(fren.im)
        plt.title('Fresnel image')
        plt.colorbar()
        
        plt.matshow(fren.tf_im)
        plt.title('sin(Im(tf))')
        plt.colorbar()
        
        plt.matshow(fren.dp, norm=plt.mpl.colors.LogNorm())
        plt.title('Diffraction pattern')
        plt.colorbar()
    
    return fren

def tie(im_grad, im0, ypix=1e-9, xpix=1e-9, kV=200.0):
    '''
    Transport of intensity Fresnel image analysis [1].
    
    [1] D. Paganin K. A. Nugent, Phys. Rev. Lett. 80(12), 2586 (1998)
    
    Parameters
    ----------
    im_grad : ndarray
        2-D image of image intensity gradient wrt z (defocus direction).
    im0 : ndarray
        Image intensity at zero defocus.
    ypix, xpix : scalar or None
        Pixel spacing along axes. If None, the pixel spacing is taken as 1.
    kV : scalar
        Beam energy in kV. The wavelength calculation is relativistic.

    Returns
    -------
    tie_phase : ndarray
        2-D array of recovered phase.
    
    See also
    --------
    mag_phase, fresnel_paraxial, fresnel

    '''
    from .fft_tools import fft2_ilaplacian, fft2_diff
    
    lamb = e_lambda(kV)
    k = 2*np.pi / lamb
    
    ilap = fft2_ilaplacian(im_grad, ypix=ypix, xpix=xpix)
    div = fft2_diff(ilap, ypix=ypix, xpix=xpix, order=1) / im0
    div2y = fft2_diff(div[0], ypix=ypix, xpix=xpix, order=1)[0]
    div2x = fft2_diff(div[1], ypix=ypix, xpix=xpix, order=1)[1]
    div2 = div2y + div2x
    ilap2 = fft2_ilaplacian(div2, ypix=ypix, xpix=xpix)
    
    tie_phase = - k * ilap2
    return tie_phase


def tem_dpc(dyx, df, thickness, pix_m, kV=200):
    '''
    Induction mapping using the phase recovered from Fresnel imaging
    using the TEM-DPC method. [1, 2]
    
    [1] Microsc. Microanal. 27, 1113 (2021) https://doi.org/10.1017/S1431927621012551
    [2] Microsc. Microanal. 27, 1123 (2021) https://doi.org/10.1017/S1431927621012575
    
    Parameters
    ----------
    dyx : ndarray
        Distortion field in pixels. 3D array of shape (2, Ny, Nx), with
        first axis containing [dy, dx].
    df : scalar
        Defocus in m. Positive is underfocus.
    thickness : scalar
        Sample thickness in m.
    pix_m : scalar
        Pixel size in m.
    kV : scalar
        Beam energy in kV. The wavelength calculation is relativistic.
    
    Returns
    -------
    Byx : ndarray
        The induction in Tesla, of the same shape as ``dyx``.
    
    See also
    --------
    fpd.DPC_Explorer, fpd.tem_tools.AlignNR
    
    '''
    
    from scipy.constants import h, e
    from .fpd_processing import rotate_vector
    
    lamb = e_lambda(kV)
    Byx = h / (lamb * e * thickness) * (dyx  * pix_m) / df
    Byx = rotate_vector(Byx, 90)
    
    return Byx

