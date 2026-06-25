import numpy as np
import scipy as sp

from scipy.ndimage import gaussian_filter1d, gaussian_filter
from skimage.transform import AffineTransform
from skimage.feature import peak_local_max
from skimage.morphology import erosion, disk
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import warnings

from collections.abc import Iterable

from .fpd_processing import rotate_vector
from .utils import gaussian_2d_peak_fit
from .AlignNR_class import AlignNR
from .fft_tools import im2fftrdf, fftcorrelate

import matplotlib.pylab as plt

plt.ion()

# cross-imports
from .synthetic_data import array_image


def lambda_iak(rho, alpha, beta, kV=200.0):
    """
    Inelastic mean free scattering length of electrons through a
    material with specific gravity rho using Iakoubovskii's method
    from 2008, as documented in p298 of Egerton (see refs).

    Parameters
    ----------
    rho : scalar
        Specific gravity in g/cm^3.
    alpha : scalar
        Convergence semi-angle in mrad.
    beta : scalar
        Collect semi-anfle in mrad.
    kV : scalar, optional
        Accelerating voltage in kV.

    Returns
    -------
    lambda : scalar
        Inelastic mean free path length in nm

    References
    ----------
    Iakoubovskii, K. , Mitsuishi, K. , Nakayama, Y. and Furuya, K. (2008)
    Thickness measurements with electron energy loss spectroscopy.
    Microsc. Res. Tech., 71: 626-631. doi:10.1002/jemt.20597

    R.F. Egerton, Electron Energy-Loss Spectroscopy in the Electron Microscope
    Springer (Third Edition)
    ISBN: 9781441995834 (online)
          9781441995827 (print)

    """

    F = (1.0 + kV / 1022.0) / (1 + kV / 511.0) ** 2
    d2 = np.abs(alpha**2 - beta**2)
    tc = 20.0
    te = 5.5 * rho**0.3 / (F * kV)
    a2 = alpha**2
    b2 = beta**2

    lnarg = (a2 + b2 + d2 + 2.0 * te**2) / (a2 + b2 + d2 + 2.0 * tc**2) * tc**2 / te**2
    lam = 200.0 * F * kV / (11.0 * rho**0.3) / np.log(lnarg)

    return lam


def e_lambda(kV=200.0, rel=True):
    """
    Electron wavelength at acceleration voltage kV.

    Returns wavelength in metres.

    Parameters
    ----------
    kV : scalar
        Accelerating voltage in kV.
    rel : bool, optional
        If True, use relativistic calculation, else use classical.

    Returns
    -------
    Wavelength : scalar
        Electron wavelength in meters.

    """

    me = sp.constants.m_e
    h = sp.constants.h
    e = sp.constants.e
    c = sp.constants.c

    V = kV * 1000.0

    if rel:
        lam = h / np.sqrt(2 * me * e * V * (1 + e * V / (2 * me * c**2)))
    else:
        lam = h / np.sqrt(2 * me * e * V)
    return lam


def airy_d(alpha, kV=200.0, rel=True):
    """
    Airy spot diameter for aberration free conditions.

    Parameters
    ----------
    alpha : scalar
        Semi-convergence angle of limiting aperture, in radians.
    kV : scalar
        Accelerating voltage in kV.
    rel : bool, optional
        If True, use relativistic calculation, else use classical.

    Returns
    -------
    d : scalar
        Diameter of spot in meters.

    """

    lam = e_lambda(kV, rel)

    d = 1.22 * lam / np.sin(alpha)

    return d


def airy_fwhm(alpha, kV=200.0, rel=True):
    """
    Airy spot FWHM for aberration free conditions.

    Parameters
    ----------
    alpha : scalar
        Semi-convergence angle of limiting aperture, in radians.
    kV : scalar
        Accelerating voltage in kV.
    rel : bool, optional
        If True, use relativistic calculation, else use classical.

    Returns
    -------
    fwhm : scalar
        FWHM of spot in meters.

    Notes
    -----
    The FWHM is an approximation of a Gaussian fit to the central lobe of the Airy disk.

    D. Thomann et al., J Microsc. 208, 49 (2002). doi:10.1046/j.1365-2818.2002.01066.x

    """

    d = airy_d(alpha, kV, rel)
    sigma = 0.21 / 0.61 * d / 2.0
    fwhm = sigma * 2 * (2 * np.log(2)) ** 0.5
    return fwhm


def d_from_two_theta(two_theta, kV=200.0):
    """
    d-spacing from deflection angle of Bragg scattered electron
    accelerated through voltage kV.

    Parameters
    ----------
    two_theta : scalar
        Deflection in mrad. The angle to the undeflected spot.
    kV : scalar, optional
        Electron accelerating voltage in kilovolts.

    Returns
    -------
    Returns d-spacing in nm.

    """

    theta = np.asarray(two_theta, float) / 2.0 / 1000
    lam_nm = e_lambda(kV, rel=True) * 1e9
    d = lam_nm / (2.0 * np.sin(theta))

    return d


def hkl_cube(alpha, n=10, kV=200.0, struct=None, print_first=10):
    """
    Compute diffraction parameters for a cubic lattice.

    Returns unique hkl values, d-spacing and deflection angles.

    Parameters
    ----------
    alpha : scalar
        Cube edge length in nm.
    n : integer, optional
        Number of values of each hkl to consider.
    kV : scalar, optional
        Electron accelerating voltage in kV.
    struct : None or string, optional
        If not None, a string controlling structure of the cell.
        Only 'fcc' and 'bcc' are currently implemented. If None, all
        reflections are returned.
    print_first : int
        Controls the printing of the calculated results. If non-zero, the first
        `print_first` lines are printed.

    Returns
    -------
    Tuple of hkl, d, bragg_2t_mrad, p
    hkl : ndarray
        Sorted hkl of unique d-spacing.
    d : ndarray
        Sorted d-spacing in nm.
    bragg_2t_mrad: ndarray
        Deflection from the direct (undeviated) spot in mrad.
    p : ndarray
        Plurality of reflection.

    Notes
    -----
    Example FCC and BCC structures are Au (0.408 nm) and alpha-Fe (0.287 nm).

    """

    import itertools

    hkl = np.asarray(
        list(itertools.product(list(range(n)), list(range(n)), list(range(n))))
    )[1:, :]

    if struct is None:
        # No filtering
        pass
    elif struct == "fcc":
        # fcc H,K,L all odd or all even
        n_odd = (hkl & 0x1).sum(-1)  # number of odd
        fcc_i = np.where(np.logical_or(n_odd == 3, n_odd == 0))[0]
        hkl = hkl[fcc_i, :]  # allowed
    elif struct == "bcc":
        # bcc H + K + L even
        all_even = np.invert(hkl.sum(-1) & 0x1)
        bcc_i = np.where(all_even)[0]
        hkl = hkl[bcc_i, :]  # allowed
    else:
        print("Structure not supported, returning all reflections.")

    hkl2 = (hkl**2).sum(-1)  # sumsq
    hkl2_si = np.argsort(hkl2)  # index of increasing size
    hkl2s = hkl2[hkl2_si]  # sorted sumsq
    hkls = hkl[hkl2_si, :]  # sorted hkl
    hkl2s_u, hkl2s_i, p = np.unique(hkl2s, return_index=True, return_counts=True)

    # unique reflection spacing
    d = alpha / hkl2s_u**0.5
    elam_nm = e_lambda(kV, rel=True) * 1e9
    # mrad of angle from undeviated
    bragg_2t_mrad = 2 * np.arcsin(elam_nm / (2 * d)) * 1000

    hkl = hkls[hkl2s_i]

    if print_first:
        print_first = min(print_first, len(hkl))
        print("Structure: %s (alpha = %0.3f nm)" % (struct, alpha))
        print("hkl      d (nm)  mrad    Plurality")
        print("----------------------------------")
        for i in range(print_first):
            print(hkl[i], " %0.4f  %6.3f  %02d" % (d[i], bragg_2t_mrad[i], p[i]))

    return hkl, d, bragg_2t_mrad, p


def rutherford_cs(z, mrad=None, kV=200.0, plot=False):
    """
    Relativistic screened rutherford differential cross-section, following
    equation 3.6 in chapter 3 of Williams and Carter.

    Parameters
    ----------
    z : 1-D iterable or scalar
        Element z-numbers.
    mrad : 1-D array or None
        Scattering angles used in the calculation. If None,
        mrad = np.linspace(0, 100, 1000).
    kV : scalar
        Electron acceleration voltage in kV.
    plot : bool
        If True, the results are plotted.

    Returns
    -------
    tuple of dif_cs, mrad
    dif_cs :
        Relativistically corrected rutherford differential cross-section in barns/rad.
        A barn is 1e-28 m**2.
    mrad : 1-D array
        See parameters.

    Examples
    --------
    Calculate cross-section for Ne, Al, and Fe.

    >>> import fpd
    >>> import matplotlib.pyplot as plt
    >>> plt.ion()

    >>> z = [10, 13, 26]
    >>> leg_txt = 'Ne Al Fe'.split()
    >>>
    >>> dif_cs, mrad = fpd.tem_tools.rutherford_cs(z)

    >>> f, ax = plt.subplots(figsize=(4,3))
    >>> ax.loglog(mrad, dif_cs.T)
    >>> ax.set_xlabel('Angle (mrad)')
    >>> ax.set_ylabel('Differential cross-section (barns/rad)')
    >>> ax.legend(leg_txt, fontsize=10, loc=0)
    >>> plt.tight_layout()


    """

    import scipy.constants as sc

    e = sc.e
    h = sc.h
    me = sc.m_e
    ep0 = sc.epsilon_0
    pi = sp.pi
    c = sc.c

    # condition inputs
    V = kV * 1e3

    if mrad is None:
        mrad = np.linspace(0, 100, 1000)
    rad = mrad / 1000.0

    if not isinstance(z, Iterable):
        z = [z]
    Z = np.array(z)[:, None]

    # relativistic screened calculation from W&C
    lam = e_lambda(kV, rel=True)
    a0 = h**2 * ep0 / (pi * me * e**2)  # (3.5)
    t0 = 0.117 * Z ** (1.0 / 3) / (kV) ** 0.5  # (3.4)

    dif_cs_dA = (Z * lam**2 / (8.0 * pi**2 * a0)) ** 2 / (
        np.sin(rad / 2.0) ** 2 + (t0 / 2.0) ** 2
    ) ** 2  # (3.6)
    dif_cs = dif_cs_dA * 2 * pi * np.sin(rad)
    # convert to barns / rad
    dif_cs /= 1e-28

    if plot:
        import matplotlib.pyplot as plt

        plt.ion()

        f, ax = plt.subplots(figsize=(4, 3))
        ax.loglog(mrad, dif_cs.T)
        ax.set_xlabel("Angle (mrad)")
        ax.set_ylabel("Differential cross-section (barns/rad)")
        leg_txt = [str(zi) for zi in Z]
        ax.legend(leg_txt, fontsize=10, loc=0)
        plt.tight_layout()

    return dif_cs, mrad


def defocus_from_ctf_crossing(k, kV=200.0, Cs=0.0):
    """
    Defocus from CTF first crossing.

    Parameters
    ----------
    k : scalar
        Spatial frequency of first minimum (1/m).
    kV : scalar
        Accelerating voltage in kV.
    Cs : scalar
        Spherical aberration coefficient in m.

    Returns
    -------
    df : scalar
        Defocus in meters. Positive is underfocus.

    Notes
    -----
    Cs for the Jeol ARM200CF is 0.5mm. [1]

    [1] C. Ricolleau et al, High Resolution Imaging and Spectroscopy Using CS-Corrected TEM with Cold FEG JEM-ARM200F. JEOL News, 2012, 47, 2-8.

    Examples
    --------
    >>> from fpd.tem_tools import defocus_from_ctf_crossing
    >>> df = defocus_from_ctf_crossing(4.745/1e-9, kV=200.0, Cs=0.0005)
    """

    lam = e_lambda(kV=kV, rel=True)
    if Cs == 0:
        n = 1
    else:
        n = 0
    df = -n / (lam * k**2) + Cs * (lam * k) ** 2 / 2.0
    return df


def defocus_from_image(
    image, pix_m=1, n_min=5, f_max=0.5, sigma=1, kV=200.0, plot=True
):
    """
    Automatically determine defocus from an image through its CTF.
    Assumes the effect of Cs is insignificant.

    Parameters
    ----------
    image : ndarray
        2-D image to analyse.
    pix_m : scalar
        Pixel spacing in metres.
    n_min : integer
        Number of minima to characterise, >= 2.
    f_max : scalar
        Maximum fraction of Nyquist frequency to analyse, in [0, 1].
    sigma : scalar
        Width of Gaussian smoothing applied to the contrast metric.
    kV : scalar
        Accelerating voltage in kV.
    plot : bool
        If True, the analysis results are plotted.

    Returns
    -------
    defocus : scalar
        Absolute value of defocus in metres.

    See also
    --------
    fpd.tem_tools.defocus_from_ctf_crossing
    fpd.fft_tools.im2fftrdf

    """

    """
    TODO
    - could add a Cs loop, then fit 2D Gaussians to peaks in space?
    - fit 1D Gaussian to get subpixel peak location?
    """

    if n_min < 2:
        raise Exception("`n_min` must be >= 2")

    # calculate azimuthally average fft
    r, rm = im2fftrdf(image - image.mean())

    # select freq range
    b = r <= 0.5 * f_max
    b[0] = False
    r_sel = r[b]
    rm_sel = rm[b]

    # de-trend
    x = r_sel
    y = np.log10(rm_sel)
    pfit = np.polyfit(x, y, 2)
    yfit = np.polyval(pfit, x)
    y = y / yfit

    # loop over frequencies, calculating mean over contrast of each peak / trough
    ns = np.arange(n_min + 1) + 1
    ys = []
    for fi in x:
        x_mins = ns**0.5 * fi
        x_maxs = (x_mins[0:-1] + x_mins[1:]) / 2.0
        x_mins = x_mins[:-1]
        y_mins = np.interp(x_mins, x, y, left=np.nan, right=np.nan)
        y_maxs = np.interp(x_maxs, x, y, left=np.nan, right=np.nan)

        yi = (y_maxs - y_mins) / (y_maxs + y_mins)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # this can be all nans
            yi = np.nanmean(yi)
        if np.isnan(yi):
            yi = 0
        ys.append(yi)
    ys = np.array(ys)
    if sigma != 0:
        ys = gaussian_filter1d(ys, sigma)

    f_min = x[np.nanargmax(ys)]
    k = f_min / pix_m
    defocus = defocus_from_ctf_crossing(k, kV=kV, Cs=0.0)
    defocus = np.abs(defocus)

    if plot:
        dx = (x[1] - x[0]) / 2
        fig, axs = plt.subplots(2, 1, sharex=True)
        ax1, ax2 = axs

        ax1.plot(x, ys)
        ax1.set_ylabel("Contrast")
        ax1.axvline(f_min, ls="--", color="r")
        ax1.set_title(
            r"f_min = %0.4f $\pm$ %0.4f 1/pix, |$\Delta f$| = %0.3e m"
            % (f_min, dx, defocus)
        )
        ax1.set_ylim(0)

        ax2.semilogy(r_sel, rm_sel)
        ax2.set_xlabel("1/pix")
        ax2.set_ylabel("|fft|")
        ns = np.arange(n_min) + 1
        x_mins = ns**0.5 * f_min
        for xi in x_mins:
            ax2.axvline(xi, ls="--", color="r")

        plt.tight_layout()

    return defocus


def ctf(k, df=0.0, kV=200.0, Cs=0.0, plot=True):
    """
    Contrast transfer function.

    Parameters
    ----------
    k : 1-D array
        Spatial frequency (1/m).
    df : scalar
        Defocus in m. Positive is underfocus.
    kV : scalar
        Accelerating voltage in kV.
    Cs : scalar
        Spherical aberration coefficient in m.

    Returns
    -------
    c : 1-D array
        Contrast transfer function.

    Notes
    -----
    Cs for the Jeol ARM200CF is 0.5mm. [1]

    [1] C. Ricolleau et al, High Resolution Imaging and Spectroscopy Using CS-Corrected TEM with Cold FEG JEM-ARM200F. JEOL News, 2012, 47, 2-8.

    Examples
    --------
    >>> import numpy as np
    >>> from fpd.tem_tools import ctf
    >>> c = ctf(k=np.linspace(0, 8, 1000)/1e-9, df=35.4e-9, kV=200.0, Cs=0.0005, plot=True)

    """

    lam = e_lambda(kV=kV, rel=True)
    arg = (-df * lam * k**2 + Cs * lam**3 * k**4 / 2.0) * np.pi
    c = np.sin(arg)

    if plot:
        import matplotlib.pyplot as plt

        plt.ion()

        f, ax = plt.subplots(figsize=(4, 3))
        ax.plot(k * 1e-9, c)
        ax.set_xlabel("k (1/nm)")
        ax.set_ylabel("CTF")
        leg_txt = "df = %0.1fnm\nCs = %0.1fmm" % (df / 1e-9, Cs / 1e-3)
        ax.legend([leg_txt], fontsize=8, loc=2)
        plt.axhline(y=0, xmin=0, xmax=1, color="k", alpha=0.5, zorder=-20)
        plt.tight_layout()
    return c


def scherzer_defocus(Cs=0.0, extended=False, kV=200.0):
    """
    Scherzer defocus.

    Parameters
    ----------
    Cs : scalar
        Spherical aberration coefficient in m.
    extended : bool
        If True, the extended defocus is returned.
    kV : scalar
        Accelerating voltage in kV.

    Returns
    -------
    df_s : scalar
        Defocus in m. Positive is underfocus.

    """

    lam = e_lambda(kV=kV, rel=True)
    df_s = (lam * Cs) ** 0.5
    if extended:
        df_s *= 1.2
    return df_s


from skimage.transform import ProjectiveTransform


class TranslationTransform(ProjectiveTransform):
    """2D Translation transformation.

    Has the following form::

        X = x + a1
        Y = y + b1

    where the homogeneous transformation matrix is::

        [[1  0  a1]
         [0  1  b1]
         [0  0   1]]

    The Translation transformation is a rigid transformation with
    translation parameters. The Euclidean transformation extends the Translation
    transformation with rotation.

    Parameters
    ----------
    matrix : (3, 3) array, optional
        Homogeneous transformation matrix.
    translation : (tx, ty) as array, list or tuple, optional
        x, y translation parameters.
    method : str
        Method used for estimation. One of ['median', 'mean'].

    Attributes
    ----------
    params : (3, 3) array
        Homogeneous transformation matrix.

    """

    def __init__(self, matrix=None, translation=None, method="median"):
        params = any(param is not None for param in (translation,))

        if params and matrix is not None:
            raise ValueError(
                "You cannot specify the transformation matrix and"
                " the implicit parameters at the same time."
            )
        elif matrix is not None:
            if matrix.shape != (3, 3):
                raise ValueError("Invalid shape of transformation matrix.")
            self.params = matrix
        elif params:
            if translation is None:
                translation = (0, 0)

            self.params = np.eye(3)
            self.params[0:2, 2] = translation
        else:
            # default to an identity transform
            self.params = np.eye(3)

        methods = ["median", "mean"]
        if method not in methods:
            raise Exception("'method' must be on of:", methods)
        self.method = method

    def estimate(self, src, dst):
        """Estimate the transformation from a set of corresponding points.

        You can determine the over-, well- and under-determined parameters
        with the total least-squares method.

        Number of source and destination coordinates must match.

        Parameters
        ----------
        src : (N, 2) array
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """

        self.params = np.eye(3)

        if self.method == "median":
            translation = np.median(dst - src, axis=0)
        elif self.method == "mean":
            translation = np.mean(dst - src, axis=0)

        self.params[0:2, 2] = translation

        return True

    @property
    def translation(self):
        return self.params[0:2, 2]


def orb_trans(
    im_ref,
    im_new,
    pct=0.1,
    fminmax=None,
    gy=2,
    gaus=None,
    gaus_der=True,
    roi_s=None,
    orb_kwd={},
    ransac_trans="euclidean",
    trans="affine",
    min_samples=5,
    residual_threshold=1,
    max_trials=1000,
    plot=False,
    optimise=True,
):
    """
    Image transform RANSAC fit of matching ORB features.

    Parameters
    ----------
    im_ref : 2-D array
        Reference image.
    im_new : 2-D array
        Image to be transformed.
    pct : float
        Percentile to clip images to during conditioning.
    fminmax : None or length 2 tuple
        If not None, the input images are FFT filtered, with frequency
        limits defined here, in reciprocal pixels. The bandpass mask is
        circular. See fpd.fft_tools.bandpass for details.
    gy : float
        The width of the Gaussian smoothing applied to the bandpass mask.
    gaus : float or None
        If not None, the images are smoothed using a Gaussian filter.
        See also `gaus_der`.
    gaus_der : bool
        If True, and `gaus` is True, the Gaussian filter calculated the
        image derivative.
    roi_s : None or tuple of slices
        If not None, the keypoints are restricted to this area. See Notes.
    orb_kwd : dictionary
        Keyword dictionary of parameters passed to the ORB feature detector.
        See skimage.feature.ORB for details. See Notes.
    ransac_trans : string
        One of ['translation', euclidean', 'similarity', 'affine']. The transform used
        for the RANSAC fitting.  See notes for details.
    trans : One of ['translation', 'euclidean', 'similarity', 'affine']. The transform used
        for the final fit using the RANSAC inliers. See notes for details.
    min_samples : int
        The minimum samples used in the RANSAC fit.
    residual_threshold : float
        The residual threshold used in the RANSAC fit. Higher numbers
        accepts more data points as inliers.
    max_trials : int
        The maximum number of trials of the RANSAC fit.
    plot : bool
        If True, the detected keypoints showing the matches and the inlier
        and outlier matches are shown.

    Returns
    -------
    model : skimage.transform._geometric
        The optimised transform.

    Notes
    -----
    The Translation transform includes translation.
    The Euclidean transform includes the above and adds rotation.
    The Similarity transform includes the above and adds scaling.
    The Affine transform includes the above and adds shear.

    It is sometimes useful to specify a lower order transform for the
    RANSAC (`ransac_trans`) than in image transform (`trans`).

    ROI may be set with, for example, roi_s=np.s_[-400:, :]. This will
    restrict the used points to the region from 400 pixels from the end in y
    and use all points in x.

    The ROI is applied after the feature extraction. If no features are extracted,
    consider increasing the maximum number of keypoints returned with, for example:
    orb_kwd={'n_keypoints':1000}. At the time of writing, the default n_keypoints
    is 500.

    The returned transform may be used to transform the image with `warp`:

    from skimage.transform import warp
    im_unw = warp(im_new, model, preserve_range=True)

    """

    from skimage.measure import ransac
    from skimage.transform import (
        AffineTransform,
        SimilarityTransform,
        EuclideanTransform,
    )
    from scipy.ndimage.filters import gaussian_filter
    from skimage.feature import match_descriptors, ORB, plot_matches
    from .fft_tools import bandpass

    def condition_im(im, pct, gaus, gaus_der):
        if gaus is None:
            gaus = 0

        g_order = 0
        if gaus_der:
            g_order = 1

        if gaus:
            im = gaussian_filter(im, gaus, g_order)
        vmin, vmax = np.percentile(im, [pct, 100 - pct])
        im = (im - vmin) / (vmax - vmin)
        im = (im.clip(0, 1) * 255).astype(np.uint8, copy=False)
        return im

    # fft filter
    if fminmax is not None:
        ims = np.concatenate([im_ref[None, ...], im_new[None, ...]])
        ims2 = np.concatenate([ims, ims[..., ::-1, :]], -2)
        ims2 = np.concatenate([ims2, ims2[..., ::-1]], -1)
        im_pass, _ = bandpass(
            ims2, fminmax=fminmax, gy=gy, mask=None, full_out=False, mode="circ"
        )
        ims_pass = im_pass[..., : ims.shape[-2], : ims.shape[-1]]
        ims_flt = ims - ims_pass
        im_ref, im_new = ims_flt
        del ims, ims2, ims_pass, ims_flt

    # condition input images
    im_ref = condition_im(im_ref, pct, gaus, gaus_der)
    im_new = condition_im(im_new, pct, gaus, gaus_der)

    # extract features
    descriptor_extractor = ORB(**orb_kwd)

    descriptor_extractor.detect_and_extract(im_ref)
    keypoints_left = descriptor_extractor.keypoints
    descriptors_left = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(im_new)
    keypoints_right = descriptor_extractor.keypoints
    descriptors_right = descriptor_extractor.descriptors

    # filter for slice
    if roi_s is not None:
        # condition input
        yminmaxs = []
        for ii, si in zip(im_ref.shape, roi_s):
            start, stop = si.start, si.stop
            if start is None:
                start = 0
            elif start < 0:
                start = ii + start

            if stop is None:
                stop = ii
            elif stop < 0:
                stop = ii + stop

            yminmax = (start, stop)
            yminmaxs.append(yminmax)

        # determine if points lie within box
        from functools import reduce

        keypoints_inrois = []
        for keypoints in [keypoints_left, keypoints_right]:
            by1 = keypoints[:, 0] >= yminmaxs[0][0]
            by2 = keypoints[:, 0] <= yminmaxs[0][1]
            bx1 = keypoints[:, 1] >= yminmaxs[1][0]
            bx2 = keypoints[:, 1] <= yminmaxs[1][1]
            keypoints_inroi = reduce(np.logical_and, [by1, by2, bx1, bx2])
            keypoints_inrois.append(keypoints_inroi)
        kp_left_inroi, kp_right_inroi = keypoints_inrois

        keypoints_left = keypoints_left[kp_left_inroi]
        descriptors_left = descriptors_left[kp_left_inroi]

        keypoints_right = keypoints_right[kp_right_inroi]
        descriptors_right = descriptors_right[kp_right_inroi]

    # match descriptors
    matches = match_descriptors(descriptors_left, descriptors_right, cross_check=True)

    # ransac inlier detection
    trans_dict = {
        "translation": TranslationTransform,
        "euclidean": EuclideanTransform,
        "similarity": SimilarityTransform,
        "affine": AffineTransform,
    }

    ransac_trans = trans_dict[ransac_trans.lower()]
    model_ransac, inliers = ransac(
        (keypoints_left[matches[:, 0]], keypoints_right[matches[:, 1]]),
        ransac_trans,
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        max_trials=max_trials,
    )
    outliers = inliers == False

    inlier_keypoints_left = keypoints_left[matches[inliers, 0]]
    inlier_keypoints_right = keypoints_right[matches[inliers, 1]]
    outlier_keypoints_left = keypoints_left[matches[outliers, 0]]
    outlier_keypoints_right = keypoints_right[matches[outliers, 1]]

    print(
        "Number of keypoints (left, right):", keypoints_right.size, keypoints_left.size
    )
    print("Number of matches:", matches.shape[0])
    print("Number of inliers:", inliers.sum())

    model = trans_dict[trans.lower()]()
    model.estimate(inlier_keypoints_left[:, ::-1], inlier_keypoints_right[:, ::-1])

    # visualize correspondence
    if plot:
        # plot matches
        import matplotlib.pyplot as plt

        plt.ion()

        fig, ax = plt.subplots(nrows=2, ncols=1)
        ax[0].matshow(im_ref)
        ax[0].axis("off")
        ax[0].scatter(
            keypoints_left[:, 1],
            keypoints_left[:, 0],
            facecolors="none",
            edgecolors="b",
            alpha=0.5,
        )
        ax[0].scatter(
            keypoints_left[matches[:, 0], 1],
            keypoints_left[matches[:, 0], 0],
            facecolors="none",
            edgecolors="r",
        )
        ax[0].set_title("%d keypoints" % (len(keypoints_left)))

        ax[1].matshow(im_new)
        ax[1].axis("off")
        ax[1].scatter(
            keypoints_right[:, 1],
            keypoints_right[:, 0],
            facecolors="none",
            edgecolors="b",
            alpha=0.5,
        )
        ax[1].scatter(
            keypoints_right[matches[:, 1], 1],
            keypoints_right[matches[:, 1], 0],
            facecolors="none",
            edgecolors="r",
        )
        ax[1].set_title("%d keypoints" % (len(keypoints_right)))

        # plot keypoints
        fig, ax = plt.subplots(nrows=2, ncols=1)
        plot_matches(
            ax[0],
            im_ref,
            im_new,
            keypoints_left,
            keypoints_right,
            matches[inliers],
            matches_color="b",
        )
        ax[0].axis("off")
        ax[0].set_title("Correct correspondences: %d" % (inliers.sum()))

        plot_matches(
            ax[1],
            im_ref,
            im_new,
            keypoints_left,
            keypoints_right,
            matches[outliers],
            matches_color="r",
        )
        ax[1].axis("off")
        ax[1].set_title("Faulty correspondences: %d" % (outliers.sum()))

    if optimise:
        print("")
        model = optimise_trans(im_ref, im_new, model, roi_s=roi_s, print_stats=True)

    return model


def apply_image_trans(ims, trans, extra_trans=None, output_mode="same", cval=0):
    """
    Apply transformations to images.

    Parameters
    ----------
    ims : 2-D or 3-D array
        Array of 2-D images with image dimensions in the last axes. If ims.ndim
        is 2, an additional axis is added.
    trans : transform or 1-D array of transforms
        Transforms to be applied.
    output_mode : string
        One of ['same', 'expand', 'overlap']. Controls how the transformed
        images are returned if multiple images are being transformed.
    extra_trans : None or skimage.transform._geometric
        If not None, an additional transform that is applied after alignment.
    cval : sequence or int, optional
        The values to set the padded or missing values to if `output_mode` is
        'expand' or 'same'.

    Returns
    -------
    im_trans : 3-D array
        Array of transformed images.

    Notes
    -----
    Note that any `extra_trans` is applied to the images in the normal mathematical
    way. For simple rotations, additional transform can be applied to shift the image
    centre to (0, 0) and back again to achieve the commonly 'expected' results.

    """

    from skimage.transform import EuclideanTransform, AffineTransform
    from skimage.transform import warp

    # condition input
    multi = True
    ims = np.array(ims)
    if ims.ndim == 2:
        ims = ims[None]
        multi = False
    if not isinstance(trans, Iterable):
        trans = [trans]
    assert len(ims) == len(trans)

    if extra_trans is not None:
        trans = [
            AffineTransform(transi.params.dot(extra_trans.params)) for transi in trans
        ]

    output_mode = output_mode.lower()

    def get_pad_crop_arrays(ims, trans):
        h, w = ims.shape[-2:]
        coords = np.array([[0, 0], [w, 0], [w, h], [0, h]])
        ctr = np.array([m(coords) for m in trans])  # transform, coords, [x, y]
        ctr = (ctr - coords[None, ...]) * np.array(
            [[1, 1], [-1, 1], [-1, -1], [1, -1]]
        )[None, ...]
        ctr = np.rollaxis(ctr, 1, 0)  # coords, transform, [x, y]
        # print(ctr)
        # ctr is now how much must be added at each coordinate
        pa = ctr.max(1)
        ca = ctr.min(1)
        # print(ca)

        """
        pa and ca are:
        l, t
        r, t
        r, b
        l, b
        these could be used for image rotation to maximise area
        instead, we won't rotate
        """
        pl = np.max(pa[[0, 3], 0])
        pr = np.max(pa[[1, 2], 0])
        pt = np.max(pa[[0, 1], 1])
        pb = np.max(pa[[2, 3], 1])

        pa = np.array([pt, pb, pl, pr])
        pa = np.ceil(pa).astype(int)
        # print(pa)

        cl = np.min(ca[[0, 3], 0])
        cr = np.min(ca[[1, 2], 0])
        ct = np.min(ca[[0, 1], 1])
        cb = np.min(ca[[2, 3], 1])

        ca = np.array([ct, cb - 1, cl, cr - 1])  # minus 1 for -ve indexing
        ca = -np.ceil(-ca).astype(int)  # minus 1 for interpolation
        # print(ca)
        return pa, ca

    if multi is False:
        im_trans = [
            warp(im_new, transi, preserve_range=True, mode="constant", cval=cval)
            for (im_new, transi) in zip(ims, trans)
        ]
        im_trans = np.array(im_trans)
    else:
        if output_mode == "expand":
            pa, ca = get_pad_crop_arrays(ims, trans)

            ims = np.pad(
                ims,
                ((0, 0), tuple(pa[:2]), tuple(pa[2:])),
                mode="constant",
                constant_values=cval,
            )
            dy, dx = pa[0], pa[2]
            shift1 = EuclideanTransform(translation=(dx, dy))
            shift2 = EuclideanTransform(translation=(-dx, -dy))

            trans = [
                AffineTransform(shift1.params.dot(transi.params).dot(shift2.params))
                for transi in trans
            ]
            im_trans = [
                warp(im_new, transi, preserve_range=True, mode="constant", cval=cval)
                for (im_new, transi) in zip(ims, trans)
            ]
            im_trans = np.array(im_trans)

        elif output_mode == "same" or output_mode == "overlap":
            im_trans = [
                warp(im_new, transi, preserve_range=True, mode="constant", cval=cval)
                for (im_new, transi) in zip(ims, trans)
            ]
            im_trans = np.array(im_trans)

            if output_mode == "overlap":
                pa, ca = get_pad_crop_arrays(ims, trans)
                im_trans = im_trans[:, -ca[0] : ca[1], -ca[2] : ca[3]]

    if multi == False:
        im_trans = im_trans[0]
    return im_trans


def optimise_trans(im_ref, imw, trans, roi_s=None, print_stats=True):
    """
    Optimise an image transform through the nrmse.

    Parameters
    ----------
    im_ref : 2-D array
        The reference image.
    imw : 2-D array
        The warped image.
    trans : skimage.transform._geometric
        Transform to be optimised.
    roi_s : None or tuple of slices
        If not None, the nrmse is calculated only from the roi.
    print_stats : bool
        If True, statistics on the nrmse are printed.

    Returns
    -------
    trans_opt : skimage.transform._geometric
        The optimised transform.

    """

    from skimage.transform import AffineTransform
    from scipy.optimize import minimize
    from functools import reduce
    from .fpd_processing import nrmse

    # condition roi_s
    if roi_s is None:
        roi_s = np.s_[:, :]

    # get transform attributes
    ats = ["scale", "rotation", "shear", "translation"]
    vals = [getattr(trans, at, None) for at in ats]

    if vals[0] is not None:
        if not isinstance(vals[0], Iterable) or len(vals[0]) == 1:
            # always have length 2 scale (similarity has length 1)
            vals[0] = (vals[0],) * 2
    vals_df = [np.array([1, 1]), 0, 0, np.array([0, 0])]
    val_lens = [len(np.atleast_1d(vi)) for vi in vals_df]

    # dicts of params and defaults and lengths
    trd_df = dict(zip(ats, vals_df))
    trd = dict(zip(ats, vals))
    val_lensd = dict(zip(ats, val_lens))

    # dicts of free and non-free
    trd_nf = dict([(k, v) for (k, v) in trd.items() if v is None])
    trd_f = dict([(k, v) for (k, v) in trd.items() if v is not None])

    # replace non-free vals w/ defaults
    for k in trd_nf.keys():
        trd_nf[k] = trd_df[k]

    def regen_free_dict(x, trd_f):
        x0_lens = [val_lensd[k] for k in trd_f.keys()]
        x0is = np.cumsum([0] + x0_lens)
        x0_sl = [np.s_[a:b] for (a, b) in zip(x0is[:-1], x0is[1:])]

        trd_frg = dict(zip(trd_f.keys(), [np.array(x[s]) for s in x0_sl]))
        return trd_frg

    def gen_trans_dict(trd_f, trd_nf):
        # trd_all = {**trd_f, **trd_nf}
        trd_all = trd_f.copy()
        trd_all.update(trd_nf)
        return trd_all

    def params_mat(scale=None, rotation=None, shear=None, translation=None):
        if scale is None:
            scale = (1, 1)
        if rotation is None:
            rotation = 0
        if shear is None:
            shear = 0
        if translation is None:
            translation = (0, 0)

        import math

        sx, sy = scale
        tx, ty = translation
        params = np.array(
            [
                [sx * math.cos(rotation), -sy * math.sin(rotation + shear), tx],
                [sx * math.sin(rotation), sy * math.cos(rotation + shear), ty],
                [0, 0, 1],
            ]
        )
        return params

    def min_fun(x, im_ref, imw, roi_s, trd_nf, trd_f):
        trd_frg = regen_free_dict(x, trd_f)
        trd_all = gen_trans_dict(trd_frg, trd_nf)
        params = params_mat(**trd_all)

        transi = AffineTransform(params)
        imuw = apply_image_trans(imw, transi)
        err = nrmse(im_ref[roi_s], imuw[roi_s])

        return err

    # generate x0
    x0 = reduce(lambda a, b: a + b, [np.atleast_1d(v).tolist() for v in trd_f.values()])

    # optimise
    m = minimize(
        min_fun, x0, args=(im_ref, imw, roi_s, trd_nf, trd_f), method="BFGS", tol=0.1
    )

    # generate trans
    x_opt = m.x
    trd_frg = regen_free_dict(x_opt, trd_f)
    trd_all = gen_trans_dict(trd_frg, trd_nf)
    params = params_mat(**trd_all)
    trans_opt = type(trans)(params)

    # print stats
    if print_stats:
        imuw = apply_image_trans(imw, trans)
        imuw_opt = apply_image_trans(imw, trans_opt)
        errs = nrmse(
            im_ref[roi_s],
            np.concatenate(
                [imw[roi_s][None], imuw[roi_s][None], imuw_opt[roi_s][None]], 0
            ),
        )
        print(
            "nrmse\n-----\ninitial:   %0.4f\nunwarped:  %0.4f\noptimised: %0.4f"
            % tuple(errs)
        )

    return trans_opt


def blob_log_detect(
    image,
    min_radius,
    max_radius,
    num_radii,
    threshold,
    overlap=0.3,
    log_scale=True,
    ref_im=None,
    sigma=2.0,
    phase_exponent=0,
    log_xc_max_r=None,
    subpix_log=False,
    subpix_xc=False,
    subpix_dict=None,
    fit_hw=2,
    plot=False,
):
    """
    Detect blobs in an image using the Laplacian of Gaussian optionally
    combined with edge filtered cross-correlation.

    Parameters
    ----------
    image : ndarray
        2-D image to process (assumed non-negative).
    min_radius : scalar
        Minimum radius to detect.
    max_radius : scalar
        Maximum radius to detect.
    num_radii : int
        Number of radii between `min_radius` and `max_radius` to use.
    threshold : scalar
        Minimum normalised intensity of `image` to detect [0, 1].
    overlap : scalar
        If detected blobs overlap in area more than this, the smaller
        one is removed. Set to 1 for no removal.
    log_scale : bool
        If True, intermediate radii are set on a log10 scale. Otherwise,
        linear scaling is used.
    ref_im : None or ndarray
        If not None, a 2-D array used for correlation of images. This image can be
        smaller than the data being processed, but it should be centred for locations
        of peaks in the correlation to match the position in `image`. Otherwise, the
        offset of the peak in the correlation image from the centre of the image
        indicates the displacement between images. See `sigma` for details on how
        `ref_im` and `image` are processed.
    sigma : scalar
        Controls how `ref_im` and `image` are processed. If positive, it is the standard
        deviation of the Gaussian used for image derivatives of `ref_im` and `image`. If
        set to 0, both `ref_im` and `image` are passed through unaffected. If negative,
        `ref_im` is modified by subtracting from itself `ref_im` smoothed by a Gaussian
        of this standard deviation.
    phase_exponent : scalar in [0, 1]
        Exponent used in the normalisation of the correlation.
        0 : cross-correlation
        1 : phase-correlation
    log_xc_max_r : None or scalar
        The maximum distance between log and cross-correlation coordinated for the
        cross-correlation peaks to be kept. If None, the average of `min_radius`
        and `max_radius` is used.
    subpix_log : bool
        If True, Gaussian 2-D peak fitting is used after blob detection to achieve
        subpixel accuracy.
    subpix_xc : bool
        If True, Gaussian 2-D peak fitting is used after cross-correlation to achieve
        subpixel accuracy.
    subpix_dict : dictionary or None
        If not None, a dictionary with additional parameters sent to `fpd.utils.gaussian_2d_peak_fit`.
    fit_hw : int, float or None
        Half-width for Gaussian peak fitting. If None, the extracted radius is used.
        If a float, the extracted radius is scaled by this number. If an int, the
        value is used directly.
    plot : bool
        If True, the result of the blob detection is plotted.

    Returns
    -------
    blobs : ndarray
        N x 6 array, where N is the number of blobs detected. The second axis
        is `y, x, r, a, dy, dx` where `r` is the radius, `a` is a measure of the signal,
        and `dy, dx` is the error in `y, x`, where available.

        For the signal, `a`:
            if ref_im is None and not subpix_log: image value at blob centre.
            if ref_im is None and subpix_log: fitted image value at blob centre.
            if ref_im is not None and not subpix_xc: cross-correlation value at blob centre.
            if ref_im is not None and subpix_xc: fitted cross-correlation value at blob centre.
        For the error estimates, `dy, dx`:
            if ref_im is None and not subpix_log: nan, nan.
            if ref_im is None and subpix_log: error values from the fit.
            if ref_im is not None and not subpix_xc: nan, nan.
            if ref_im is not None and subpix_xc: error values from the fit.

    Notes
    -----
    Subpixel data is achieved with fpd.utils.gaussian_2d_peak_fit.

    See Also
    --------
    background_erosion, friedel_filter, vector_combinations, lattice_angles, lattice_magnitudes, synthetic_lattice, lattice_inlier, optimise_lattice, lattice_resolver, lattice_from_inliers, fpd.fft_tools.cepstrum2, strain

    Notes
    -----
    The Laplacian of Gaussian is from skimage.feature.blob_log.

    """

    # TODO switch to namedtuple so more values can be easily returned?

    if ref_im is not None:
        if sigma > 0:
            from .utils import gaus_im_div1d

            ref_im_div = gaus_im_div1d(ref_im, sigma)
        elif sigma < 0:
            ref_im = ref_im.astype(float, copy=False)
            ref_im_div = ref_im - gaussian_filter(ref_im, np.abs(sigma))
        else:
            ref_im_div = ref_im
        if plot:
            plt.matshow(ref_im_div)

    from skimage.feature import blob_log

    min_sigma = min_radius / 2**0.5
    max_sigma = max_radius / 2**0.5

    image_f = image / image.max()

    ### blob detect
    blobs_log = blob_log(
        image_f,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_radii,
        threshold=threshold,
        overlap=overlap,
        log_scale=True,
    )
    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * 2**0.5
    # add 4th column for signal
    blobs_log = np.column_stack((blobs_log, np.zeros(len(blobs_log))))

    # extract simple single pixel value for signal (here and throughout)
    yi, xi = blobs_log.round(0).astype(int)[:, [0, 1]].T
    blobs_log[:, 3] = image[yi, xi]

    ### peak-fit directly
    if subpix_log:
        if fit_hw is None:
            fit_hw = blobs_log[:, 2].astype(int)
        elif isinstance(fit_hw, int):
            fit_hw = blobs_log[:, 2] * 0 + fit_hw
        elif isinstance(fit_hw, float):
            fit_hw = (fit_hw * blobs_log[:, 2]).astype(int)

        spd = {
            "yc": blobs_log[:, 0],
            "xc": blobs_log[:, 1],
            "fit_hw": fit_hw,
            "smoothing": 0,
            "plot": False,
        }
        if subpix_dict is not None:
            spd.update(subpix_dict)
        popt, perr = gaussian_2d_peak_fit(image=image, **spd)
        good_fit = np.isfinite(popt[:, 0])
        # recheck within bounds or check if <1pix?
        blobs_log = np.column_stack(
            (
                popt[good_fit, 1:3],  # yx
                blobs_log[good_fit, 2],  # r
                popt[good_fit, 0] + popt[good_fit, -1],  # offset + amplitude
                perr[good_fit, 1:3],
            )
        )  # error in yx
    else:
        dydx = np.empty_like(blobs_log[:, :2])
        dydx[:] = np.nan
        blobs_log = np.column_stack((blobs_log, dydx))

    ### use pattern matching
    if ref_im is not None:
        image_log = image.copy()
        image_log[image_log == 0] = 1
        image_log = np.log(image_log)

        if sigma > 0:
            image_div = gaus_im_div1d(image_log, sigma)
        else:
            image_div = image_log
        c = fftcorrelate(image_div, ref_im_div, phase_exponent=phase_exponent)

        # find all peaks and then filter with the KDTree below.
        pks = peak_local_max(
            c,
            min_distance=1,
            threshold_abs=None,
            threshold_rel=None,
            exclude_border=True,
        )

        # filter xc peaks by log blobs
        ct = cKDTree(pks)
        if log_xc_max_r is None:
            log_xc_max_r = (min_radius + max_radius) / 2.0
        inds = ct.query_ball_point(x=blobs_log[:, :2], r=log_xc_max_r)

        pks_filt = []
        t = np.zeros(len(blobs_log), dtype=bool)
        for i, indsi in enumerate(inds):
            li = len(indsi)
            if li == 0:
                t[i] = False
                continue
            elif li == 1:
                pks_filt.append(pks[indsi[0]])
                t[i] = True
            else:
                # li > 1, filter peaks by intensity
                yi, xi = pks[indsi].T
                j = c[yi, xi].argmax()
                pks_filt.append(pks[indsi[j]])
                t[i] = True
        pks_filt = np.array(pks_filt)

        # build array with yx, r and a measure of amplitude
        r = blobs_log[:, 2][t]
        yi, xi = pks_filt.round(0).astype(int)[:, [0, 1]].T
        a = c[yi, xi]
        blobs_xc = np.column_stack((pks_filt, r, a))

        ### do peak-fitting
        if subpix_xc:
            if fit_hw is None:
                fit_hw = blobs_xc[:, 2].astype(int)
            elif isinstance(fit_hw, int):
                fit_hw = blobs_xc[:, 2] * 0 + fit_hw
            elif isinstance(fit_hw, float):
                fit_hw = (fit_hw * blobs_xc[:, 2]).astype(int)

            spd = {
                "yc": blobs_xc[:, 0],
                "xc": blobs_xc[:, 1],
                "fit_hw": fit_hw,
                "smoothing": 0,
                "plot": False,
            }
            if subpix_dict is not None:
                spd.update(subpix_dict)
            popt, perr = gaussian_2d_peak_fit(image=c, **spd)
            good_fit = np.isfinite(popt[:, 0])
            # recheck within bounds or check if <1pix?

            blobs_xc = np.column_stack(
                (
                    popt[good_fit, 1:3],  # yx
                    blobs_xc[good_fit, 2],  # r
                    popt[good_fit, 0] + popt[good_fit, -1],  # offset + amplitude
                    perr[good_fit, 1:3],
                )
            )  # error in yx
        else:
            dydx = np.empty_like(blobs_xc[:, :2])
            dydx[:] = np.nan
            blobs_xc = np.column_stack((blobs_xc, dydx))

    if plot:
        cmap = plt.cm.gray
        try:
            # make copy so set bad is only for this instance
            cmap = cmap.__copy__()
        except:
            # for older matplotlibs
            import copy

            cmap = copy.deepcopy(cmap)
        cmap.set_bad("k")

        # blobs
        fig = plt.figure()
        plt.imshow(
            image,
            interpolation="nearest",
            cmap=cmap,
            norm=plt.matplotlib.colors.LogNorm(),
        )
        ax = plt.gca()
        axlims = plt.axis()
        for blob in blobs_log:
            y, x, r = blob[:3]
            ci = plt.Circle((x, y), r, color="red", linewidth=1, fill=False)
            ax.add_patch(ci)
        plt.axis("off")
        plt.tight_layout()

        # cross-cor
        if ref_im is not None:
            plt.matshow(c)
            plt.axis("off")
            # plt.plot(pks[:, 1], pks[:, 0], 'xr', alpha=0.2)
            plt.plot(pks_filt[:, 1], pks_filt[:, 0], "xr")
            plt.plot(blobs_xc[:, 1], blobs_xc[:, 0], "ko", mfc="none")
            plt.tight_layout()

        # amplitude
        amp_blobs = blobs_log
        if ref_im is not None:
            amp_blobs = blobs_xc

        fig = plt.figure()
        ax = plt.gca()
        amps = amp_blobs[:, 3]
        cmap = plt.cm.gray_r
        norm = plt.Normalize(vmin=0, vmax=amps.max())
        clrs = cmap(norm(amps))
        for i, blob in enumerate(amp_blobs):
            y, x, r, a = blob[:4]
            ci = plt.Circle((x, y), r, color="red", linewidth=1, fill=True, fc=clrs[i])
            ax.add_patch(ci)
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap))
        plt.axis(axlims)
        ax.set_aspect(1)
        plt.tight_layout()

    if ref_im is not None:
        return blobs_xc
    else:
        return blobs_log


def friedel_filter(
    blobs,
    cyx,
    min_distance_radii_scale=None,
    min_distance_pad=None,
    optimise_cyx=False,
    plot=False,
):
    """
    Filter (x, y, r) points to keep only those with inversion symmetry
    within some distance.

    Parameters
    ----------
    blobs : ndarray
        N x M array with N rows each where the first 3 columns are (y, x, r),
        where r is the radius.
    cyx : iterable
        Centre position of the direct beam.
    min_distance_radii_scale : None or scalar
        If None and `min_distance_pad` is None, all points with a unique
        nearest neighbour are returned. Otherwise, the sum of the two radii
        are scaled by this and used to set the threshold distance above
        which points are rejected. `min_distance_pad` adds to this distance.
    min_distance_pad : None or scalar
        See `min_distance_radii_scale` for details.
        Else, only those within `min_distance` are returned.
    optimise_cyx : bool
        If True, `cyx` is adjusted to minimise distance between inversion symmetry
        points. In this case, cyx is also returned. The blobs, including the central
        one, are not adjusted.
    plot : bool
        If True, the results of the filtering are plotted.

    Returns
    -------
    blobs_filtered ; ndarray
        A filtered version of `blobs`.
    cyx_opt : tuple, optional
        If optimise_cyx is True, an optimised centre position is also returned.

    See Also
    --------
    background_erosion, blob_log_detect, vector_combinations, lattice_angles, lattice_magnitudes, synthetic_lattice, lattice_inlier, optimise_lattice, lattice_resolver, lattice_from_inliers, fpd.fft_tools.cepstrum2, strain

    """

    n_pts = len(blobs)
    yx = blobs[:, :2] - np.array(cyx)
    yx_inv = -yx

    # calculate distances for all combinations
    dst = np.ones((n_pts, n_pts), dtype=float)
    dst[:] = np.nan
    for i, yxi in enumerate(yx):
        dst_is = ((yx_inv - yxi[None]) ** 2).sum(1) ** 0.5
        dst[i] = dst_is

    if False:
        f = plt.figure()
        plt.imshow(dst, interpolation="nearest", origin="bottom")
        plt.xlabel("point index")
        plt.ylabel("inverted point index")
        plt.title("distance")

    # indices of nn inverted point for each point
    xinds0 = np.arange(n_pts)
    yinds0 = np.nanargmin(dst, 0)

    # and the same for the other axis
    yinds1 = xinds0
    xinds1 = np.nanargmin(dst, 1)

    # where the NN is not unique over one axis, choose the pairing in both axes
    xyinds0 = np.column_stack([xinds0, yinds0])
    xyinds1 = np.column_stack([xinds1, yinds1])

    # change dtype so we work with rows
    dtype = np.dtype((np.void, xyinds0.dtype.itemsize * xyinds0.shape[1]))
    ar1 = np.ascontiguousarray(xyinds0).view(dtype=dtype)
    ar2 = np.ascontiguousarray(xyinds1).view(dtype=dtype)

    # get common pairings
    com_vals = np.intersect1d(ar1, ar2, assume_unique=False)
    pki = np.in1d(ar1, com_vals)  # mask of in-pair in blob
    pt_ind_pairs = com_vals.view(xyinds0.dtype).reshape((-1, 2))

    # trim coords if dst above threshold from assembled list
    threshold = 0
    if min_distance_pad is not None:
        threshold = min_distance_pad
    if min_distance_radii_scale is not None:
        threshold = threshold + blobs[pt_ind_pairs, 2].sum(1) * min_distance_radii_scale

    if min_distance_pad is not None or min_distance_radii_scale is not None:
        pair_dsts = dst[tuple(pt_ind_pairs.T)]
        ki = pair_dsts <= threshold
    else:
        ki = np.ones(pt_ind_pairs.shape[0], dtype=bool)

    pti, pti_inv = pt_ind_pairs[ki].T
    blobs_filtered = blobs[pti]

    if optimise_cyx:
        # ci = np.argmin((yx**2).sum(1))

        dyx = yx[pti] - (yx_inv[pti_inv])
        # dcyx = -np.percentile(dyx, 50, 0)
        # print(dcyx)
        dcyx = -dyx.mean(0) / 2.0
        # print(dcyx)
        cyx = cyx - dcyx

    if plot:
        bki = np.where(pki)[0][ki]
        nkmask = np.ones(blobs.shape[0], dtype=bool)
        nkmask[bki] = False
        bk = blobs[bki]
        bnk = blobs[nkmask]

        cy, cx = cyx

        fig = plt.figure()
        ax = plt.gca()
        ax.invert_yaxis()
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        for blob in bk:
            y, x, r = blob[:3]
            y = y - cy
            x = x - cx
            c1 = plt.Circle((x, y), r, color="red", linewidth=1, fill=False, alpha=0.8)
            c2 = plt.Circle(
                (-x, -y), r, color="blue", linewidth=1, fill=False, alpha=0.8
            )
            ax.add_patch(c1)
            ax.add_patch(c2)
        for blob in bnk:
            y, x, r = blob[:3]
            y = y - cy
            x = x - cx
            c1 = plt.Circle(
                (x, y), r, color="green", linewidth=1, fill=False, alpha=0.3
            )
            c2 = plt.Circle(
                (-x, -y), r, color="lime", linewidth=1, fill=False, alpha=0.3
            )
            ax.add_patch(c1)
            ax.add_patch(c2)

        ax.set_aspect(1)
        ax.relim()
        ax.autoscale_view(True, True, True)
        # plt.axis('off')
        plt.title("red: pts, blue: inverted")
        plt.tight_layout()

    if optimise_cyx:
        return blobs_filtered, cyx
    return blobs_filtered


def synthetic_lattice(
    cyx, ab, angles, reps=None, shape=None, max_r=None, plot=False, max_reps=21
):
    """
    Generate a synthetic lattice filling a 2-D space of different shapes.

    Parameters
    ----------
    cyx : iterable
        Centre position of the direct beam.
    ab : iterable
        Length 2 lattice parameters.
    angles : iterable
        Length 2 angles of lattice vectors in radians.
    reps : iterable or None
        If not None, `reps` is a length 2 iterable representing number of lattice
        repeats on each axis. Must be odd.
    shape : iterable or None
        If not None, `shape` is a length 2 iterable representing the 'image' shape
        to be filled by the lattice.
    max_r : scalar or None
        If not None, the lattice is generated up to the `max_r` radius from `cyx`.
        See notes.
    plot : bool
        If True, the results are plotted.
    max_reps : int
        Maximum number of repeats allowed.

    Returns
    -------
    yxg : ndarray
        Array of y, x coordinates of the lattice, of shape N x 2.

    Notes
    -----
        One and only one of `reps`, `shape`, or `max_r` must be specified.

    See Also
    --------
    background_erosion, blob_log_detect, friedel_filter, vector_combinations, lattice_angles, lattice_magnitudes, lattice_inlier, optimise_lattice, lattice_resolver, lattice_from_inliers, fpd.fft_tools.cepstrum2, strain

    """

    r1, r2 = ab
    t1, t2 = angles
    alpha = t2 - t1

    spec_nones = [x is not None for x in [reps, shape, max_r]]
    if sum(spec_nones) > 1:
        raise Exception("Only one of [reps, shape, max_r] may be defined.")
    if sum(spec_nones) < 1:
        raise Exception("One of [reps, shape, max_r] must be defined.")

    d90 = alpha - np.pi / 2
    dy, dx = r2 * np.cos(d90), r1  # along y and x axes

    if shape is not None:
        # use min step to gen reps and filter later
        dyx_min = min([dy, dx])
        h, w = shape
        if (np.array(cyx) < 0).any():
            raise Exception("'cyx' must be within shape (>=0).")
        if cyx[0] >= h or cyx[1] >= w:
            raise Exception("'cyx' must be within shape.")

        corner_dyxs = np.array(
            [
                [0 - cyx[1], 0 - cyx[0]],
                [0 - cyx[1], w - cyx[0]],
                [h - cyx[1], w - cyx[0]],
                [h - cyx[1], 0 - cyx[0]],
            ]
        )

        # max number of dyx_min
        ns = (corner_dyxs / np.array([dy, dx])[None]).max()
        n_min = np.ceil(ns * 2**0.5).astype(int) * 2 + 1
        reps = (n_min,) * 2

    if max_r is not None:
        # use min step to gen reps and filter later
        dyx_min = min([dy, dx])
        n_min = np.ceil(max_r / dyx_min).astype(int) * 2 + 1
        reps = (n_min,) * 2

    if reps is not None:
        reps = [min(max_reps, x) for x in reps]

        ns_even = [(n % 2) == 0 for n in reps]
        if any(ns_even):
            raise Exception("Values in 'reps' must be odd.")
        n1, n2 = reps

        yxg = (
            np.mgrid[0:n2, 0:n1]
            - np.array([int((n2 - 1) / 2), int((n1 - 1) / 2)])[:, None, None]
        )

    # scale unit vectors by desired vectors
    yxg = yxg * np.array([r2, r1])[..., None, None]

    if d90 != 0:
        # skew along axis
        trans_rot = AffineTransform(rotation=np.pi / 2)
        trans_rot_inv = AffineTransform(rotation=-np.pi / 2)
        trans_shr = AffineTransform(shear=-d90)
        trans = (trans_rot + trans_shr) + trans_rot_inv
        yxg_flt = np.rollaxis(yxg, 0, 3).reshape((-1, 2))
        yxg_flt = trans(yxg_flt)
        yxg = yxg_flt.reshape(yxg.shape[1:] + (2,))
        yxg = np.rollaxis(yxg, 2, 0)

    # rotate and offset
    yxg = rotate_vector(yxg, np.rad2deg(t1))
    yxg = yxg + np.array(cyx)[:, None, None]

    if max_r is not None:
        # select all within max_r
        rs = ((yxg - np.array(cyx)[:, None, None]) ** 2).sum(0) ** 0.5
        rb = rs <= max_r
        yxg = yxg[:, rb]
    elif shape is not None:
        # select all within shape
        yb = np.logical_and(yxg[0] >= 0, yxg[0] <= shape[0])
        xb = np.logical_and(yxg[1] >= 0, yxg[1] <= shape[1])
        b = np.logical_and(yb, xb)
        yxg = yxg[:, b]
    else:
        # reps
        yxg = yxg.reshape((2, -1))

    yxg = yxg.T

    if plot:
        plt.figure()
        plt.plot(yxg[:, 1], yxg[:, 0], "ro")
        plt.plot(cyx[1], cyx[0], "bo")
        plt.gca().invert_yaxis()
        plt.gca().set_aspect(1)
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")

        ax = plt.gca()
        if max_r is not None:
            c1 = plt.Circle((cyx[1], cyx[0]), max_r, color="g", linewidth=1, fill=False)
            ax.add_patch(c1)

        if shape is not None:
            c1 = plt.Rectangle(
                xy=(0, 0),
                width=shape[1],
                height=shape[0],
                color="g",
                linewidth=1,
                fill=False,
            )
            ax.add_patch(c1)

        # display angle / size
        def format_coord(x, y):
            x = x - cyx[1]
            y = y - cyx[0]
            r = np.hypot(x, y)
            t = np.arctan2(y, x)
            td = np.rad2deg(t)
            return "x=%1.4f, y=%1.4f, r=%1.4f, t=%1.4f" % (x, y, r, td)

        ax.format_coord = format_coord

    return yxg


def vector_combinations(blobs, plot=False):
    """
    Calculate all combinations of vectors between points
    in euclidean and polar form.

    Parameters
    ----------
    blobs : ndarray
        N x 3 array with N rows of (y, x, r), where r is the radius.
    plot : bool
        If True, the results are plotted.

    Returns
    -------
    dyx ; ndarray
        Mx2 array of vectors in euclidean form.
    rt : ndarray
        Mx2 array of vectors in polar form, with angle in radians.

    See Also
    --------
    background_erosion, blob_log_detect, friedel_filter, lattice_angles, lattice_magnitudes, synthetic_lattice, lattice_inlier, optimise_lattice, lattice_resolver, lattice_from_inliers, fpd.fft_tools.cepstrum2, strain

    """

    # combinations of dyx for every point (excluding self)
    yxs = blobs[:, :2]
    n_pts = len(yxs)
    sel_inds = np.eye(n_pts) < 1

    dyxs = np.ones((n_pts, n_pts, 2), dtype=float)
    dyxs[:] = np.nan

    for i, yxi in enumerate(yxs):
        dyx = yxs - yxi[None]
        dyxs[i] = dyx
    dyxs = dyxs[sel_inds]

    if plot:
        plt.figure()
        plt.plot(dyxs[:, 1], dyxs[:, 0], "o")
        plt.gca().set_aspect(1)
        plt.gca().invert_yaxis()
        plt.xlabel("dx (pixels)")
        plt.ylabel("dy (pixels)")

    # r and theta from dyx
    r = (dyxs**2).sum(1) ** 0.5
    t = np.arctan2(dyxs[:, 0], dyxs[:, 1])
    rt = np.column_stack([r, t])
    if plot:
        plt.figure()
        plt.plot(t, r, "o")
        plt.xlabel("theta (rad)")
        plt.ylabel("r (pixels)")

    return dyxs, rt


def lattice_angles(
    rt,
    nfold=None,
    bin_deg=1.0,
    hist_gaus=2.0,
    weight_hist=True,
    trim_r_max_pct=79,
    min_distance=None,
    plot=False,
):
    """
    Estimate lattice angles using a statistical approach.

    Parameters
    ----------
    rt : ndarray
        Nx2 array of radii and theta for vector combinations.
    nfold : integer or None
        If None, the angle between vectors is free and will be
        estimated from the most populous angles. Otherwise, angle sets
        matching this symmetry are found. Note that the symmetry is purely
        in angle.
    bin_deg : scalar
        Histogram bin size in degrees.
    hist_gaus : scalar
        Width of Gaussian smoothing applied to histogram.
    weight_hist : bool
        If True, the histogram is weighted by r**-2 to be geometrically
        flat.
    trim_r_max_pct : None of scalar
        If not None, only data points with this percent of max(r) are used.
    min_distance : None or int
        Minimum distance between peaks when `nfold` is 2, in degrees. Set
        `min_distance` equal to `bin_deg` for the minimum spacing.
    plot : bool
        If True, the results are plotted.

    Returns
    -------
    Tuple of a1, a2:
    a1 : scalar
        Angle 1 in radians.
    a2 : scalar
        Angle 2 in radians.

    Notes
    -----
    Angle `a1` is always the first angle greater than zero, with `a2` > `a1`.

    The plot has the y-axis inverted so that the display matched the standard
    image convention of the origin being at the top left. Consequently, positive
    angles appear as a clockwise rotation.

    See Also
    --------
    background_erosion, blob_log_detect, friedel_filter, vector_combinations, lattice_magnitudes, synthetic_lattice, lattice_inlier, optimise_lattice, lattice_resolver, lattice_from_inliers, fpd.fft_tools.cepstrum2, strain

    """

    r, t = rt.T

    if trim_r_max_pct is not None:
        r_max = r.max() * trim_r_max_pct / 100.0
        b = r <= r_max
        r = r[b]
        t = t[b]

    weights = None
    if weight_hist:
        weights = r ** (-2)

    # condition nfold
    if nfold is None:
        nfold = 2
    else:
        if nfold < 2:
            raise Exception("'nfold' must be >= 2")
        if nfold % 2 != 0:
            raise Exception("'nfold' must be even")

    alpha = np.deg2rad(360 / int(nfold))
    t = np.mod(t, alpha)  # this is positive

    hist, bin_edges = np.histogram(
        t, bins=int(360.0 / bin_deg), range=(-np.pi, np.pi), weights=weights
    )
    bin_cnts = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    hist_sm = gaussian_filter1d(hist.astype(float, copy=False), hist_gaus, mode="wrap")

    # find peaks
    if nfold != 2:
        pks = peak_local_max(hist_sm, num_peaks=1)[:, 0]
        i = pks[0]
        a1 = bin_cnts[i]
        a2 = a1 + alpha
        # a1 is min +ve angle. a2 is 1st of next 'peaks'.
    else:
        min_deg = 5
        if min_distance is not None:
            min_deg = min_distance
        min_distance = max([1, int(np.deg2rad(min_deg / bin_deg))])
        pks = peak_local_max(hist_sm, num_peaks=2, min_distance=min_distance)[:, 0]
        i1, i2 = pks[0], pks[-1]
        a1 = bin_cnts[i1]
        a2 = bin_cnts[i2]
        # these are +ve, in 0:180 degs, and are ordered by 'intensity'
        # reorder by angle
        if a2 < a1:
            a1, a2 = a2, a1
        alpha = a2 - a1

    if plot:
        plt.figure()
        plt.plot(bin_cnts, hist_sm)
        plt.axvline(a2, color="r", ls="-")

        plt.axvline(a1, color="r")
        if alpha is not None:
            plt.axvline(a2, color="r", ls="--")

        plt.title(
            "a1: %0.1f, a2: %0.1f deg. Alpha: %0.1f"
            % (np.rad2deg(a1), np.rad2deg(a2), np.rad2deg(a2 - a1))
        )
        plt.xlabel("Angle (radians)")
        plt.ylabel("Weighted Counts")

    return a1, a2


def lattice_magnitudes(
    rt,
    angles,
    window_deg=5,
    bin_pix=1.0,
    hist_gaus=2.0,
    min_vector_mag=None,
    max_vector_mag=None,
    mode="peaks",
    peak_min_distance=None,
    plot=False,
):
    """
    Determine lattice vector magnitudes using FFT or peak analysis of statistics.

    Parameters
    ----------
    rt : ndarray
        Nx2 array of radii and theta for vector combinations.
    angles : iterable
        The angles in radians at which the lattice parameters will be estimated.
    window_deg : scalar
        Window width centred on angles used to select data for statistical analysis.
    bin_pix : scalar
        Histogram bin width in pixels.
    hist_gaus : scalar
        Width of Gaussian smoothing applied to histogram in pixels.
    min_vector_mag : None or scalar
        If not None, the FFT frequencies are limited to >= 1/max_vector_mag.
    max_vector_mag : None or scalar
        If not None, the FFT frequencies are limited to <= 1/max_vector_mag.
    mode : str
        One of ['fft', 'peaks'], determining the spacing analysis.
    peak_min_distance : None or int
        If not None, this sets the minimum distance between peaks in the 'peaks'
        `mode`, in `bin_pix`.
    plot : bool
        If True, the results are plotted.

    Returns
    -------
    lattice_constants : list
        Length N list of lattice constants in pixels.

    See Also
    --------
    background_erosion, blob_log_detect, friedel_filter, vector_combinations, lattice_angles, synthetic_lattice, lattice_inlier, optimise_lattice, lattice_resolver, lattice_from_inliers, fpd.fft_tools.cepstrum2, strain

    """

    modes = ["fft", "peaks"]
    if mode.lower() not in modes:
        raise Exception("'mode' %s not understood." % (mode))
    mode = mode.lower()
    # TODO add cepstrum?

    r, t = rt.T

    lattice_constants = []
    for a in angles:
        b = np.abs(t - a) <= np.deg2rad(window_deg / 2.0)
        tb, rb = t[b], r[b]

        # form histogram
        pad = int((hist_gaus * 4) / bin_pix)
        pad = max([pad, 16])

        hmax = (int(rb.max() // bin_pix) + pad) * bin_pix
        hmin = -pad * bin_pix
        bins = int((hmax - hmin) / bin_pix)

        hist, bin_edges = np.histogram(rb, bins=bins, range=(hmin, hmax))

        # add zero point
        zero_ind = np.abs(bin_edges).argmin()
        hist[zero_ind] = hist.max()

        bin_width = bin_edges[1] - bin_edges[0]
        bin_cnts = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        hist_sm = gaussian_filter1d(
            hist.astype(float, copy=False), hist_gaus / bin_pix, mode="wrap"
        )

        # periodicity analysis
        if mode == "fft":
            hist_sm *= np.hanning(hist_sm.size)
            hist_sm -= hist_sm.mean()

            n = None
            fft_r = np.abs(np.fft.fft(hist_sm), n)
            fft_f = np.fft.fftfreq(len(hist_sm), d=bin_width)

            maxi = int(len(fft_f) / 2.0)
            fft_r = fft_r[1:maxi]
            fft_f = fft_f[1:maxi]

            if max_vector_mag is not None:
                max_vector_magi = np.where(fft_f <= 1.0 / max_vector_mag)[0][-1]
            else:
                max_vector_magi = None
            if min_vector_mag is not None:
                min_vector_magi = np.where(fft_f >= 1.0 / min_vector_mag)[0][0]
            else:
                min_vector_magi = None
            fft_r = fft_r[max_vector_magi:min_vector_magi]
            fft_f = fft_f[max_vector_magi:min_vector_magi]

            # find peak
            pks = peak_local_max(fft_r, num_peaks=1)[:, 0]
            pki = pks[0]
            pk_fft = 1 / fft_f[pki]
            lattice_constants.append(pk_fft)

        if mode == "peaks":
            min_distance = int(max([3, hist_gaus / bin_pix]))
            if peak_min_distance is not None:
                min_distance = max([min_distance, peak_min_distance])
            pks = peak_local_max(hist_sm, min_distance=min_distance)[:, 0]

            # spacing from percentile of diffs
            if None in [min_vector_mag, max_vector_mag]:
                pk_diff_ind = np.diff(pks)
                lattice_constant = np.abs(np.percentile(pk_diff_ind, 50) * bin_width)
            else:
                # improve using all permutations with limits
                from itertools import combinations

                difs = np.array([np.abs(i2 - i1) for (i1, i2) in combinations(pks, 2)])
                if max_vector_mag is not None:
                    difs = difs[difs <= max_vector_mag / bin_pix]
                if min_vector_mag is not None:
                    difs = difs[difs >= min_vector_mag / bin_pix]
                lattice_constant = np.abs(np.percentile(difs, 50) * bin_width)
            lattice_constants.append(lattice_constant)

        if plot:
            plt.figure()

            if mode == "fft":
                n = 3
            if mode == "peaks":
                n = 2

            ax1 = plt.subplot2grid((n, 3), (0, 0), colspan=1, rowspan=2)
            ax2 = plt.subplot2grid((n, 3), (0, 1), colspan=2, rowspan=2, sharey=ax1)

            ax1.plot(tb, rb, "o")
            ax1.axvline(a, color="r")
            ax1.set_xlabel("Angle (rad)")
            ax1.set_ylabel("Vector magnitude (pix)")
            plt.sca(ax1)
            plt.minorticks_on()

            ax2.plot(hist_sm, bin_cnts)
            ax2.set_xlabel("Counts")
            plt.minorticks_on()
            ax1.set_ylim(ax2.get_ylim())

            if mode == "peaks":
                ax2.set_title("lattice: %0.3f" % (lattice_constants[-1]))

            if mode == "fft":
                ax3 = plt.subplot2grid((n, 3), (2, 0), colspan=3)
                ax3.semilogx(1 / fft_f, fft_r)
                ax3.axvline(lattice_constants[-1], color="r")
                ax3.set_xlabel("Lattice parameter (pix)")
                ax3.set_ylabel("abs(FFT)")
                ax3.set_title("lattice: %0.3f" % (lattice_constants[-1]))

            plt.tight_layout()

    return lattice_constants


# def estimate_nfold(rt, max_nfold=10, bin_deg=1.0, hist_gaus=2.0, weight_hist=True, trim_r_pct=79, min_distance=None, plot=False):
#'''
# Estimate lattice angular symmetry using a statistical approach.

# Note that this only uses the angular data and so will return C12 for HCP.

# See `lattice_angles_hist` for details of parameters except for those
# listed below.

# Parameters
# ----------
# max_nfold : integer
# Maximum nfold to consider. Must be even.
# plot : bool
# If True, the results will be plotted.

# Returns
# -------
# nfold : int
# The rotation symmetry of the data.

#'''

# r, t = rt.T

# if trim_r_pct is not None:
# r_max = np.percentile(r, trim_r_pct)
# b = r<=r_max
# r = r[b]
# t = t[b]

# weights = None
# if weight_hist:
# weights = r**(-2)

## condition max_nfold
# if max_nfold is None:
# max_nfold = 2
# else:
# if max_nfold < 2:
# raise Exception("'max_nfold' must be >= 2")
# if max_nfold % 2 !=0:
# raise Exception("'max_nfold' must be even")

# min_deg = 5
# if min_distance is not None:
# min_deg = min_distance
# min_distance = max([1, int(np.deg2rad(min_deg/bin_deg))])

# nfolds = np.arange(2, max_nfold+1, 2, int)
# ars = []
# for nfold in nfolds:
# alpha = np.deg2rad(360/int(nfold))
# th = np.mod(t, alpha)    # this is positive

# hist, bin_edges = np.histogram(th, bins=int(360.0/bin_deg), range=(-np.pi, np.pi), weights=weights)
# bin_cnts = (bin_edges[:-1] + bin_edges[1:])/2.0
# hist_sm = gaussian_filter1d(hist.astype(float, copy=False), hist_gaus, mode='wrap')

# pks = peak_local_max(hist_sm, num_peaks=2, min_distance=min_distance)[:, 0]
# i1, i2 = pks[0], pks[-1]
# int1 = hist_sm[i1]
# int2 = hist_sm[i2]

# ars.append(int1 / int2)
# ars = np.array(ars)

## get index at maximum ratio
# nfoldi = np.argmax(ars)
# nfold = nfolds[nfoldi]

# if plot:
# plt.figure()
# plt.plot(nfolds, ars)
# plt.axvline(nfold, color='r', ls='-')

# plt.title('nfold : %d' %(nfold))
# plt.xlabel('nfold')
# plt.ylabel('Peak ratio')

# return nfold


def lattice_inlier(yx, yxg, r=4, plot=False):
    """
    Determine inliers between two lattices.

    Parameters
    ----------
    yx : ndarray
        Array of y, x coordinates of the blobs, of shape N x 2.
    yxg : ndarray
        Array of y, x coordinates of the synthetic lattice, of shape M x 2.
    r : scalar
        The distance between points considered as inliers.
    plot : bool
        If True, the results are plotted.

    Returns
    -------
    inliers : ndarray
        1-D boolean array of blob inlier status, of length N.

    See Also
    --------
    background_erosion, blob_log_detect, friedel_filter, vector_combinations, lattice_angles, lattice_magnitudes, synthetic_lattice, optimise_lattice, lattice_resolver, lattice_from_inliers, fpd.fft_tools.cepstrum2, strain

    """

    lattice_kdt = cKDTree(yxg)
    matches = lattice_kdt.query_ball_point(yx, r=r)
    inliers = np.array([len(m) == 1 for m in matches])
    # print(inliers.sum())
    lattice_inlier_inds = np.array([i[0] for i in matches[inliers]])

    if plot:
        yxg_outliers = np.delete(yxg, lattice_inlier_inds, 0)

        yx_outliers = yx[inliers == False]
        yx_inliers = yx[inliers]
        yxg_inliers = yxg[lattice_inlier_inds]

        rs = ((yxg_inliers - yx_inliers) ** 2).sum(1) ** 0.5
        r_sum = (rs**2).sum() ** 0.5

        plt.figure()
        plt.plot(yx_outliers[:, 1], yx_outliers[:, 0], "kx", ms=4)
        plt.plot(yx_inliers[:, 1], yx_inliers[:, 0], "bo", ms=4)
        plt.plot(yxg_inliers[:, 1], yxg_inliers[:, 0], "ro", mfc="none")
        # plt.plot(cyx[1], cyx[0], 'bo', ms=6, mfc='none')
        plt.plot(
            yxg_outliers[:, 1], yxg_outliers[:, 0], "ro", mfc=(0,) * 4, mec=(0.5,) * 4
        )

        plt.gca().invert_yaxis()
        plt.gca().set_aspect(1)
        plt.title("error: %0.3f" % (r_sum))
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")

    return inliers


def optimise_lattice(
    yx,
    cyx,
    ab,
    angles,
    shape,
    weights=None,
    constraints=None,
    options=None,
    plot=False,
    **kwd,
):
    """
    Optimise lattice parameters by minimising euclidean distance
    between supplied data points and a synthetic lattice.

    Parameters
    ----------
    yx : ndarray
        Array of y, x coordinates of the blobs, of shape N x 2.
    cyx : iterable
        Centre position of the direct beam.
    ab : iterable
        Length 2 lattice parameters.
    angles : iterable
        Length 2 angles of lattice vectors in radians.
    shape : iterable
        A length 2 iterable representing the 'image' shape to be filled by
        the lattice.
    weights : None or ndarray
        If not None, an array of weights used for the fit, of length N.
    constraints : dict or sequence of dict
        Parameter constraints for error minimisation. See `scipy.optimize.minimize`
        for details. See Notes for discussion of the correct form and examples.
    options: None or dict
        If not None, a dictionary of options passed the minimiser.
    plot : bool
        If True, the results are plotted.

    Returns
    -------
    tuple of cyx_opt, ab_opt, angles_opt, error
    `cyx_opt, ab_opt, angles_opt` are the optimised values of `cyx, ab, angles`.
    `error` is the quadrature combination of the euclidean distance between the
    experimental and modelled lattices. Divide by the number of points to get
    the error per point.

    Notes
    -----
    Additional keyword arguments are passed to the minimiser.

    Constraints between parameters may be set with the `constraints` parameter.

    The parameters optimised in the fit are in a tuple:
    x = cy, cx, a1, a2, r1, r2
    where the c's are the centre coords, the a's are the angles, and the r's are
    the magnitudes.

    This tuple is indexed in the constraint definition. For example, to fix the angle
    between the lattice vectors to 90 degrees, one could specify the following
    constraint:

    constraint = {'type': 'eq', 'fun': lambda x:  x[3] - x[2] - np.pi/2}

    Here we have set an equality constraint defined in the lambda function (==0).

    Bounds may be set with the kwds. For example, constraints on (cy, cx) may be
    set with the following bounds:

    bounds = [(120, 140), (120, 140)] + [(None, None)]*4

    Where the 2 element sequence indicates the lower and upper bound. (None, None)
    is used for no bounds.

    See Also
    --------
    background_erosion,  blob_log_detect, friedel_filter, vector_combinations, lattice_angles, lattice_magnitudes, synthetic_lattice, lattice_inlier, lattice_resolver, strain

    """

    if shape is None:
        # shape = tuple(yx.ptp(0).astype(int) + 2)

        # set as max distance of cyx from any edge *2 to ensure full coverage
        dyx = yx - np.array(cyx)[None]
        shape = tuple(np.ceil(np.abs(dyx).max(0) * 2).astype(int))

    def fun(x, yx, shape):
        cy, cx, a1, a2, r1, r2 = x
        yxg = synthetic_lattice(cyx=(cy, cx), ab=(r1, r2), angles=(a1, a2), shape=shape)
        lattice_kdt = cKDTree(yxg)
        dst, ind = lattice_kdt.query(yx, k=1)
        if weights is not None:
            r_sum = (dst**2 * weights).sum() / weights.sum()
            # so same 'distance' is returned as in non-weighted case:
            r_sum = (r_sum * len(dst)) ** 0.5
        else:
            r_sum = (dst**2).sum() ** 0.5
        return r_sum

    x0 = (cyx[0], cyx[1], angles[0], angles[1], ab[0], ab[1])
    if options is None:
        options = {}
    if "tol" not in kwd:
        kwd.update({"tol": 1e-6})
    res = minimize(
        fun, x0, args=(yx, shape), constraints=constraints, options=options, **kwd
    )
    x_opt = res.x
    error = res.fun

    cyx_opt = x_opt[:2]
    angles_opt = x_opt[2:4]
    ab_opt = x_opt[4:]

    if plot:
        # plot results
        yxg = synthetic_lattice(
            cyx=cyx_opt, ab=ab_opt, angles=angles_opt, shape=shape, plot=False
        )

        pstr = (
            "cyx: (%0.2f, %0.2f), angles: (%0.2f, %0.2f), ab: (%0.2f, %0.2f)"
            % tuple(x_opt)
        )

        plt.figure()
        plt.plot(yx[:, 1], yx[:, 0], "o", ms=4)
        plt.plot(yxg[:, 1], yxg[:, 0], "ro", mfc="none")
        plt.plot(x_opt[1], x_opt[0], "bo", ms=6, mfc="none")
        plt.gca().invert_yaxis()
        plt.gca().set_aspect(1)
        plt.title("error: %0.3f\n%s" % (error, pstr))
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")

    return cyx_opt, ab_opt, angles_opt, error


def lattice_resolver(
    ab, angles, res_angles, reps=(5, 5), search_dangle=1.0, r_max=None
):
    """
    Resolve lattice parameters along specific axes defined by `res_angles`.

    Parameters
    ----------
    ab : ndarray
        Lattice space parameters of shape (2, nr, nc).
    angles : ndarray
        Lattice angles of shape (2, nr, nc) in radians.
    res_angles : iterable
        Angles to resolve the lattice parameters in degrees (length 2).
    reps : iterable
        A length 2 iterable representing number of lattice
        repeats on each axis. Must be odd. See Notes.
    search_dangle : scalar
        Angular range used to select vertices, in degrees.
    r_max : None or scalar
        If not None, the maximum radius in which to consider synthetic
        lattice points.

    Returns
    -------
    abs_res : ndarray
        Resolved lattice space parameters.
    angles_res : ndarray
        Resolved lattice angles.

    Notes
    -----
    When resolving latices with different shapes across a dataset, increasing
    `reps` may be required for there to be a lattice point in the synthetic lattice.

    Inputs with NaNs with return NaNa.

    See Also
    --------
    background_erosion, blob_log_detect, friedel_filter, vector_combinations, lattice_angles, lattice_magnitudes, synthetic_lattice, lattice_inlier, optimise_lattice, lattice_from_inliers, fpd.fft_tools.cepstrum2, strain

    """

    nr, nc = ab.shape[-2:]
    abs_res = []
    angles_res = []
    for ri in range(nr):
        for ci in range(nc):
            abi = ab[:, ri, ci]
            anglesi = angles[:, ri, ci]

            if np.isnan(abi).any() or np.isnan(anglesi).any():
                abs_res.append((np.nan,) * 2)
                angles_res.append((np.nan,) * 2)
                continue

            yxg = synthetic_lattice((0, 0), abi, anglesi, reps=reps, plot=False)
            r = (yxg**2).sum(1) ** 0.5
            t = np.arctan2(yxg[:, 0], yxg[:, 1])

            # clip synth if requested
            if r_max is not None:
                b = np.where(r <= r_max)[0]
                r = r[b]
                t = t[b]

            # rm zero point
            zi = np.abs(r).argmin()
            r = np.delete(r, zi, 0)
            t = np.delete(t, zi, 0)

            # convert to deg
            td = np.rad2deg(t)

            deg1, deg2 = res_angles
            # d_deg = deg2 - deg1

            # get best angle
            i1 = np.abs(td - deg1).argmin()
            # i2_deg = td[i1] + d_deg
            i2 = np.abs(td - deg2).argmin()

            angles_res.append(t[[i1, i2]])

            # find minimum radius along angles
            i1_inrange = np.where(np.abs(td - td[i1]) <= search_dangle)[0]
            r1 = r[i1_inrange].min()

            i2_inrange = np.where(np.abs(td - td[i2]) <= search_dangle)[0]
            r2 = r[i2_inrange].min()

            abs_res.append([r1, r2])

    abs_res = np.array(abs_res)
    angles_res = np.array(angles_res)
    abs_res.shape = (nr, nc, 2)
    angles_res.shape = (nr, nc, 2)
    abs_res = np.moveaxis(abs_res, 2, 0)
    angles_res = np.moveaxis(angles_res, 2, 0)

    return abs_res, angles_res


def lattice_from_inliers(
    yx,
    cyx,
    r=8,
    max_n=10,
    r_min=None,
    r_max=None,
    min_dangle=10,
    max_dangle=170,
    degen_mode="max_geom_r",
    degen_func=None,
    reps=41,
    plot=False,
):
    """
    Estimate lattice parameters from synthetic inlier detection.

    Parameters
    ----------
    yx : ndarray
        Array of y, x coordinates of potential vertices, of shape N x 2.
    cyx : iterable
        Centre position of the direct beam.
    r : scalar
        The distance between points considered as inliers.
    max_n : integer
        Maximum number of points to consider. Set to ` np.inf` to include all.
    r_min : None of scalar
        Minimum radius to consider.
    r_max : None of scalar
        Maximum radius to consider.
    min_dangle : scalar
        Minimum angle between vectors, in degrees.
    max_dangle : scalar
        Maximum angle between vectors, in degrees.
    degen_mode : string
        Mode used to split different lattices with the same number of inliers.
            max_geom_r : geometrical mean of lattice vectors.
        See also `degen_func`.
    degen_func : None or callable
        If not None, a function of with signature float = f(r1, r2, a1, a2)
        where the `r` values are the vector magnitudes, and the `a` parameters
        are the vector angles in radians in [0 2pi]. Lattices with maximal
        function return values retained.
    reps : integer
        Number of lattice repeats on each axis in the synthetic lattice.
        Must be odd.
    plot : bool
        If True, the results are plotted.

    Returns
    -------
    ab : iterable
        Length 2 lattice parameters.
    angles : iterable
        Length 2 angles of lattice vectors in radians.

    See Also
    --------
    background_erosion, blob_log_detect, friedel_filter, vector_combinations, lattice_angles, lattice_magnitudes, synthetic_lattice, lattice_inlier, optimise_lattice, lattice_resolver, fpd.fft_tools.cepstrum2, strain

    """

    # calc r, theta and sort by r
    yxc = yx - np.array(cyx)[None]
    rmag = (yxc**2).sum(1) ** 0.5
    theta = np.arctan2(yxc[:, 0], yxc[:, 1])

    si = np.argsort(rmag, axis=0)
    rs = rmag[si]

    # check limits
    rmax = rs.max()
    rmin = rs.min()
    if r_min is not None:
        if r_min > rmax:
            raise ValueError("'r_min' must be < maximum radius, %0.2f" % (rmax))
        if r_min <= rmin:
            r_min = None
    if r_max is not None:
        if r_max < rmin:
            raise ValueError("'r_max' must be > minimum radius, %0.2f" % (rmin))
        if r_max >= rmax:
            r_max = None

    # find indices
    start_i = 1  # 1 because of centre spot
    if r_min is not None:
        start_i = np.argmax(rs >= r_min)
        # this could include central spot if r_min is set very small
    if r_max is not None:
        end_i = np.argmax(rs >= r_max) + 1
        n = end_i - start_i
        if n > max_n:
            end_i -= n - max_n
    else:
        if max_n == np.inf:
            end_i = len(rs)
        else:
            end_i = start_i + max_n

    # in range
    inds = si[start_i:end_i]
    # rm_ir = rmag[inds]
    # theta_ir = theta[inds]

    # determine degenerate inlier split function
    degen_dict = {"max_geom_r": lambda r1, r2, a1, a2: r1 * r2}
    if degen_func is not None:
        degen_dict.update({"degen_func": degen_func})
        degen_mode = "degen_func"
    if degen_mode not in degen_dict.keys():
        raise NotImplementedError("degen_mode `%s` is not implemented" % (degen_mode))
    degen_sel = degen_dict[degen_mode]

    # loop over combinations
    from itertools import combinations

    combos = list(combinations(inds, 2))

    best_combo_i = 0
    best_combo_val = 0
    best_degen_val = 0
    for i, ci in enumerate(combos):
        r1, r2 = rmag[list(ci)]
        a1, a2 = theta[list(ci)]

        # check dangles
        da_min = np.abs(a2 - a1)
        da_max = np.pi * 2 - da_min
        if da_max < da_min:
            da_min, da_max = da_max, da_min
        if np.rad2deg(da_min) < min_dangle:
            continue
        if np.abs(np.rad2deg(da_max) - 180) < min_dangle:
            continue

        # synthetic lattice
        yxg = synthetic_lattice(
            cyx=(0, 0), ab=(r1, r2), angles=(a1, a2), reps=(reps,) * 2, plot=False
        )
        inliers = lattice_inlier(yxc, yxg, r=r, plot=False)

        n_in = inliers.sum()
        # n_out = inliers.size - n_in
        if n_in > best_combo_val:
            best_combo_i = i
            best_combo_val = n_in
            best_degen_val = degen_sel(r1, r2, a1, a2)
        if n_in == best_combo_val:
            # if multiples w/ equal number of inliers, chose biggest r
            cur_degen_val = degen_sel(r1, r2, a1, a2)
            if cur_degen_val > best_degen_val:
                best_combo_i = i
                best_combo_val = n_in
                best_degen_val = cur_degen_val
    best_inds = list(combos[best_combo_i])
    ab = rmag[best_inds]
    angles = theta[best_inds]

    # return w/ smallest angle first
    if angles[1] < angles[0]:
        angles = angles[::-1]
        ab = ab[::-1]

    if plot:
        yxg = synthetic_lattice(
            cyx=(0, 0), ab=ab, angles=angles, reps=(reps,) * 2, plot=False
        )
        inliers = lattice_inlier(yxc, yxg, r=r, plot=True)
        lim = np.abs(yxc).max() * 1.1
        plt.xlim(-lim, lim)
        plt.ylim(lim, -lim)
        y, x = yxc[best_inds].T
        plt.plot(x, y, "+g", ms=12)

    return (ab, angles)


def background_erosion(
    image, radius, sigma1=2, rf1=1.2, rf2=0.6, sigma2f=0.05, plot=False
):
    """
    Estimate background profile in diffraction patterns by erosion and filtering.
    The order of the operation matches the parameter order in the docstring.

    Parameters
    ----------
    image : ndarray
        Image to process.
    radius : scalar
        Radius of the features to be removed in pixels.
    sigma1 : scalar or None
        Standard deviation of pre-processing Gaussian filtering in pixels. This
        should be set to smooth out local minima such as noise or bad pixels. If
        None, no smoothing is performed.
    rf1 : scalar
        Scale factor used with `radius` to set radii of the disk structural
        element used to remove bright features. Note that `sigma1` is added to
        `radius` before scaling for `rf1`. In general, `rf1` should be >1 to
        remove most features.
    rf2 : scalar or None:
        If not None, a second erosion with a disk structural element of radius
        `radius` scaled by this `rf2`. This should be <1, and is used to
        reduce small residual peaks.
    sigma2f : scalar or None
        Standard deviation of post-processing Gaussian filter, expressed as a
        fraction of the geometrical mean of the image size in pixels. This is
        used to remove high frequency components of the background. If None, no
        smoothing is performed.
    plot : bool
        if True, the results are plotted.

    Returns
    -------
    imbk : ndarray
        The background image.

    See Also
    --------
    blob_log_detect, friedel_filter, vector_combinations, lattice_angles, lattice_magnitudes, synthetic_lattice, lattice_inlier, optimise_lattice, lattice_resolver, lattice_from_inliers, fpd.fft_tools.cepstrum2

    """

    if sigma1 is None:
        sigma1 = 0

    if sigma2f is None:
        sigma2f = 0
        sigma2 = 0
    else:
        sigma2 = np.ceil(image.size**0.5 * sigma2f)

    r1 = int(np.ceil((radius + sigma1) * rf1))

    if sigma1 != 0:
        imf1 = gaussian_filter(image, sigma=sigma1)
    else:
        imf1 = image

    imf2 = erosion(imf1, disk(r1))
    if rf2 is not None:
        r2 = int(np.ceil(radius * rf2))
        imf3 = erosion(imf2, disk(r2))
    else:
        imf3 = im2

    if sigma2 != 0:
        imf4 = gaussian_filter(imf3, sigma=sigma2)
    else:
        imf4 = imf3

    if plot:
        log_im = image - imf4
        bad = log_im <= 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_im = np.log(image - imf4)
        log_im[bad] = np.nan

        f, axs = plt.subplots(2, 3, sharex=True, sharey=True)
        for axi, imi, titi in zip(
            axs.flat,
            [imf1, imf2, imf3, imf4, image - imf4, log_im],
            [
                "sigma1 (%0.2f)" % (sigma1),
                "erosion1 (%0.2f)" % (rf1),
                "erosion2 (%0.2f)" % (rf2),
                "sigma2f (%0.2f)" % (sigma2f),
                "image - background",
                "log(image - background)",
            ],
        ):
            im = axi.imshow(imi)
            plt.colorbar(im, ax=axi)
            axi.set_title(titi, fontsize=10)
    return imf4


def _nc_cor_func(x, im, nc, ncm):
    s = x
    cor_im = im * (1 - (nc / ncm - 1) * s)
    return cor_im


def _nc_func(x, im, nc, ncm):
    s = x
    cor_im = _nc_cor_func(x, im, nc, ncm)
    std = cor_im.std()
    return std


def nc_correct(im, nc, plot=False):
    """
    Corrects `im` for gun noise `nc` through linear scalling.

    Parameters
    ----------
    im : ndarray
        Input image to be corrected.
    nc : ndarray
        Gun noise image to be used for correction.
    plot : bool
        If True, images are plotted.

    Returns
    -------
    cim : ndarray
        Corrected image.

    """

    ncm = nc.mean()
    x0 = 1
    res = minimize(_nc_func, x0, args=(im, nc, ncm))
    cim = _nc_cor_func(res.x, im, nc, ncm)

    if plot:
        f, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 3))
        for imi, axi, titi in zip(
            [im, cim, nc], axs.flat, ["image", "corrected", "nc"]
        ):
            im = axi.imshow(imi, interpolation="nearest")
            axi.set_title(titi)
            f.colorbar(mappable=im, ax=axi)
        plt.tight_layout()

    return cim


def strain(
    r_ref,
    t_ref,
    r,
    t,
    invert_space=False,
    origin="top",
    plot=True,
    pct=0,
    centred_cmap=True,
):
    """
    Calculate infinitesimal strain tensor of polar coordinates (r, t)
    using reference coordinates (r_ref, t_ref).

    Parameters
    ----------
    r_ref : length 2 iterable
        Reference vector magnitudes. If data is in reciprocal space,
        ensure `invert_space=True` to obtain real-space data.
    t_ref : length 2 iterable
        Reference vector angles, in radians. See `origin`.
    r : ndarray
        Data vector magnitudes. If data is in reciprocal space, ensure
        `invert_space=True` to obtain real-space data. The data may be:
        - 1-D of length 2 for a single point.
        - 2-D with shape 2xM for a line profile.
        - 3-D with shape 2xMxN for a y- and x-axes of size M and N.
    t : ndarray
        Data vector angles, in radian. The shape should match that of
        `r`. See `origin`.
    invert_space : bool
        If True, the input space in inverted before the strain is
        calculated.
    origin : str
        One of ['top', 'bottom'], determining the y-axis origin of the
        input data. The output data always has the y-axis origin at
        the bottom.
    plot : boolean
        If True and `r` and `t` do not represent a single data point,
        the results are plotted. For surface plots, the range is set
        individually in each panel. See `pct` and `centred_cmap` for
        surface plot control.
    pct : scalar
        The percentile in (0, 100) that sets the plotted range in
        surface plots. The range is [pct, 100-pct].
    centred_cmap : bool
        If True, a centred coolwarm color map is used for surface plots.
        Otherwise, the default colourmap is used and the plotted range
        is not forced to be symmetric.

    Returns
    -------
    es : ndarray
        Infinitesimal strain tensor components, of shape [4, ...], where
        the 4 elements are [exx, eyy, exy, er]. exy is taken as positive
        when the angle between positive x- and y-axes becomes more acute.
        The rotation, er, is positive for anticlockwise rotation (always
        with the y-axis origin at the bottom, regardless of the `origin`
        parameter value which only influences interpretation of the input
        data, not the output data).

    Notes
    -----
    If the reference is spatially resolved, then it should be reduced to
    representative values by, for example, taking the mean or median of
    the vector magnitudes and angles.

    The coordinate system on which the strain is to be assessed may not
    match the coordinate space that is of interest, such as an interface.
    Once identified, any misalignment may be corrected by adding appropriate
    offsets to the angle inputs, `t_ref` and `t`.

    If the precise offset angle is unknown, say, due to the ideal reference
    being orthogonal and the real reference being slightly off of orthogonal,
    then splitting the error across both vectors is a reasonable compromise.

    See Also
    --------
    background_erosion, blob_log_detect, friedel_filter, vector_combinations, lattice_angles, lattice_magnitudes, synthetic_lattice, lattice_inlier, optimise_lattice, lattice_resolver, lattice_from_inliers, fpd.fft_tools.cepstrum2, strain

    """

    # process origin
    origins = ["top", "bottom"]
    if origin not in origins:
        raise Exception("'origin (%s) must be one of: " % (origin), str(origins))

    if origin == "top":
        # reverse angles for y pointing up
        t_ref = -t_ref
        t = -t

    # ensure arrays
    r_ref = np.array(r_ref)
    t_ref = np.array(t_ref)
    r = np.array(r)
    t = np.array(t)

    # get dimension and shape
    nd = r.ndim - 1
    os = r.shape

    # make 3-D by default
    for _ in range(3 - len(os)):
        r = r[..., None]
        t = t[..., None]

    # generate output array
    es = np.empty((4,) + r.shape[1:])

    # calculate ref coordinates
    xr = r_ref * np.cos(t_ref)
    yr = r_ref * np.sin(t_ref)
    ref_coords = np.array([xr, yr]).T
    if invert_space:
        ref_coords = np.linalg.inv(ref_coords).T

    # loop over points
    for i in range(r.shape[1]):
        for j in range(r.shape[2]):
            # calculate coordinates
            xi = r[:, i, j] * np.cos(t[:, i, j])
            yi = r[:, i, j] * np.sin(t[:, i, j])
            i_coords = np.array([xi, yi]).T
            if invert_space:
                try:
                    i_coords = np.linalg.inv(i_coords).T
                except np.linalg.LinAlgError:
                    es[:, i, j] = np.nan
                    continue

            # calculate transformation matrix
            # tm = np.linalg.solve(ref_coords, i_coords)
            tm, residuals, rank, s = np.linalg.lstsq(ref_coords, i_coords, rcond=None)

            # convert matrix to entry parameters (in notes).
            t11, t12, t21, t22 = tm.flatten()

            # infinitesimal strain tensor
            e11 = t11 - 1
            e12 = (t21 + t12) / 2.0
            # e21 = e12
            e22 = t22 - 1
            er = -(t21 - t12) / 2.0
            # e12 +ve is more acute

            es[:, i, j] = e11, e22, e12, er
    es = np.squeeze(es)

    if plot:
        labs = "exx, eyy, exy, er".split(", ")
        if nd == 0:
            # point
            pass
        elif nd == 1:
            # lines
            f, axs = plt.subplots(2, 2, sharex=True, sharey=False)
            for axi, di, labi in zip(axs.flatten(), es, labs):
                axi.plot(di)
                axi.set_title(labi)
            plt.tight_layout()
        elif nd == 2:
            # surfaces

            # limits
            vmin, vmax = np.nanpercentile(es, [pct, 100 - pct], axis=(-2, -1))
            if centred_cmap:
                vmin, vmax = (
                    np.nanmax(np.abs([vmin, vmax]), 0)[:, None] * np.array([-1, 1])
                ).T
                cmap = "coolwarm"
            else:
                cmap = None

            f, axs = plt.subplots(2, 2, sharex=True, sharey=True)
            for i, (axi, di, labi) in enumerate(zip(axs.flatten(), es, labs)):
                imi = axi.matshow(di, cmap=cmap, vmin=vmin[i], vmax=vmax[i])
                # axi.set_title(labi)
                plt.colorbar(imi, ax=axi, label=labi)
            plt.tight_layout()

    return es
