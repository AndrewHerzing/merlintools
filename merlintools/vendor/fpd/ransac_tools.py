
from skimage.measure.fit import BaseModel
from skimage.measure import ransac
import scipy as sp
import numpy as np
from scipy.interpolate import SmoothBivariateSpline
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm
import copy

from .utils import seq_image_array, unseq_image_array


class _Plane3dModel(BaseModel):
    """Total least squares estimator for plane fit to 3-D point cloud.

    Attributes
    ----------
    model_data : array
        Fitted model.
    params : array
        model = C[0]*x + C[1]*y + C[2]

    """

    def my_model(self, p, *args):
        (x, y, z) = args
        m = p[0]*x + p[1]*y + p[2]
        return m

    def estimate(self, data):
        """Estimate model from data.

        Parameters
        ----------
        data : (N, D) array
            Flattened length N (x,y,z) point cloud to fit to.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """

        if False:
            return False
        else:
            # best-fit linear plane
            # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.lstsq.html
            x, y, z = data.T
            A = np.c_[x, y, np.ones(data.shape[0])]
            C,_,_,_ = sp.linalg.lstsq(A, z)    # coefficients

            self.params = C
            return True


    def residuals(self, data):
        """Determine residuals of data to model.

        Parameters
        ----------
        data : (N, D) array
            Flattened length N (x,y,z) point cloud to fit to.

        Returns
        -------
        residuals : (N) array
            Z residual for each data point.

        """

        x, y, z = data.T
        C = self.params
        # evaluate model at points
        args = (x, y, z)
        self.model_data = self.my_model(self.params, *args)
        z_error = self.model_data - z
        return z_error


class _Poly3dModel(BaseModel):
    """Total least squares estimator for quadratic fit to 3-D point cloud.

    Attributes
    ----------
    model_data : array
        Fitted model.
    params : array
        Model params.

    Notes
    -----
    Quadratic equation based on:
    https://stackoverflow.com/questions/18552011/3d-curvefitting/18648210#18648210.

    """

    def my_model(self, p, *args):
        (x, y, z) = args
        m = np.dot(np.c_[np.ones(x.shape), x, y, x*y, x**2, y**2], p)
        return m


    def estimate(self, data):
        """Estimate model from data.

        Parameters
        ----------
        data : (N, D) array
            Flattened length N (x,y,z) point cloud to fit to.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """

        if False:
            return False
        else:
            # best-fit quadratic curve
            A = np.c_[np.ones(data.shape[0]), data[:,:2],
                      np.prod(data[:,:2], axis=1), data[:,:2]**2]
            C,_,_,_ = sp.linalg.lstsq(A, data[:,2])
            self.params = C
            return True


    def residuals(self, data):
        """Determine residuals of data to model.

        Parameters
        ----------
        data : (N, D) array
            Flattened length N (x,y,z) point cloud to fit to.

        Returns
        -------
        residuals : (N) array.
            Z residual for each data point.
        """

        x, y, z = data.T
        C = self.params
        # evaluate model at points
        args = (x, y, z)
        self.model_data = self.my_model(self.params, *args)
        z_error = self.model_data - z
        return z_error


class _Poly3dParaboloidModel(BaseModel):
    """Total least squares estimator for paraboloid fit to 3-D point cloud.

    Attributes
    ----------
    model_data : array
        Fitted model.
    params : array
        Model params.

    """


    def my_model(self, p, *args):
        (x, y, z) = args
        m = np.abs(p[0])*((x-p[1])**2 + (y-p[2])**2) + p[3]
        return m

    def my_dfun(self, p, *args):
        (x, y, z) = args
        da = ((x-p[1])**2 + (y-p[2])**2)
        dx = -2*np.abs(p[0])*(x-p[1])
        dy = -2*np.abs(p[0])*(y-p[2])
        dc = np.ones(dy.shape[0])
        return np.array([da, dx, dy, dc])

    def my_fit(self, p, *args):
        m = self.my_model(p, *args)
        (x, y, z) = args
        return m - z


    def estimate(self, data):
        """Estimate model from data.

        Parameters
        ----------
        data : (N, D) array
            Flattened length N (x,y,z) point cloud to fit to.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """

        # checks for validity of selected data would go here
        if False:
            return False
        else:
            x, y, z = data.T
            p0 = (0.1, x.mean(), y.mean(), 0)
            args = (x, y, z)
            popt, ier = sp.optimize.leastsq(func=self.my_fit, x0=p0, args=args,
                                            maxfev=10000, full_output=False,
                                            Dfun=self.my_dfun, col_deriv=True)
            self.params = popt
            if ier > 4:
                return False
            else:
                return True


    def residuals(self, data):
        """Determine residuals of data to model.

        Parameters
        ----------
        data : (N, D) array
            Flattened length N (x,y,z) point cloud to fit to.

        Returns
        -------
        residuals : (N) array
            Z residual for each data point.
        """

        x, y, z = data.T
        C = self.params
        # evaluate model at points
        args = (x, y, z)
        self.model_data = self.my_model(self.params, *args)
        z_error = self.model_data - z
        return z_error


def _model_class_gen(model_f, p0, param_dict=None):
    class FuncModel(BaseModel):
        """Least squares fit of arbitrary function to 3-D point cloud.

        Attributes
        ----------
        model_data : array
            Fitted model.
        params : array
            Model params.

        """
        
        def my_model(self, p, *args):
            m = model_f(p, *args)
            return m

        def my_fit(self, p, *args):
            m = self.my_model(p, *args)
            (x, y, z) = args
            return m - z

        def p0_f(self):
            # probably is a cleaner way than function for storing p0.
            return p0
        
        def param_dict_f(self):
            return(param_dict)

        def estimate(self, data):
            """Estimate model from data.

            Parameters
            ----------
            data : (N, D) array
                Flattened length N (x,y,z) point cloud to fit to.

            Returns
            -------
            success : bool
                True, if model estimation succeeds.
            """
            params = {'maxfev': 10000}
            
            param_dict = self.param_dict_f()
            if param_dict is None:
                param_dict = {}
            params.update(param_dict)
            
            # checks for validity of selected data would go here
            if False:
                return False
            else:
                x, y, z = data.T
                p0 = self.p0_f()
                args = (x, y, z)
                popt, ier = sp.optimize.leastsq(func=self.my_fit, x0=p0,
                                                args=args, full_output=False,
                                                **params)
                self.params = popt
                if ier > 4:
                    return False
                else:
                    return True


        def residuals(self, data):
            """Determine residuals of data to model.

            Parameters
            ----------
            data : (N, D) array
                Flattened length N (x,y,z) point cloud to fit to.

            Returns
            -------
            residuals : (N) array
                Z residual for each data point.
            """

            x, y, z = data.T
            C = self.params

            # evaluate model at points
            args = (x, y, z)
            self.model_data = self.my_model(self.params, *args)
            z_error = self.model_data - z
            return z_error

    return FuncModel


class _Spline3dModel(BaseModel):
    """Smoothing spline fit to 3-D point cloud.

    Attributes
    ----------
    model_data : array
        Fitted model.
    params : scipy.interpolate.fitpack2.SmoothBivariateSpline
        Spline class.

    """
    
    def my_model(self, p, *args):
        # p is spline class
        (x, y, z) = args
        m = p(x, y, grid=False)
        return m
    
    def estimate(self, data):
        """Estimate model from data.

        Parameters
        ----------
        data : (N, D) array
            Flattened length N (x,y,z) point cloud to fit to.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """

        try:
            params = {'kx' : 3,
                        'ky' : 3,
                        's': None}
            
            # check if it param_dict exists
            try:
                self.param_dict
            except AttributeError:
                self.param_dict = None
            if self.param_dict is None:
                self.param_dict = {}
            params.update(self.param_dict)
        
            # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.interpolate.SmoothBivariateSpline.html
            x, y, z = data.T
            spl = SmoothBivariateSpline(x, y, z, **params)
            self.params = spl
            return True
        except Exception as e:
            print(e)
            return False


    def residuals(self, data):
        """Determine residuals of data to model.

        Parameters
        ----------
        data : (N, D) array
            Flattened length N (x,y,z) point cloud to fit to.

        Returns
        -------
        residuals : (N) array
            Z residual for each data point.

        """

        x, y, z = data.T
        # evaluate model at points
        args = (x, y, z)
        self.model_data = self.my_model(self.params, *args)
        z_error = self.model_data - z
        return z_error



def ransac_im_fit(im, mode=1, residual_threshold=0.1, min_samples=10,
                  max_trials=1000, model_f=None, p0=None, mask=None,
                  scale=False, fract=1, param_dict=None, plot=False,
                  axes=(-2, -1)):
    '''
    Fits a plane, polynomial, convex paraboloid, arbitrary function, or
    smoothing spline to an image using the RANSAC algorithm.

    Parameters
    ----------
    im : ndarray
        ndarray with images to fit to.
    mode : integer [0:4]
        Specifies model used for fit.
        0 is function defined by `model_f`.
        1 is plane.
        2 is quadratic.
        3 is concave paraboloid with offset.
        4 is smoothing spline.
    model_f : callable or None
        Function to be fitted.
        Definition is model_f(p, *args), where p is 1-D iterable of params
        and args is iterable of (x, y, z) arrays of point cloud coordinates.
        See examples.
    p0 : tuple
        Initial guess of fit params for `model_f`.
    mask : 2-D boolean array
        Array with which to mask data. True values are ignored.
    scale : bool
        If True, `residual_threshold` is scaled by stdev of `im`.
    fract : scalar (0, 1]
        Fraction of data used for fitting, chosen randomly. Non-used data
        locations are set as nans in `inliers`.

    residual_threshold : float
        Maximum distance for a data point to be classified as an inlier.
    min_samples : int or float
        The minimum number of data points to fit a model to.
        If an int, the value is the number of pixels.
        If a float, the value is a fraction (0.0, 1.0] of the total number of pixels.
    max_trials : int, optional
        Maximum number of iterations for random sample selection.
    param_dict : None or dictionary.
        If not None, the dictionary is passed to the model estimator.
        For arbitrary functions, this is passed to scipy.optimize.leastsq.
        For spline fitting, this is passed to scipy.interpolate.SmoothBivariateSpline.
        All other models take no parameters.
    plot : bool
        If True, the data, including inliers, model, etc are plotted.
    axes : length 2 iterable
        Indices of the input array with images.
   
    Returns
    -------
    Tuple of fit, inliers, n, where:
    fit : 2-D array
        Image of fitted model.
    inliers : 2-D array
        Boolean array describing inliers.
    n : array or None
        Normal of plane fit. `None` for other models.

    Notes
    -----
    See skimage.measure.ransac for details of RANSAC algorithm.

    `min_samples` should be chosen appropriate to the size of the image
    and to the variation in the image.

    Increasing `residual_threshold` increases the fraction of the image
    fitted to.
   
    The entire image can be fitted to without RANSAC by setting:
    max_trials=1, min_samples=1.0, residual_threshold=`x`, where `x` is a
    suitably large value.

    Examples
    --------
    `model_f` for paraboloid with offset:

    >>> def model_f(p, *args):
    ...     (x, y, z) = args
    ...     m = np.abs(p[0])*((x-p[1])**2 + (y-p[2])**2) + p[3]
    ...     return m
    >>> p0 = (0.1, 10, 20, 0)


    To plot fit, inliers etc:

    >>> from fpd.ransac_tools import ransac_im_fit
    >>> import matplotlib as mpl
    >>> from numpy.ma import masked_where
    >>> import numpy as np
    >>> import matplotlib.pylab as plt
    >>> plt.ion()


    >>> cmap = mpl.cm.gray
    >>> cmap.set_bad('r')

    >>> image = np.random.rand(*(64,)*2)
    >>> fit, inliers, n = ransac_im_fit(image, mode=1)
    >>> cor_im = image-fit
   
    >>> pct = 0.5
    >>> vmin, vmax = np.percentile(cor_im, [pct, 100-pct])
    >>>
    >>> f, axs = plt.subplots(1, 4, sharex=True, sharey=True)
    >>> _ = axs[0].matshow(image, cmap=cmap)
    >>> _ = axs[1].matshow(masked_where(inliers==False, image), cmap=cmap)
    >>> _ = axs[2].matshow(fit, cmap=cmap)
    >>> _ = axs[3].matshow(cor_im, vmin=vmin, vmax=vmax)



    To plot plane normal vs threshold:

    >>> from fpd.ransac_tools import ransac_im_fit
    >>> from numpy.ma import masked_where
    >>> import numpy as np
    >>> from tqdm import tqdm
    >>> import matplotlib.pylab as plt
    >>> plt.ion()

    >>> image = np.random.rand(*(64,)*2)
    >>> ns = []
    >>> rts = np.logspace(0, 1.5, 5)
    >>> for rt in tqdm(rts):
    ...     nis = []
    ...     for i in range(64):
    ...         fit, inliers, n = ransac_im_fit(image, residual_threshold=rt, max_trials=10)
    ...         nis.append(n)
    ...     ns.append(np.array(nis).mean(0))
    >>> ns = np.array(ns)

    >>> thx = np.arctan2(ns[:,1], ns[:,2])
    >>> thy = np.arctan2(ns[:,0], ns[:,2])
    >>> thx = np.rad2deg(thx)
    >>> thy = np.rad2deg(thy)

    >>> _ = plt.figure()
    >>> _ = plt.semilogx(rts, thx)
    >>> _ = plt.semilogx(rts, thy)

    '''
   
   
    # Set model
    # Functions defining classes are needed to pass parameters since class must
    # not be instantiated or are monkey patched (only in spline implementation)
    if mode == 0:
        # generate model_class with passed function
        if p0 is None:
            raise NotImplementedError('p0 must be specified.')
        model_class = _model_class_gen(model_f, p0, param_dict)
    elif mode == 1:
        # linear
        model_class = _Plane3dModel
    elif mode == 2:
        # quadratic
        model_class = _Poly3dModel
    elif mode == 3:
        # concave paraboloid
        model_class = _Poly3dParaboloidModel
    elif mode == 4:
        # spline
        class _Spline3dModel_monkeypatched(_Spline3dModel):
            pass
        model_class = _Spline3dModel_monkeypatched
        model_class.param_dict = param_dict
   
    multiim = False
    if im.ndim > 2:
        multiim = True
        ims, unflat_shape = seq_image_array(im, axes)
        pbar = tqdm(total=ims.shape[0])
    else:
        ims = im[None]
   
    fits = []
    inlierss = []
    ns = []
    
    for imi in ims:
        # set data
        yy, xx = np.indices(imi.shape)
        zz = imi
        if mask is None:
            keep = (np.ones_like(imi)==1).flatten()
        else:
            keep = (mask==False).flatten()
        data = np.column_stack([xx.flat[keep], yy.flat[keep], zz.flat[keep]])
       
        if type(min_samples) is int:
            # take number directly
            pass
        else:
            # take number as fraction
            min_samples = int(len(keep)*min_samples)
            print("min_samples is set to: %d" %(min_samples))

        # randomly select data
        sel = np.random.rand(data.shape[0]) <= fract
        data = data[sel.flatten()]

        # scale residual, if chosen
        if scale:
            residual_threshold = residual_threshold * np.std(data[:,2])
        
        # determine if fitting to all
        full_fit = min_samples == data.shape[0]
    
        if not full_fit:
            # do ransac fit
            model, inliers = ransac(data=data,
                                    model_class=model_class,
                                    min_samples=min_samples,
                                    residual_threshold=residual_threshold,
                                    max_trials=max_trials)
        else:
            model = model_class()
            inliers = np.ones(data.shape[0]) == 1
        
        # get params from fit with all inliers
        model.estimate(data[inliers])
        # get model over all x, y
        args = (xx.flatten(), yy.flatten(), zz.flatten())
        fit = model.my_model(model.params, *args).reshape(imi.shape)
           
        if mask is None and fract==1:
            inliers = inliers.reshape(imi.shape)
        else:
            inliers_nans = np.empty_like(imi).flatten()
            inliers_nans[:] = np.nan
            yi = np.indices(inliers_nans.shape)[0]

            sel_fit = yi[keep][sel.flatten()]
            inliers_nans[sel_fit] = inliers
            inliers = inliers_nans.reshape(imi.shape)

        # calculate normal for plane
        if mode == 1:
            # linear
            C = model.params
            n = np.array([-C[0], -C[1], 1])
            n_mag = np.linalg.norm(n, ord=None, axis=0)
            n = n/n_mag
        else:
            # non-linear
            n = None
       
        if plot:
            import matplotlib.pylab as plt
            import matplotlib as mpl
            from numpy.ma import masked_where
            from mpl_toolkits.axes_grid1 import ImageGrid
           
            plt.ion()
            cmap = mpl.cm.gray
            cmap = copy.copy(cmap)
            cmap.set_bad('r')

            cor_im = imi - fit
            pct = 0.1
            vmin, vmax = np.percentile(cor_im, [pct, 100-pct])

            fig = plt.figure()
            grid = ImageGrid(fig, 111,
                            nrows_ncols=(1, 4),
                            axes_pad=0.1,
                            share_all=True,
                            label_mode="L",
                            cbar_location="right",
                            cbar_mode="single")
           
            images = [imi, masked_where(inliers==False, imi), fit, cor_im]
            titles = ['Image', 'Inliers', 'Fit', 'Corrected']
            for i, image in enumerate(images):
                img = grid[i].imshow(image, cmap=cmap, interpolation='nearest')
                grid[i].set_title(titles[i])
            img.set_clim(vmin, vmax)
            plt.colorbar(img, cax=grid.cbar_axes[0])


            #f, axs = plt.subplots(1, 4, sharex=True, sharey=True)
            #_ = axs[0].matshow(imi, cmap=cmap)
            #_ = axs[1].matshow(masked_where(inliers==False, imi), cmap=cmap)
            #_ = axs[2].matshow(fit, cmap=cmap)
            #_ = axs[3].matshow(cor_im, vmin=vmin, vmax=vmax)
           
            #for i, title in enumerate(['Image' , 'Inliers', 'Fit', 'Corrected']):
                #axs[i].set_title(title)
            #plt.tight_layout()
        fits.append(fit)
        inlierss.append(inliers)
        ns.append(n)
        if multiim:
            pbar.update(1)
    fit = np.array(fits)
    inliers = np.array(inlierss)
    n = np.array(ns)
   
    if multiim:
        pbar.close()
        
        #reshape
        fit = unseq_image_array(fit, axes, unflat_shape)
        inliers = unseq_image_array(inliers, axes, unflat_shape)
        n = unseq_image_array(n, axes, unflat_shape)
    else:
        fit = fit[0]
        inliers = inliers[0]
        n = n[0]
   
    return (fit, inliers, n)



class _FuncModel_1d(BaseModel):
    """Least squares fit of arbitrary function to xy_data.

    Attributes
    ----------
    model_data : array
        Fitted model.
    params : array
        Model params.
    pcov : array
        Model fit covarience matrix.

    """
    
    def my_model(self, p, x):
        m = self.model_f(p, x)
        return m
    
    def my_fit(self, x, *p):
        m = self.my_model(p, x)
        return m
    
    def param_dict_f(self):
        return(param_dict)

    def estimate(self, data):
        """Estimate model from data.

        Parameters
        ----------
        data : (N, D) array
            Flattened length N (x,y,s) point cloud to fit to, where s is sigma.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """
        params = {'maxfev': 10000}
        
        param_dict = self.param_dict
        if param_dict is None:
            param_dict = {}
        params.update(param_dict)
        
        # checks for validity of selected data would go here
        if False:
            return False
        else:
            x, y, s = data.T
            try:
                popt, pcov = sp.optimize.curve_fit(f=self.my_fit,
                                                   xdata=x, 
                                                   ydata=y,
                                                   p0=self.p0,
                                                   sigma=s,
                                                   **params)
                rtn = True
                self.params = popt
                self.pcov = pcov
            except Exception as e:
                rtn = False
                #print(e)
            finally:
                return rtn
    
    def residuals(self, data):
        """Determine residuals of data to model.

        Parameters
        ----------
        data : (N, D) array
            Flattened length N (x,y,s) point cloud to fit to, where s is sigma

        Returns
        -------
        residuals : (N) array
            Z residual for each data point.
        """

        x, y, s = data.T
        # evaluate model at points
        self.model_data = self.my_model(self.params, x)
        z_error = self.model_data - y
        return z_error


class _Linear1dModel(BaseModel):
    """Total least squares estimator for line fit to 1-D points.

    Attributes
    ----------
    model_data : array
        Fitted model.
    params : array
        model = C[0]*x + C[1]

    """

    def my_model(self, p, x):
        m = p[0]*x + p[1]
        return m

    def estimate(self, data):
        """Estimate model from data.

        Parameters
        ----------
        data : (N, 2) array
            x,y data to fit to.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """

        if False:
            return False
        else:
            # best-fit linear plane
            # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.lstsq.html
            # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.lstsq.html
            
            x, y = data.T
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            
            self.params = (m, c)
            return True


    def residuals(self, data):
        """Determine residuals of data to model.

        Parameters
        ----------
        data : (N, 2) array
            x,y data to fit to.

        Returns
        -------
        residuals : (D) array
            Z residual for each data point.

        """

        x, y = data.T
        # evaluate model at points
        self.model_data = self.my_model(self.params, x)
        z_error = self.model_data - y
        return z_error


class _Quadratic1dModel(BaseModel):
    """Total least squares estimator for quadratic fit to 1-D points.

    Attributes
    ----------
    model_data : array
        Fitted model.
    params : array
        model = C[0]*x**2 + C[1]*x + C[2]

    """

    def my_model(self, p, x):
        m = p[0]*x**2 + p[1]*x + p[2]
        return m

    def estimate(self, data):
        """Estimate model from data.

        Parameters
        ----------
        data : (N, 2) array
            x,y data to fit to.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """

        if False:
            return False
        else:
            # best-fit linear plane
            # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.lstsq.html
            # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.lstsq.html
            
            x, y = data.T            
            A = np.vstack([x**2, x, np.ones(len(x))]).T
            params = np.linalg.lstsq(A, y, rcond=None)[0]
            
            self.params = params
            return True


    def residuals(self, data):
        """Determine residuals of data to model.

        Parameters
        ----------
        data : (N, 2) array
            x,y data to fit to.

        Returns
        -------
        residuals : (D) array
            Z residual for each data point.

        """

        x, y = data.T
        # evaluate model at points
        self.model_data = self.my_model(self.params, x)
        z_error = self.model_data - y
        return z_error


class _Cubic1dModel(BaseModel):
    """Total least squares estimator for cubic fit to 1-D points.

    Attributes
    ----------
    model_data : array
        Fitted model.
    params : array
        model = C[0]*x**3 + C[1]*x**2 + C[2]*x + C[3]

    """

    def my_model(self, p, x):
        m = p[0]*x**3 + p[1]*x**2 + p[2]*x +p[3]
        return m

    def estimate(self, data):
        """Estimate model from data.

        Parameters
        ----------
        data : (N, 2) array
            x,y data to fit to.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """

        if False:
            return False
        else:
            # best-fit linear plane
            # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.lstsq.html
            # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.lstsq.html
            
            x, y = data.T            
            A = np.vstack([x**3, x**2, x, np.ones(len(x))]).T
            params = np.linalg.lstsq(A, y, rcond=None)[0]
            
            self.params = params
            return True


    def residuals(self, data):
        """Determine residuals of data to model.

        Parameters
        ----------
        data : (N, 2) array
            x,y data to fit to.

        Returns
        -------
        residuals : (D) array
            Z residual for each data point.

        """

        x, y = data.T
        # evaluate model at points
        self.model_data = self.my_model(self.params, x)
        z_error = self.model_data - y
        return z_error


class _Spline1dModel(BaseModel):
    """Smoothing spline fit to 1-D point cloud.

    Attributes
    ----------
    model_data : array
        Fitted model.
    params : scipy.interpolate.fitpack2.SmoothBivariateSpline
        Spline class.

    """
    
    def my_model(self, p, x):
        # p is spline class
        m = p(x)
        return m
    
    def estimate(self, data):
        """Estimate model from data.

        Parameters
        ----------
        data : (N, 2) array
            x,y data to fit to.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """

        try:
            params = {'k' : 3,
                      's': None}
            
            # check if it param_dict exists
            try:
                self.param_dict
            except AttributeError:
                self.param_dict = None
            if self.param_dict is None:
                self.param_dict = {}
            params.update(self.param_dict)
        
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html
            x, y = data.T
            spl = UnivariateSpline(x, y, **params)
            self.params = spl
            return True
        except Exception as e:
            print(e)
            return False


    def residuals(self, data):
        """Determine residuals of data to model.

        Parameters
        ----------
        data : (N, 2) array
            x,y data to fit to.

        Returns
        -------
        residuals : (N) array
            Z residual for each data point.

        """
        
        x, y = data.T
        # evaluate model at points
        self.model_data = self.my_model(self.params, x)
        z_error = self.model_data - y
        return z_error



def ransac_1D_fit(x, y, mode=1, residual_threshold=0.1, min_samples=10,
                  max_trials=1000, model_f=None, p0=None, mask=None,
                  scale=False, fract=1, param_dict=None, plot=False):
    '''
    Fits a straight line, quadratic, arbitrary function, or
    smoothing spline to 1-D data using the RANSAC algorithm.

    Parameters
    ----------
    x, y : ndarrays
        1-D x and y data to fit to.
    mode : integer [0:4]
        Specifies model used for fit.
        0 is function defined by `model_f`.
        1 is linear.
        2 is quadratic.
        3 is cubic.
        4 is smoothing spline.
    model_f : callable or None
        Function to be fitted.
        Definition is model_f(p, x), where p is 1-D iterable of params
        and x is the x-data array.
        See examples.
    p0 : tuple
        Initial guess of fit params for `model_f`.
    mask : 1-D boolean array
        Array with which to mask data. True values are ignored.
    scale : bool
        If True, `residual_threshold` is scaled by stdev of `y`.
    fract : scalar (0, 1]
        Fraction of data used for fitting, chosen randomly. Non-used data
        locations are set as nans in `inliers`.

    residual_threshold : float
        Maximum distance for a data point to be classified as an inlier.
    min_samples : int or float
        The minimum number of data points to fit a model to.
        If an int, the value is the number of pixels.
        If a float, the value is a fraction (0.0, 1.0] of the total number of pixels. 
    max_trials : int, optional
        Maximum number of iterations for random sample selection.
    param_dict : None or dictionary.
        If not None, the dictionary is passed to the model estimator.
        For arbitrary functions, this is passed to scipy.optimize.curve_fit.
        For spline fitting, this is passed to scipy.interpolate.UnivariateSpline.
        All other models take no parameters.
    plot : bool
        If True, the data, including inliers, model, etc are plotted.
    
    Returns
    -------
    Tuple of fit, inliers, model where:
    fit : 1-D array
        y-data of fitted model.
    inliers : 1-D array
        Boolean array describing inliers.
    model : class
        Model used in the fitting. For polynomial fits, model.params contains a tuple
        of the fit coefficients in decreasing power.

    Notes
    -----
    See skimage.measure.ransac for details of RANSAC algorithm.

    `min_samples` should be chosen appropriate to the size of the data
    and to the variation in the data.

    Increasing `residual_threshold` increases the fraction of the data
    fitted to.
    
    The entire data can be fitted to without RANSAC by setting:
    max_trials=1, min_samples=1.0, residual_threshold=`x`, where `x` is a
    suitably large value.

    Examples
    --------
    `model_f` for quadratic:

    >>> def model_f(p, x):
    ...     return p[0]*x**2 + p[1]*x + p[2]
    >>> p0 = (1,)*3

    '''
    
    
    # Set model
    # Functions defining classes are needed to pass parameters since class must
    # not be instantiated or are monkey patched (only in spline implementation)
    if mode == 0:
        # generate model_class with passed function
        class _FuncModel_1d_monkeypatched(_FuncModel_1d):
            pass
        model_class = _FuncModel_1d_monkeypatched
        # parse sigma
        # added to data for ransac to use s correctly
        s = None
        if param_dict is not None:
            s = param_dict.pop('sigma', None)
        if s is None:
            # not specified, so create it
            s = np.ones_like(x)
        model_class.param_dict = param_dict
        if p0 is None:
            raise NotImplementedError('p0 must be specified.')
        model_class.p0 = p0
        def model_f_wrap(self, p, x):
            return model_f(p, x)
        model_class.model_f = model_f_wrap
    elif mode == 1:
        # linear
        model_class = _Linear1dModel
    elif mode == 2:
        # quadratic
        model_class = _Quadratic1dModel
    elif mode == 3:
        # cubic
        model_class = _Cubic1dModel
    elif mode == 4:
        # spline
        class _Spline1dModel_monkeypatched(_Spline1dModel):
            pass
        model_class = _Spline1dModel_monkeypatched
        model_class.param_dict = param_dict

    # set data
    if mask is None:
        keep = (np.ones_like(x)==1).flatten()
    else:
        keep = (mask==False).flatten()
    if mode != 0:    
        data = np.column_stack([x.flat[keep], y.flat[keep]]).T
    else:
        data = np.column_stack([x.flat[keep], y.flat[keep], s.flat[keep]]).T
        
    if type(min_samples) is int:
        # take number directly
        pass
    else:
        # take number as fraction
        min_samples = int(len(keep)*min_samples)
        print("min_samples is set to: %d" %(min_samples))
    
    # randomly select data
    sel = np.random.rand(data.shape[1]) <= fract
    data = data[:, sel.flatten()]
    
    # scale residual, if chosen
    if scale:
        residual_threshold = residual_threshold * np.std(data[1])
    
    # determine if fitting to all    
    full_fit = min_samples == data.shape[1]
    
    if not full_fit:    
        # do ransac fit
        model, inliers = ransac(data=data.T,
                                model_class=model_class,
                                min_samples=min_samples,
                                residual_threshold=residual_threshold,
                                max_trials=max_trials)
    else:
        model = model_class()
        inliers = np.ones(data.shape[1]) == 1
    
    # get params from fit with all inliers
    model.estimate(data[:, inliers].T)
    # get model over all x
    fit = model.my_model(model.params, x)
    
    if mask is None and fract==1:
        inliers = inliers
    else:
        inliers_nans = np.empty_like(fit)
        inliers_nans[:] = np.nan
        xi = np.indices(inliers_nans.shape)[0]

        sel_fit = xi[keep][sel.flatten()]
        inliers_nans[sel_fit] = inliers
        inliers = inliers_nans
        inliers = inliers==1

    if plot:
        import matplotlib.pylab as plt
        plt.ion()
        
        outliers = inliers==False
        
        plt.figure()
        plt.plot(x[inliers], y[inliers], 'bo', label='Inliers')
        plt.plot(x[outliers], y[outliers], 'ro', label='Outliers')
        plt.plot(x, fit, 'k-', label='Fit')
        plt.legend(loc=0)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title("mode=%d" %(mode))
        plt.show()

    return (fit, inliers, model.params)


