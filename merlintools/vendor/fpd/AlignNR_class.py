
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter, gaussian_filter1d
import threading
import warnings
import multiprocessing as mp
from collections.abc import Iterable
from threadpoolctl import threadpool_limits

import matplotlib.pylab as plt
plt.ion()

from .fpd_processing import nrmse


class AlignNR:
    def __init__(self, images, reference='mean', dyx0=None, dyxi_max=1.0,
                 alpha=10, niter=200, nrmse_min=0.1, nrmse_rel=1e-5, nrmse_mode='quad',
                 pct=0, grad_sigma=1, reg_mode='gaussian', reg_sigma=2, reg_kwargs={},
                 cval=np.nan, ishift=False, print_stats=False, plot=True, block=True,
                 nthreads=None):
        '''
        Non-rigid image alignment by gradient decent. 
        
        This approach is similar to Smart Align [1]. SimpleElastix is from 
        the medical imaging world [2] and has python bindings to the advanced
        elastix C++ library, but you have to compile and roll your own code.
        The class here is pure python.
        
        [1] Jones et al., Advanced Structural and Chemical Imaging 1, 8 (2015).
            https://doi.org/10.1186/s40679-015-0008-4
        [2] http://simpleelastix.github.io
        
        Parameters
        ----------
        images : ndarray
            Image(s) to be aligned, of shape [[n,] y, x]. If a single 2D
            image, `ref_im` must not be None. For multiple images, the first
            axis is the stack dimension.
        reference : ndarray, string, or int 
            The reference image or mode by which it is obtained from `images`.
            If an integer, it is the index of the `images` stack used. If an
            ndarray, it is taken as the reference image. If a string, it may
            be one of ['mean', 'median']:
            - 'mean' : the average of the image stack.
            - 'median' : the median of the stack.
        dyx0 : None or ndarray
            The initial deformation field, of shape [[n,] y, x]. If `images`
            is 3D and dyx0 is 2D, it is reshaped to 3D and each deformation
            field optimised independently. If None, it is taken as zero.
        dyxi_max : scalar or None:
            Magnitude of maximum incremental displacement field value along axes,
            i.e. [-dyxi_max, dyxi_max]. This is applied before further regularisation.
        alpha : scalar
            Correction coefficient at each step in pixels / grad. The larger the
            number, the faster the alignment converges. Too high of a value 
            may cause instabilities.
        niter : integer
            The maximum number of iterations. Whichever of `niter`, `nrmse_min`
            and `nrmse_rel` is met first controls the stop criterion.
        nrmse_min : scalar
            Normalised root mean square error (NRMSE) below which alignment stops.
            If `images` is a stack, the mean value is used. Whichever of `niter`,
            `nrmse_min` and `nrmse_rel` is met first controls the stop criterion.
        nrmse_rel : scalar
            The change in NRMSE between iterations below which alignment stops.
            If `images` is a stack, the mean value is used. Whichever of `niter`,
            `nrmse_min` and `nrmse_rel` is met first controls the stop criterion.
        nrmse_mode : string
            One of ['mean', 'median', 'max', 'quad'], the method of determining the NRMSE. 
        pct : None, scalar or iterable
            The percentile with which `images` and `ref_im` intensity is normalised
            and cropped to, i.e. [pct, 100-pct]. If an iterable of length 2, it is
            taken as the lower and upper pct values. It is applied to each image
            separately. If None, no scaling is applied.
        grad_sigma : scalar
            The Gaussian standard deviation of the gradient of Gaussian used to
            calculate the image derivatives. If 0, no smoothing is used.
            Increasing `grad_sigma` will reduce noise.
        reg_mode : string, callable or iterable
            Mode of regularising the distortion fields between iterations. If a
            callable, the form is dyxi = f(dyxi, **reg_kwargs) where 'dyxi'
            is a 2D array. Parameters may be passed through `reg_kwargs`. See
            Notes. If an iterable, it is of two callables which are applied to
            the y- and x- distortion fields, respectively. If singular, the same
            function is applied across both axes. See notes. If a string,
            possible values are ['gaussian'].
            - gaussian : 2D Gaussian convolution of width `reg_sigma`.
            See also `dyxi_max`.
        reg_kwargs : dict
            Keyword arguments passed to `reg_mode` if it is a callable.
        reg_sigma : scalar
            Standard deviation of Gaussian used for regularisation. See `ref_mode`.
        cval : float
            Fill value used for void pixels.
        ishift : bool
            If True, intensity modification is approximated at each step. As this
            involves derivative calculations, the use of periodic images may reduce
            edge effects. The image intensity change can be applied after alignment
            with:     
                im *= 1 + np.gradient(dyx[0], axis=0) + np.gradient(dyx[1], axis=1)
        print_stats : bool
            If True, print statistics during alignment.
        plot : bool
            If True, the results are plotted during iteration. This is currently
            rather simple. Note that not every displacement point in the quiver plot
            is plotted for clarity. This will block until closed. The `plot' method
            may also be called at any time.
        block : bool
            If True, the alignment thread will block the main thread until it
            completes. Setting to True allows can be useful for incorporating
            into a script where the results must be available before continuing.
            If `plot` is True, the class always blocks, but can be aborted through
            the GUI button.
        nthreads : None or int
            If None, an optimum number of threads are used to process the images
            in parallel. Set to 1 for single threaded, and greater than 1 for
            the specified number of cores.
        
        Notes
        -----
        When passing a callable to `reg_mode` there are many possible functions.
        For example, to apply edge preserving smoothing denoise_tv_chambolle from
        skimage.restoration may be passed; the net shifts may be constrained to
        zero by passing a function that removes the mean; or the modulation may be
        limited to certain axes.
        
        Example of smoothed spline:
        
        from scipy.interpolate import SmoothBivariateSpline
        def myf(z):
            yi, xi = np.indices(z.shape)
            s = np.random.choice(z.size, int(z.size / 8.0))
            S = SmoothBivariateSpline(xi.flat[s], yi.flat[s], z.flat[s], kx=3, ky=3, s=None)
            z = S.ev(xi.flatten(), yi.flatten()).reshape(z.shape)
            return z
        
        Attributes
        ----------
        images : ndarray
            The original images.
        i : integer
            Alignment loop iteration number.
        nrmse : float
            The current NRMSE. See `nrmse_mode`.
        dnrmse : float
            Delta in `nrmse` between iterations. See `nrmse_mode`.
        criteria_bool : tuple
            Tuple of criteria boolean states: (i >= niter), (nrmse < nrmse_min),
            and (delta rms) < nrmse_rel).
        dyx : ndarray
            The distortion field.
        images_aligned : ndarray
            The aligned images.
        interrupt : bool
            If set to True, the alignment will be stopped once the current
            iteration is complete.
        alpha : scalar
            See input parameter docstring.
        
        Methods
        -------
        warp
            Warp images according to `dyx`.
        plot
            Raise the GUI.
            
        '''
        
        
        '''        
        Future TODO list
        ----------------
        - plotting! it is currently rather basic
            - use timer rather than plt.pause. Would allow faster interaction.
              Need to keep mpl in main thread and avoid manual updates, so
              maybe have timer in thread set event and have main thread handle
              events.
            - make plots faster by setting data
        
        - make align start a method? Would potentially allow an additional layer
          of control; eg multiple down samples, stepped alpha, ...
        - adding properties would also help with interdependent changing variables
        
        - If ref is not updating, only operate on individual images that
          don't meet criteria?
        - Allow a moving ref, eg neighbours?
        
        - Helmholtz-Hodge decomposition for curl and div weighting of incremental dyx.
        
        - vary alpha per-pixel
        - vary alpha depending on error
        
        - buffer for stop criterion to allow averages to be used
        
        - remove normalisation from images and apply to grads instead?
        
        - could have variable controlling if ref_mode expects dyx together
          or dy and dx separately.
        - reg_kwargs could be iterable
          
        - downsample and upsample for speed
          from skimage.transform import rescale, downscale_local_mean
        '''
        
        # parameters from input
        self.images = images
        self._images_normed = images.copy()
        self._reference = reference
        self._dyx0 = dyx0
        self._dyxi_max = np.abs(dyxi_max)
        self.alpha = alpha
        self._niter = niter
        self._nrmse_min = nrmse_min
        self._nrmse_rel = nrmse_rel
        self._nrmse_mode = nrmse_mode.lower()
        self._pct = pct
        self._grad_sigma = grad_sigma
        self._reg_mode = reg_mode
        self.reg_sigma = reg_sigma
        self._reg_kwargs = reg_kwargs
        self.cval = cval
        self._ishift = ishift
        self._print_stats = print_stats
        self._plot = plot
        self._block = block
        self._nthreads = nthreads
        
        # initialise parameters
        self.i = 0
        self.nrmse = np.inf
        self.dnrmse = np.inf
        self.criteria_bool = (False,)*3
        
        # thread stuff
        self._data_lock = threading.Lock()
        self._interrupt_evnt = threading.Event()
        self._arrays_prepared = threading.Event()
        self._align_finished = threading.Event()
        self._fig_closed = threading.Event()
        
        # initialise properties
        self.interrupt = False
        
        # possible options
        self._ref_modes = ['mean', 'median']
        self._reg_modes = ['gaussian']
        self._nrmse_modes = ['mean', 'median', 'max', 'quad']
        
        # prepare and run
        self._condition_params()
        self._run_align()
        if self._plot:
            self.plot()
        
    @property
    def interrupt(self):
        return self._interrupt_evnt.is_set()

    @interrupt.setter
    def interrupt(self, val):
        if val:
            self._interrupt_evnt.set()
        else:
            self._interrupt_evnt.clear()
    
    def _run_align(self):
        align_thread = threading.Thread(target=self._align)
        align_thread.start()
        if self._block and not self._plot:
            # align must not block so that plot can run in main thread
            align_thread.join()
    
    
    def plot(self):
        '''
        Raise the GUI.
        
        Mouse scroll can be used to navigate the stack slices, and
        the reg_sigma and alpha values.
        
        '''
        
        # wait for initial data to be ready
        self._arrays_prepared.wait()
        
        
        # plots
        fig = plt.figure(figsize=(10, 5))#constrained_layout=True)
        gs = fig.add_gridspec(3, 3)
        ax0 = fig.add_subplot(gs[:, 0])
        ax1 = fig.add_subplot(gs[:, 1], sharex=ax0, sharey=ax0)
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[1, 2], sharex=ax2)
        ax4 = fig.add_subplot(gs[2, 2], sharex=ax3)
        
        # images
        ax0.set_aspect(1)
        ax0.get_xaxis().set_visible(False)
        ax0.get_yaxis().set_visible(False)
        
        # quiver
        ax1.set_aspect(1)
        ax1.invert_yaxis()
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        tit = ax1.set_title('i: %02d, nrmse: %0.4f, dyx_max: %0.4f' %(self.i, self.nrmse, 0))
        
        ## 2nd column
        ax4.set_xlabel('iteration (i)')
        xlim_max = self._niter+10
        
        # error
        ax2.set_ylabel('nrmse')
        ax2.axhline(self._nrmse_min, ls=':', color='r')
        ax2.axvline(self._niter, ls=':', color='r')
        ax2.axis([0, xlim_max, 0, 1.0])
        ax2.autoscale(False)
        
        # derror
        ax3_ymin = self._nrmse_rel
        if ax3_ymin <= 0:
            ax3_ymin = 1e-9
        ax3.set_ylabel('|dnrmse|')
        if self._nrmse_rel >0:
            ax3.axhline(self._nrmse_rel, ls=':', color='r')
        ax3.axvline(self._niter, ls=':', color='r')
        ax3.axis([0, xlim_max, ax3_ymin, 1.0])
        ax3.set_yscale('log')
        ax3.autoscale(False)
        
        # dyx metric
        ax4.set_ylabel('<dyx>')
        ax4.axvline(self._niter, ls=':', color='r')
        ax4.axis([0, xlim_max, 0, 10.0])
        ax4.autoscale(False)
        
        self._plt_im = None
        self._plt_q = None
        self._plt_set = False
        self._ax2_plt_limits_set = False
        self._ax3_plt_limits_set = False
        def update_plot(val):
            if not self._ax2_plt_limits_set and np.isfinite(self.nrmse):
                ax2.axis([0, xlim_max, 0, self.nrmse+0.1])
                self._ax2_plt_limits_set = True
            if not self._ax3_plt_limits_set and np.isfinite(self.dnrmse):
                ax3.axis([0, xlim_max, ax3_ymin, self.dnrmse+0.1])
                self._ax3_plt_limits_set = True
            
            i = sslice.val
            
            # get data
            with self._data_lock:
                dy, dx = self.dyx[i].copy()
                imi = self.images_aligned[i].copy()
            
            # image
            try:
                self._plt_im.remove()
            except:
                pass
            self._plt_im = ax0.matshow(imi)
            
            # quiver
            dyxr = np.hypot(dy, dx)
            shape = dy.shape
            
            s = np.ceil(np.product(shape)**0.5 / 24).astype(int)
            s = max(s, 1)
            s = np.s_[::s, ::s]
            try:
                self._plt_q.remove()
            except:
                pass
            self._plt_q = ax1.quiver(self._cx[s], self._cy[s], dx[s], -dy[s], scale=None, pivot='tail', scale_units='xy', headwidth=6, headlength=8)
            tit.set_text('i: %02d, nrmse: %0.4f, dyx_max: %0.4f' %(self.i, self.nrmse, dyxr.max()))
            
            if self._plt_set == False:
                ax0.autoscale(False)
                ax1.autoscale(False)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    plt.tight_layout()
                plt.subplots_adjust(bottom=0.25)
            self._plt_set = True
            
            # lines
            with self._data_lock:
                ax2_pd = self._nrmse_ax
                ax3_pd = np.abs(self._dnrmse_ax)
                ax4_pd = self._dyx_metric_ax[i]
            
            # error etc
            for axi, pdi, ci in zip([ax2, ax3, ax4],
                                    [ax2_pd, ax3_pd, ax4_pd],
                                    ['b-', 'g-', 'r-']):
                try:
                    for lni in axi.get_lines():
                        if lni.get_linestyle() == '-':
                            lni.remove()
                except:
                    pass
                axi.plot(pdi, ci)
        
        # add button to set interrupt
        from matplotlib.widgets import Button, Slider
        ax_interrupt = plt.axes([0.05, 0.05, 0.1, 0.075])
        self._plt_binterrupt = Button(ax_interrupt, 'Halt')
        def binterrupt_handler(event):
            self._interrupt_evnt.set()
        self._plt_binterrupt.on_clicked(binterrupt_handler)
        
        # add slider for slice selection
        axcolor = 'lightgoldenrodyellow'
        axslice = plt.axes([0.25, 0.03, 0.65, 0.03], facecolor=axcolor)
        with warnings.catch_warnings():
            if self._stack_len==1:
                warnings.simplefilter("ignore")
            sslice = Slider(axslice, 'Slice', 0, self._stack_len-1, valinit=0, valstep=1)
        sslice.on_changed(update_plot)
        sslice.my_scroll_delta = 1
        
        # add alpha slider
        def set_alpha(val):
            self.alpha = val
        axalpha = plt.axes([0.25, 0.07, 0.65, 0.03], facecolor=axcolor)
        salpha = Slider(axalpha, 'alpha', 0.1, max(self.alpha*4, 50), valinit=self.alpha)
        salpha.on_changed(set_alpha)
        salpha.my_scroll_delta = 0.5
        
        # add sigma slider
        def set_reg_sigma(val):
            self.reg_sigma = val
        axregsig = plt.axes([0.25, 0.11, 0.65, 0.03], facecolor=axcolor)
        sregsig = Slider(axregsig, 'reg_sigma', 0.1, max(self.reg_sigma*4, 50), valinit=self.reg_sigma)
        sregsig.on_changed(set_reg_sigma)
        sregsig.my_scroll_delta = 0.5
        
        # add scroll
        def onscroll(event):
            sliders = [sslice, salpha, sregsig]
            axes = [s.ax for s in sliders]
            
            if event.inaxes not in axes:
                return
            
            s = sliders[axes.index(event.inaxes)]
            delta = s.my_scroll_delta
            
            val = s.val
            if event.button == 'up':
                new_val = val + delta
                if new_val <= s.valmax:
                    s.set_val(new_val)
                else:
                    s.set_val(s.valmax)
            if event.button == 'down':
                new_val = val - delta
                if new_val >= s.valmin:
                    s.set_val(new_val)
                else:
                    s.set_val(s.valmin)
        cid = fig.canvas.mpl_connect('scroll_event', onscroll)
        
        def onclose(event):
            self._fig_closed.set()
        cid_close = fig.canvas.mpl_connect('close_event', onclose)
        
        plt.show(block=False)
        self._fig_closed.clear()
        
        while True:
            update_plot(None)
            fig.canvas.draw_idle()
            fig.canvas.start_event_loop(0.2)
            if self._align_finished.is_set() or self._fig_closed.is_set():
                break
    
        
    def _condition_params(self):
        # make images 3d 
        if self._images_normed.ndim == 2:
            self._images_normed = self._images_normed[None]
        self._stack_len = len(self._images_normed)
        
        # reference
        ref_is_int = isinstance(self._reference, int)
        ref_is_str = isinstance(self._reference, str)
        ref_is_arr = isinstance(self._reference, np.ndarray)
        
        self._ref_im_from_stack = False
        if ref_is_int:
            self._ref_im = self._images_normed[self._reference]
            if len(self._images_normed) == 1:
                raise Exception("'ref_im' is identical to the stack of a single image.")
        elif ref_is_arr:
            self._ref_im = self._reference.copy()
        elif ref_is_str:
            self._ref_im_from_stack = True
            # check string mode is understood
            if self._reference.lower() not in self._ref_modes:
                raise Exception("'reference' (%s) must be in: " %(self._reference), self._ref_modes)
            self._ref_im = np.zeros(self._images_normed.shape[-2:])
        
        # check reg_mode
        if callable(self._reg_mode):
            # single callable
            self._reg_mode = (self._reg_mode, self._reg_mode)
        elif len(self._reg_mode) > 1:
            # string or func
            if callable(self._reg_mode[0]):
                # function, one for y and x
                assert len(self._reg_mode) == 2
            elif self._reg_mode.lower() not in self._reg_modes:
                # string
                raise Exception("'reg_mode' (%s) must be in: " %(self._reg_mode), self._reg_modes)
        
        # broadcast dyx0 if needed
        if self._dyx0 is not None:
            if self._dyx0.ndim == 3:
                self._dyx0 = np.repeat(self._dyx0[None], len(self._images_normed), axis=0)
        
        # check nrmse_mode
        if self._nrmse_mode.lower() not in self._nrmse_modes:
            raise Exception("'nrmse_mode' (%s) must be in: " %(self._nrmse_mode), self._nrmse_modes)
        
        # check threading
        nt = mp.cpu_count()
        if self._nthreads is not None and self._nthreads >= nt - 1:
            warnings.warn("'nthreads' (%d) >= threads - 1 (%d); performance may be reduced." %(self._nthreads, nt-1))
        
        # check pct
        if self._ishift:
            if self._pct is not None:
                warnings.warn("With 'ishift=True', 'pct=None' allows intensities to be transferred correctly")
                # this could be eased by updating the pct each time
                # and operating on non-normalised images
    
    def _prepare_images(self):
        # set range and calc grad
        if self._ref_im_from_stack:
            self._grad_ref_im = np.zeros((2,) + self._ref_im.shape)
        else:
            self._ref_im = self._im_norm(self._ref_im, self._pct)
            self._grad_ref_im = self._grad(self._ref_im, self._grad_sigma)
            
        self._images_normed = self._im_norm(self._images_normed, self._pct)
        # image grads are always calculated in main loop
        self._grad_images = np.zeros((len(self._images_normed), 2) + self._ref_im.shape)
        
        # create images
        self.images_aligned = self._images_normed.copy()
        
        # deformation field
        if self._dyx0 is None:
            self.dyx = np.zeros_like(self._grad_images)
        else:
            self.dyx = self._dyx0.copy()
        
        # coords
        self._cy, self._cx = np.indices(self._ref_im.shape)
        
        # history of errors
        self._nrmse_ax = []
        self._dnrmse_ax = []
        self._dyx_metric_ax = np.zeros((self._stack_len, 0))
        
        self._arrays_prepared.set()
    
    def _im_norm(self, im, pct, axis=(-2, -1)):
        if pct is None:
            return im
        
        # if not None, normalise
        if not isinstance(pct, Iterable):
            pct = (pct,)*2
        vmin, vmax = np.percentile(im, [pct[0], 100-pct[1]], axis=axis)
        if im.ndim==2:
            im = (im - vmin) / (vmax - vmin)
        elif im.ndim==3:
            im = (im - vmin[:, None, None]) / (vmax - vmin)[:, None, None]
        im = im.clip(0, 1)
        return im
    
    def _grad(self, im, grad_sigma=0):
        if grad_sigma==0:
            im_gyx = np.array(np.gradient(im, axis=(-2, -1)))
        else:
            im_gy = gaussian_filter1d(im, grad_sigma, order=1, axis=-2)
            im_gx = gaussian_filter1d(im, grad_sigma, order=1, axis=-1)
            im_gyx = np.array([im_gy, im_gx])
        
        # swap 1st 2 axes if 4d, so 1st axis is image #, 2nd is grads
        if im_gyx.ndim == 4:
            im_gyx = np.moveaxis(im_gyx, 1, 0)
        return im_gyx
    
    def warp(self, images, dyx=None, cval=np.nan, ishift=None):
        '''
        Warp images according to `dyx'.
        
        Parameters
        ----------
        images : ndarray
            The image(s) to be warped, of shape ([n,] y, x).
        dyx : None or ndarray
            The distortion matrix to be used, of shape ([n,] [fy, dx], y, x). If None,
            the internal distortion field is used.
        cval : float
            The value used to fill warped space.
        ishift : bool or None
            If True, intensity modification is approximated. See class docstring
            for details. If None, the class attribute is used.
        
        Returns
        -------
        wims : ndarray
            Warped images. Singular axes are removed.
        
        '''
        
        if ishift is None:
            ishift = self._ishift
        
        if dyx is None:
            dyx = self.dyx.copy()
        
        if dyx.ndim == 3:
            dyx = dyx.copy()[None]
        
        images = images.copy()
        if images.ndim == 2:
            images = images[None]
        
        wims = np.zeros_like(images)
        
        cy, cx = np.indices(images.shape[-2:])
        
        for j in range(len(dyx)):
            imj = images[j]
            dyxj = dyx[j]
            
            ncy = cy + dyxj[0]
            ncx = cx + dyxj[1]
            imj = map_coordinates(imj,
                                  np.vstack([ncy.flatten(), ncx.flatten()]),
                                  cval=cval)
            imj = imj.reshape(self._ref_im.shape)
            if ishift:
                imj = self._add_ishift(imj, dyxj)
            wims[j] = imj
        return np.squeeze(wims)
    
    def _add_ishift(self, image, dyx):
        dy = np.gradient(dyx[0], axis=0)
        dx = np.gradient(dyx[1], axis=1)
        d = dy + dx
        image = image * (1 + d)
        return image
        
    def _align(self):
        self._align_finished.clear()
        self._prepare_images()
        
        # nrmse variables
        self._rmss = np.zeros(len(self._images_normed))
        self.nrmse = np.inf
        self.dnrmse = np.inf
        rms_prev = np.inf
        
        stop = False
        self.i = 0
        while not stop:
            if self._interrupt_evnt.is_set():
                break
            
            # update ref_im and grad if it varies.
            if self._ref_im_from_stack:
                if self._reference == 'mean':
                    self._ref_im = np.mean(self.images_aligned, axis=0)
                elif self._reference == 'median':
                    self._ref_im = np.median(self.images_aligned, axis=0)
                self._grad_ref_im[:] = self._grad(self._ref_im, self._grad_sigma)
            
            # calc grad of images
            self._grad_images[:] = self._grad(self.images_aligned, self._grad_sigma)
            
            
            def process_image(j):
                # incremental displacement
                dyxj = -(self.images_aligned[j] - self._ref_im)[None] * (self._grad_images[j] + self._grad_ref_im)
                
                # regularise
                if self._dyxi_max is not None:
                    dyxj = dyxj.clip(-self._dyxi_max, self._dyxi_max)
                
                if callable(self._reg_mode[0]):
                    dyxj = np.array([fi(dyxji, **self._reg_kwargs) for fi, dyxji in zip(self._reg_mode, dyxj)])
                elif self._reg_mode == 'gaussian':
                    dyxj = gaussian_filter(dyxj, (0,) + (self.reg_sigma,)*2)
                
                # update deformation field
                with self._data_lock:
                    self.dyx[j] += self.alpha * dyxj
                
                # unwarp                
                ncy = self._cy + self.dyx[j, 0]
                ncx = self._cx + self.dyx[j, 1]
                imj = map_coordinates(self._images_normed[j],
                                      np.vstack([ncy.flatten(), ncx.flatten()]),
                                      cval=0) #np.nan)
                imj = imj.reshape(self._ref_im.shape)
                if self._ishift:
                    imj = self._add_ishift(imj, self.dyx[j])
                
                with self._data_lock:
                    self.images_aligned[j] = imj
                
                # calc error
                self._rmss[j] = nrmse(self._ref_im, imj, allow_nans=True)
            
            # loop over images, updating displacement fields
            if self._nthreads==1 or len(self.images_aligned)==1:
                for j in range(len(self.images_aligned)):
                    process_image(j)
            else:
                if self._nthreads is None:
                    nt = max(1, mp.cpu_count()-1)
                else:
                    nt = max(1, self._nthreads)
                with threadpool_limits(limits=1):
                    pool = mp.pool.ThreadPool(nt)
                    pool.map(process_image, range(len(self.images_aligned)))
                    pool.close()
            # update error
            if self._nrmse_mode == 'mean':
                self.nrmse = np.mean(self._rmss)
            elif self._nrmse_mode == 'median':
                self.nrmse = np.median(self._rmss)
            elif self._nrmse_mode == 'max':
                self.nrmse = np.max(self._rmss)
            elif self._nrmse_mode == 'quad':
                self.nrmse = (np.power(self._rmss, 2).sum() / self._rmss.size)**0.5
            
            # update stop criterion
            self.dnrmse = rms_prev - self.nrmse
            rms_prev = self.nrmse
            self.i += 1
            
            # dyx metric (could set size by niter and fill with nans, then filter elsewhere)
            dyx_metric = ((self.dyx**2).sum(1)**0.5).mean((-2, -1))[:, None]
            
            # build axis
            with self._data_lock:
                self._nrmse_ax.append(self.nrmse)
                self._dnrmse_ax.append(self.dnrmse)
                self._dyx_metric_ax = np.append(self._dyx_metric_ax, dyx_metric, axis=-1)
            
            self.criteria_bool = (self.i >= self._niter), (self.nrmse < self._nrmse_min), (np.abs(self.dnrmse) < self._nrmse_rel)
            # abs on drms to avoid instability
            if self._print_stats:
                if self.i == 1:
                    print('%5s, %9s, %9s, %s' %(tuple('i nrmse dnrmse criteria_bool'.split())))
                    print('------------------------------------------------------')
                print('%05d, %9.9f, %9.9f, (%r, %r, %r)' %((self.i, self.nrmse, self.dnrmse) + self.criteria_bool))
            stop = np.any(self.criteria_bool)
        # ended
        self._interrupt_evnt.clear()
        
        # align original images using non-normalised images
        # we don't use nans for warp, so add at end here (if requested)
        self.images_aligned = self.warp(self.images, self.dyx, cval=self.cval)
        
        self._align_finished.set()


