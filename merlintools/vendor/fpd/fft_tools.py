
import matplotlib.pylab as plt
import numpy as np
import scipy as sp

from scipy.ndimage import gaussian_filter, gaussian_filter1d
from matplotlib.widgets import Slider
from matplotlib.widgets import RadioButtons

plt.ion()

# favour sp.fft over np.fft
try:
    from scipy import fft as fft_module
except:
    from numpy import fft as fft_module



# TODO could specify s for faster fft in fft2 etc

def fft2rdf(rrf, im_fft, spf=1):
    '''
    Calculate radial distribution of fft image.
    
    Parameters
    ----------
    im_fft : ndarray
        2-D array of absolute FFT value.
    rrf : ndarray
        2-D array of FFT frequency magnitudes.
    spf : scalar
        Sub-pixel factor for returned values.
    
    Returns
    -------
    r : 1-D array
        Frequency array.
    radial_mean : 1-D array
        Average radial distribution.
    
    '''
    
    r_scale = 1.0/rrf.max() * np.sqrt(2)*im_fft.shape[0] / 2.0 * spf
    rrf = rrf * r_scale 
    rrf_int = rrf.round(0).astype(int)

    tbin = np.bincount(rrf_int.ravel(), im_fft.ravel())
    nr = np.bincount(rrf_int.ravel())
    with np.errstate(invalid='ignore', divide='ignore'):
        radial_mean = tbin / nr
    r = np.arange(radial_mean.shape[0]) / r_scale
    
    return r, radial_mean


def pad_image(image, pad_to=None):
    '''
    Pad an image to a shape.
    
    Parameters
    ----------
    image : ndarray
        2D array to be transformed.
    pad_to : None or string
        If not None, `image` is padded according to the string:
        - 'square' : the image is made square.
    
    Return
    ------
    image : ndarray
        Padded or original image.
    '''
    
    pad_to_modes = ['square']
    
    if pad_to is not None:
        pad_to = pad_to.lower()
        if pad_to not in pad_to_modes:
            raise Exception("'pad_to' (%s) must be one of:" %(pad_to), pad_to_modes)
        rows, cols = image.shape
        sq = max(rows, cols)
        image = np.pad(image, ((0, sq-rows), (0, sq-cols)))
    return image


def im2fftrdf(image, dy=1, dx=1, spf=1, pad_to='square'):
    '''
    Calculate radial distribution of the fft of an image.
    
    Parameters
    ----------
    image : ndarray
        2-D array of image intensities.
    dy, dx : scalar
        Pixel spacing along axes.
    spf : scalar
        Sub-pixel factor for returned values.
    pad_to : None or string
        If not None, `image` is padded according to the string:
        - 'square' : the image is made square.
    
    Returns
    -------
    r : 1-D array
        Frequency array.
    radial_mean : 1-D array
        Average radial distribution of frequency magnitudes.
    
    '''
    
    image = pad_image(image, pad_to=pad_to)
    rows, cols = image.shape
    
    yf = fft_module.fftfreq(rows, dy)
    xf = fft_module.fftfreq(cols, dx)
    
    yyf, xxf = np.meshgrid(yf, xf, indexing='ij')
    rrf = np.hypot(yyf, xxf)
    
    # scale allows us to use bincounting
    # set for ~1.0 pixel resolution
    r_scale = 1.0/rrf.max() * np.hypot(rows, cols) / 2.0 * spf
    rrf = rrf * r_scale 
    rrf_int = rrf.round(0).astype(int)
    
    im_fft = np.abs(fft_module.fft2(image))

    tbin = np.bincount(rrf_int.ravel(), im_fft.ravel())
    nr = np.bincount(rrf_int.ravel())
    with np.errstate(invalid='ignore', divide='ignore'):
        radial_mean = tbin / nr
    r = np.arange(radial_mean.shape[0]) / r_scale
    
    return r, radial_mean


def _gen_mask(rrf, fminmax, gy, mode='circ'):
    modes = ['circ', 'horiz', 'vert']
    if mode.lower() not in modes:
        raise NotImplementedError
    rrf = rrf.copy()
    imr, imc = rrf.shape
    fmin, fmax = fminmax
    gx = gy*float(imc)/imr
    g = (gy, gx)
    if mode == modes[0]:
        pass
    elif mode == modes[1]:
        xfreq = rrf[int(rrf.shape[0]/2), :]
        rrf = rrf*0 + xfreq[None, :]
    elif mode == modes[2]:
        yfreq = rrf[:, int(rrf.shape[1]/2)]
        rrf = rrf*0 + yfreq[:, None]
    mask = np.logical_and(rrf>=fmin, rrf<=fmax)
    mask = gaussian_filter(mask*1.0, g)
    return mask


def _gen_im_pass(imf, mask):
    ''' mask non-quad swapped FFT imf with mask and return real ifft'''
    im_pass = fft_module.ifft2(imf * fft_module.ifftshift(mask)).real
    return im_pass 


def bandpass(im, fminmax, gy, mask=None, full_out=False, mode='circ'):
    '''
    Basic bandpass filter, accepting multiple images at once.
    
    Parameters
    ----------
    im : ndarray
        Array of images to be filtered, with images in last 2 dimensions.
        The images need not be square.
    fminmax : length 2 iterable
        Minimum and maximum frequencies of band in 1/pixels.
    gy : scalar
        Gaussian sigma for edge smoothing in (fft) pixels.
        The x-axis value is calculated to match.
    mask : 2-D array or None
        If None, `fminmax` is used to generate mask.
        Else, the passed array with values 0-1 is used directly.  
    full_out : bool
        If True, additional outputs are returned.
    mode : str
        Type of bandpass in ['circ', 'horiz', 'vert']
        
    Returns
    -------
    Tuple of (im_pass, (im_fft, mask, extent)
    
    im_pass : ndarray
        Filtered images.
    im_fft : ndarray
        Absolute value of FFT.
    mask : ndarray
        Float mask image.
    extent : tuple
        FFT and mask image extent (left, right, top, bottom).
    
    If `full_out`, also returned are a tuple of (rrf, yf, xf, imf):
    
    rrf : ndarray
        2-D array of FFT frequency magnitudes.
    yf : ndarray
        1-D array of FFT y-axis frequencies.
    xf : ndarray
        1-D array of FFT x-axis frequencies.
    imf : ndarray
        Unswapped 2-D array of complex FFT.
        
    Examples
    --------
    Filter random noise image:
    
    >>> import numpy as np
    >>> from fpd.fft_tools import bandpass
    
    >>> im = np.random.rand(*(256,)*2)-0.5
    >>> im_pass, (im_fft, mask, extent) = bandpass(im, (1.0/12, 1.0/5), gy=2)
    
    >>> _ = plt.matshow(im)
    >>> _ = plt.matshow(im - 0.5*im_pass)
    >>> _ = plt.matshow(im - im_pass)   
    >>> _ = plt.matshow(im_fft, extent=extent)
    >>> _ = plt.matshow(mask, extent=extent)

    '''

    modes = ['circ', 'horiz', 'vert']
    if mode.lower() not in modes:
        raise NotImplementedError
    
    fmin, fmax = fminmax
    im_shape = im.shape
    imr, imc = im_shape[-2:]

    yf, xf = [fft_module.fftshift(fft_module.fftfreq(t)) for t in (imr, imc)]
    extent = (xf.min(), xf.max(), yf.max(), yf.min())

    xyf = np.meshgrid(xf, yf, indexing='xy')
    #xxf, yyf = xyf
    rrf = np.linalg.norm(xyf, ord=None, axis=0)
    
    imf = fft_module.fft2(im, s=None, axes=(-2, -1), norm=None)
    im_fft = np.abs(fft_module.fftshift(imf, axes=(-2, -1)))
    
    if mask is None:
        mask = _gen_mask(rrf, fminmax, gy, mode)
    else:
        mask = mask
    im_pass = _gen_im_pass(imf, mask)   
    
    out = (im_pass, (im_fft, mask, extent))
    if full_out:
        out += ((rrf, yf, xf, imf),)
    return out


class BandpassPlotter:
    def __init__(self, im, fminmax=(0, 0.5), gy=2, fact=0.75, cmap='gray', blit=True, mode='circ'):
        '''
        Interactive plotting of bandpass filter.
        See `bandpass` for documentation.
        
        Parameters
        ----------
        fact : scalar [0: 1]
            Factor of passed data subtracted from data (middle plot).
        cmap : matplotlib cmap
            Colourmap used for image plotting.
        blit : bool
            Controls blitting of plot updates.
        
        Examples
        --------
        >>> import matplotlib.pylab as plt
        >>> import numpy as np
        >>> import fpd
        >>> plt.ion()
        
        >>> py, px = 32.0, 16.0
        >>> y, x = np.indices((256,)*2)
        >>> im = np.sin(y/py*2*np.pi) + np.sin(x/px*2*np.pi)
        
        >>> b = fpd.fft_tools.BandpassPlotter(im, fminmax=(0.045, 0.08), gy=1, fact=1)
        
        '''     
        
        self.im = im
        self.fmin, self.fmax = fminmax
        self.gy = gy
        self.cmap = cmap
        self.blit = blit
        self.fact = fact
        self.mode = mode
        self.calc_data(first=True)
        self.init_fig()
        self.init_plots()
    
    
    def calc_avg_fft(self):
        # rdf
        if self.mode == 'circ':
            self.r, self.radial_mean = fft2rdf(self.rrf, self.im_fft)
        elif self.mode == 'horiz':
            i = (self.xf==0).argmax()
            self.r, self.radial_mean = self.xf[i:], self.im_fft.mean(0)[i:]
        elif self.mode == 'vert':
            i = (self.yf==0).argmax()
            self.r, self.radial_mean = self.yf[i:], self.im_fft.mean(1)[i:]
    
    
    def calc_data(self, first=False, level=0):
        out = bandpass(self.im, (self.fmin, self.fmax), gy=self.gy, full_out=first, mode=self.mode)
        if first:
            im_pass, (im_fft, mask, extent), (rrf, yf, xf, imf) = out
            self.extent = extent
            self.yf = yf
            self.xf = xf
            self.rrf = rrf
            self.imf = imf
            
            self.im_pass = im_pass
            self.im_fft = im_fft
            self.mask = mask
            
            self.calc_avg_fft()

        else:
            # update
            if level <= 1:
                ## mask changes
                self.mask = _gen_mask(self.rrf, (self.fmin, self.fmax), self.gy, self.mode)
                self.calc_avg_fft()
            if level <= 2:
                self.im_pass = _gen_im_pass(self.imf, self.mask)        
        if level <= 3:
            # factor changes
            self.im_dif = self.im - self.fact*self.im_pass
    
    def init_fig(self):
        nr, nc = self.im_pass.shape
        woh = float(nc)/nr
        
        self.fig = plt.figure(figsize=(9.25, 7.5))
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        plt.set_cmap(self.cmap)
        
        # axes
        gs = plt.mpl.gridspec.GridSpec(2, 3, 
                                       left=None, bottom=0.25, right=None, top=None,
                                       wspace=None, hspace=None, 
                                       width_ratios=None, height_ratios=None)
        self.ax_im = plt.subplot(gs[0, 0])
        self.ax_im_dif = plt.subplot(gs[0, 1], sharey=self.ax_im, sharex=self.ax_im)
        self.ax_im_dif.set_xlabel('pix')
        self.ax_im_pass = plt.subplot(gs[0, 2], sharey=self.ax_im, sharex=self.ax_im)
        
        self.ax_mask = plt.subplot(gs[1, 0])
        self.ax_fft = plt.subplot(gs[1, 1], sharey=self.ax_mask, sharex=self.ax_mask)
        self.ax_fft.set_xlabel('1/pix')
        self.ax_rdf = plt.subplot(gs[1, 2])
        
        
        self.ax_mask.minorticks_on()
        self.ax_fft.minorticks_on()
        
        
        # sliders
        axcolor = 'lightgoldenrodyellow'
        
        if int(plt.matplotlib.__version__.split('.') [0]) >= 2:
            d = {'facecolor': axcolor}
        else:
            d = {'axisbg': axcolor}
        
        axfmin = plt.axes([0.25, 0.02, 0.65, 0.025], **d)
        axfmax = plt.axes([0.25, 0.06, 0.65, 0.025], **d)
        axsig = plt.axes([0.25, 0.10, 0.65, 0.025], **d)
        axfact = plt.axes([0.25, 0.14, 0.65, 0.025], **d)
        
        v = '%1.3f'
        self.sfmin = Slider(axfmin, 'Fmin', 0.0, np.sqrt(2)*0.5,
                            valinit=self.fmin, valfmt=v)
        self.sfmax = Slider(axfmax, 'Fmax', 0.0, np.sqrt(2)*0.5,
                            valinit=self.fmax, valfmt=v)
        self.ssig = Slider(axsig, 'Sigma', 0.0, 10.0,
                           valinit=self.gy, valfmt=v)
        self.sfact = Slider(axfact, 'Factor', 0.0, 1.0,
                            valinit=self.fact, valfmt=v)
        
        self.sfmin.on_changed(self.update_mask)
        self.sfmax.on_changed(self.update_mask)
        self.ssig.on_changed(self.update_mask)
        self.sfact.on_changed(self.update_fact)
        
        # buttons
        norm_ax = plt.axes([0.02, 0.02, 0.07, 0.07], **d)
        self.rnorm = RadioButtons(norm_ax, ('Lin', 'Log'))
        self.rnorm.on_clicked(self.update_norm)
        
        mode_ax = plt.axes([0.02, 0.10, 0.07, 0.10], **d)
        labels = ['circ', 'horiz', 'vert']
        mode_ind = labels.index(self.mode)
        self.rmode = RadioButtons(mode_ax, labels)
        self.rmode.on_clicked(self.update_mode)
        self.rmode.eventson = False
        self.rmode.set_active(mode_ind)
        self.rmode.eventson = True
        
    def update_mode(self, label):
        self.mode = label
        # TODO: change overlays between lines / circles 
        self.update_mask(None)
        plt.draw()
    
    def update_norm(self, label):
        norm_dict = {'Lin': plt.mpl.colors.NoNorm, 'Log': plt.mpl.colors.LogNorm}
        norm = norm_dict[label]
        self.ax_fft.images[0].set_norm(norm())
        plt.draw()
    
    def on_scroll(self, event):
        ss = [self.sfmin, self.sfmax, self.ssig, self.sfact]
        axs = [s.ax for s in ss]
        try:
            ind = axs.index(event.inaxes)
        except ValueError:
            # not in any axis we want
            return
        # we are in an axis
        s = ss[ind]
        v = s.val
        vmin, vmax = s.valmin, s.valmax
        step = (vmax-vmin)/100.0

        if event.button == 'down':
            nv = v - step
        elif event.button == 'up':
            nv = v + step
        if nv < vmin:
            nv = vmin
        if nv > vmax:
            nv = vmax
        s.set_val(nv)
        
        if ind == 3:
            self.update_fact(None)
        else:
            self.update_mask(None)
        
    
    def init_plots(self):
        self.ax_im.imshow(self.im, interpolation='nearest')
        self.ax_im.set_title('Image')
        
        self.ax_fft.imshow(self.im_fft, extent=self.extent,
                           interpolation='nearest')
        self.fmin_circle = plt.Circle((0, 0), self.fmin, edgecolor='r',
                                      facecolor='none', lw=0.5)
        self.fmax_circle = plt.Circle((0, 0), self.fmax, edgecolor='r',
                                      facecolor='none', lw=0.5)
        self.ax_fft.add_artist(self.fmin_circle)
        self.ax_fft.add_artist(self.fmax_circle)
        
        numrows, numcols = self.im_fft.shape
        def format_coord(x, y):
            col = int(x + 0.5)
            row = int(y + 0.5)
            if col>=0 and col<numcols and row>=0 and row<numrows:
                z = self.im_fft[row,col]
                r = np.hypot(x, y)
                return 'x=%0.4f, y=%0.4f, r=%0.4f'%(x, y, r)
            else:
                return 'x=%1.4f, y=%1.4f' %(x, y)
        self.ax_fft.format_coord = format_coord

        
        self.ax_im_dif_im = self.ax_im_dif.imshow(self.im_dif, interpolation='nearest')
        self.ax_im_dif.set_title('Im - f*Pass')
        self.ax_im_pass_im = self.ax_im_pass.imshow(self.im_pass, interpolation='nearest')
        self.ax_im_pass.set_title('Pass')
        self.ax_mask_im = self.ax_mask.imshow(self.mask, extent=self.extent, 
                                              interpolation='nearest', 
                                              vmin=0, vmax=1)
        self.ax_mask.set_title('Mask')
        
        self.ax_rdf_line, = self.ax_rdf.semilogy(self.r, self.radial_mean, '-k')
        self.fmin_l = self.ax_rdf.axvline(self.fmin, color='r', lw=0.5)
        self.fmax_l = self.ax_rdf.axvline(self.fmax, color='r', lw=0.5)
        
        self.fig.canvas.draw_idle()
    
    
    def update_plots(self):
        if self.blit:
            # blitting
            axs = self.fig.axes[1:6]
            backgrounds = [self.fig.canvas.copy_from_bbox(ax.bbox) for ax in axs]
            
            for i, (ax, bk) in enumerate(zip(axs, backgrounds)):
                if i != 3:
                    # skip fft
                    self.fig.canvas.restore_region(bk)
        
        # set data
        self.ax_im_dif_im.set_data(self.im_dif)
        self.ax_im_dif_im.set_clim(self.im_dif.min(), self.im_dif.max())
        
        self.fmin_circle.set_radius(self.fmin)
        self.fmax_circle.set_radius(self.fmax)
        
        self.ax_im_pass_im.set_data(self.im_pass)
        self.ax_im_pass_im.set_clim(self.im_pass.min(), self.im_pass.max())
        
        self.ax_mask_im.set_data(self.mask)
        
        self.fmin_l.set_xdata([self.fmin,]*2)
        self.fmax_l.set_xdata([self.fmax,]*2)
        self.ax_rdf_line.remove()
        self.ax_rdf_line, = self.ax_rdf.semilogy(self.r, self.radial_mean, '-k')
        self.ax_rdf.relim()
        self.ax_rdf.autoscale_view(True,True,True)
        
        if not self.blit:
            self.fig.canvas.draw_idle()
        else:
            # blitting
            artists = [self.ax_im_dif_im, self.ax_im_pass_im, 
                    self.ax_mask_im, [self.fmin_circle, self.fmax_circle],
                    [self.fmin_l, self.fmax_l]]
            for ax, a in zip(axs, artists):
                if isinstance(a, list):
                    [ax.draw_artist(ai) for ai in a]
                else:
                    ax.draw_artist(a)
                self.fig.canvas.blit(ax.bbox)
            
    
    def update_mask(self, val):
        self.fmin = self.sfmin.val
        self.fmax = self.sfmax.val
        self.gy = self.ssig.val
        self.calc_data(0)
        self.update_plots()
        
        
    def update_fact(self, val):
        self.fact = self.sfact.val
        self.calc_data(3)
        self.update_plots()


def fft2_diff(im, ypix=None, xpix=None, order=1):
    '''
    Fourier transform based numerical differentiation and integration of images.
    
    Parameters
    ----------
    im : ndarray
        2-D image.
    ypix, xpix : scalar or None
        Pixel spacing along axis. If None, the pixel spacing is taken as 1.
    order : int
        Order of the differentiation. A value of 1 is first order. If negative,
        integration is performed. For integration, the constant is taken as zero.
    
    Returns
    -------
    im_grad : ndarray
        3-D array of image gradients, with first axis [y, x].
    
    '''
    
    # condition inputs
    if ypix is None:
        ypix = 1
    if xpix is None:
        xpix = 1
       
    ky, kx = [fft_module.fftfreq(si, pi) for (si, pi) in zip(im.shape, (ypix, xpix))]
    if order < 0:
        ky[0] = 1
        kx[0] = 1
    kyy, kxx = np.array(np.meshgrid(ky, kx, indexing='ij')) * 2*np.pi

    im_ft = fft_module.fft2(im) 

    # fft derivative
    xmul = im_ft * (kxx * 1j)**order
    ymul = im_ft * (kyy * 1j)**order
    if order < 0:
        xmul[0, 0] = 0
        ymul[0, 0] = 0
    im_gradx = fft_module.ifft2(xmul).real
    im_grady = fft_module.ifft2(ymul).real
    
    im_grad = np.array([im_grady, im_gradx])
    return im_grad


def fft2_laplacian(im, ypix=None, xpix=None):
    '''
    Fourier transform based numerical laplacian of images.
    
    Parameters
    ----------
    im : ndarray
        2-D image.
    ypix, xpix : scalar or None
        Pixel spacing along axes. If None, the pixel spacing is taken as 1.
    
    Returns
    -------
    lap : ndarray
        2-D array of image laplacian.
    
    '''
    
    # condition inputs
    if ypix is None:
        ypix = 1
    if xpix is None:
        xpix = 1
    
    ky, kx = [fft_module.fftfreq(si, pi) for (si, pi) in zip(im.shape, (ypix, xpix))]
    kyy, kxx = np.array(np.meshgrid(ky, kx, indexing='ij')) * 2*np.pi

    im_ft = fft_module.fft2(im) 

    # fft laplacian
    delsq = -(kyy**2 + kxx**2)
    
    lap_ft = im_ft * delsq
    lap_ft[0, 0] = 0
    lap = fft_module.ifft2(lap_ft).real
    return lap


def fft2_ilaplacian(im, ypix=None, xpix=None):
    '''
    Fourier transform based numerical inverse laplacian of images.
    
    Parameters
    ----------
    im : ndarray
        2-D image.
    ypix, xpix : scalar or None
        Pixel spacing along axes. If None, the pixel spacing is taken as 1.
    
    Returns
    -------
    ilap : ndarray
        2-D array of image inverse laplacian.
    
    '''
    
    # condition inputs
    if ypix is None:
        ypix = 1
    if xpix is None:
        xpix = 1
    
    ky, kx = [fft_module.fftfreq(si, pi) for (si, pi) in zip(im.shape, (ypix, xpix))]
    kyy, kxx = np.array(np.meshgrid(ky, kx, indexing='ij')) * 2*np.pi

    im_ft = fft_module.fft2(im) 

    # fft inv laplacian
    delsq = -(kyy**2 + kxx**2)
    delsq[0, 0] = 1
    
    ilap_ft = im_ft / delsq
    ilap_ft[0, 0] = 0
    ilap = fft_module.ifft2(ilap_ft).real
    return ilap


def fft2_igrad(grady, gradx, ypix=None, xpix=None):
    '''
    Fourier transform based numerical inverse gradient of images.
    The offset is taken as zero.
    
    Parameters
    ----------
    grady, gradx : ndarray
        2-D image gradient along y- and x-axes.
    ypix, xpix : scalar or None
        Pixel spacing along axes. If None, the pixel spacing is taken as 1.
    
    Returns
    -------
    im : ndarray
        2-D array of the image.
    
    '''
    
    # condition inputs
    if ypix is None:
        ypix = 1
    if xpix is None:
        xpix = 1    

    ky, kx = [fft_module.fftfreq(si, pi) for (si, pi) in zip(grady.shape, (ypix, xpix))]

    # through laplacian
    kyy, kxx = np.array(np.meshgrid(ky, kx, indexing='ij')) * 2*np.pi
    delsq = -(kyy**2 + kxx**2)
    delsq[0, 0] = 1

    im_ft = (fft_module.fft2(grady) * (kyy * 1j) + fft_module.fft2(gradx) * (kxx * 1j)) / delsq
    im_ft[0, 0] = 0
    im = fft_module.ifft2(im_ft).real
    
    return im


def fft_diff(y, xpix=None, order=1):
    '''
    Fourier transform based numerical differentiation and integration of 1-D profiles.
    
    Parameters
    ----------
    y : ndarray
        1-D array.
    xpix : scalar or None
        Pixel spacing along axis. If None, the pixel spacing is taken as 1.
    order : int
        Order of the differentiation. A value of 1 is first order. If negative,
        integration is performed. For integration, the constant is taken as zero.
    
    Returns
    -------
    grad : ndarray
        1-D array of gradient..
    
    '''
    
    # TODO:
    # could allow ndarray input and specify axis (but this is already in fft2_diff, so...)
    
    # condition inputs
    if xpix is None:
        xpix = 1
       
    kx = fft_module.fftfreq(len(y), xpix) * 2*np.pi
    if order < 0:
        kx[0] = 1

    y_ft = fft_module.fft(y) 

    # fft derivative
    xmul = y_ft * (kx * 1j)**order
    if order < 0:
        xmul[0] = 0
    grad = fft_module.ifft(xmul).real
    
    return grad


def cepstrum2(image, mode='real', sigma=1, pad_to=None, plot=False):
    '''
    2D cepstrum transform.
    
    Parameters
    ----------
    image : ndarray
        2D array to be transformed.
    mode : string
        Controls the cepstrum type:
        'real' : real cepstrum
        'power' : power cepstrum
    sigma : scalar or None
        If not None, the post-transform 2-D Gaussian smoothing width.
    pad_to : None or string
        If not None, `image` is padded according to the string:
        - 'square' : the image is made square.
    plot : bool
        If True, the results are plotted on a log scale and
        with an custom format display.
    
    Return
    ------
    cp : ndarray
        Cepstrum transform.
    
    Examples
    --------
    import numpy as np
    import matplotlib.pylab as plt
    plt.ion()

    from fpd.tem_tools import synthetic_lattice
    from fpd.synthetic_data import array_image, disk_image
    from fpd.fft_tools import cepstrum2
    
    # generate lattice points
    im_shape = (256,)*2
    cyx = (np.array(im_shape) -1) / 2
    d0 = disk_image(intensity=100, radius=4, size=im_shape[0], sigma=1)
    yxg = synthetic_lattice(cyx=cyx, ab=(42, 62), angles=(0, np.pi/2), shape=im_shape, plot=False)
    yxg -= cyx

    # keep ~half of the points
    b = np.random.choice(np.indices(yxg.shape[:1])[0], yxg.shape[0]//2, replace=False)
    yxg = yxg[b]
    yxg += np.random.normal(scale=0.5, size=yxg.shape)

    # make the image
    im = array_image(d0, yxg)

    # scale and add background
    yi, xi = np.indices(im_shape)
    ri = np.hypot(xi - xi.mean(), yi - yi.mean())
    s = -(ri - ri.max())
    s /= s.max()
    im = im * s + s*20

    # plot image
    plt.matshow(im)

    # calculate cepstrum and plot
    cp = cepstrum2(im, plot=True)
    
    '''
    
    image = pad_image(image, pad_to=pad_to)
    
    mode = mode.lower()
    modes = ['real', 'power']
    if mode not in modes:
        raise Exception("'mode' (%s) must be in: " %(mode), modes)
    
    imagef = fft_module.fft2(image)
    imagefc = np.log(np.abs(imagef)**2)
    cp = np.abs(fft_module.ifft2(imagefc))**2
    cp = fft_module.fftshift(cp)
    
    if mode == 'real':
        cp = cp**0.5 / 2
    
    if sigma is not None and sigma != 0:
        cp = gaussian_filter(cp, sigma)
    
    if plot:
        plt.matshow(cp, norm=plt.mpl.colors.LogNorm())
        
        # display angle / size
        ax = plt.gca()
        cyx = np.array(image.shape) / 2
        
        def format_coord(x, y):
            x = x - cyx[1]
            y = y - cyx[0]
            r = np.hypot(x, y)
            t = np.arctan2(y, x)
            td = np.rad2deg(t)
            return 'x=%1.1f, y=%1.1f, r=%1.1f, t=%1.1f' % (x, y, r, td)
        ax.format_coord = format_coord
        
    return cp

def cepstrum(a, mode='real', sigma=1, plot=False):
    '''
    1D cepstrum transform.
    
    Parameters
    ----------
    image : ndarray
        1D array to be transformed.
    mode : string
        Controls the cepstrum type:
        'real' : real cepstrum
        'power' : power cepstrum
    sigma : scalar or None
        If not None, post-transform 2-D Gaussian smoothing width.
    plot : bool
        If True, the results are plotted on a log scale.
    
    Return
    ------
    cp : ndarray
        Cepstrum transform.
    
    '''
    
    mode = mode.lower()
    modes = ['real', 'power']
    if mode not in modes:
        raise Exception("'mode' (%s) must be in: " %(mode), modes)
    
    af = fft_module.fft(a)
    afc = np.log(np.abs(af)**2)
    cp = np.abs(fft_module.ifft(afc))**2
    
    # select +ve querfrency
    n = len(a) // 2
    if len(a) % 2:
        # odd
        n = (len(a)-1)//2 +1
    cp = cp[:n]
    
    if mode == 'real':
        cp = cp**0.5 / 2
    
    if sigma is not None and sigma != 0:
        cp = gaussian_filter1d(cp, sigma)
    
    if plot:
        plt.figure()
        plt.semilogy(cp)
    return cp


def fftcorrelate(im1, im2, phase_exponent=0):
    '''
    Correlate two images using FFTs.

    Parameters
    ----------
    im1 : 2-D array
        First real image.
    im2 : 2-D array
        Second real image. This can be smaller than `im1`, but should be
        have the relevant feature centred within its limits if the output
        is to have peaks at the locations of the equivalent features in `im`.
        Otherwise, the offset between features is relative to the centre of
        the image.
    phase_exponent : scalar in [0, 1]
        Exponent used in the normalisation of the correlation.
        0 : cross-correlation
        1 : phase-correlation

    Returns
    -------
    c : 2D array
        The result of correlation of `im1` and `im2`. The array is the
        same size as `im1` and centered with respect to the equivalent full
        output, matching the behaviour of scipy.signal.fftconvolve with 
        `mode='same'`.
    
    '''
    
    im2 = im2[::-1, ::-1]
    
    s1 = im1.shape
    s2 = im2.shape
    shape = [s1[i] + s2[i] - 1 for i in range(im1.ndim)]
    
    # ffts
    fshape = [fft_module.next_fast_len(shapei, True) for shapei in shape]
    f1 = fft_module.fft2(im1, fshape)
    f2 = fft_module.fft2(im2, fshape)
    c = f1 * f2
    if phase_exponent != 0:
        c /= np.abs(c) ** phase_exponent + np.finfo(np.float32).eps
    c = np.abs(fft_module.ifft2(c, fshape))
    
    # crops
    fslice = tuple([slice(sz) for sz in shape])
    c = c[fslice]
    c = sp.signal.signaltools._centered(c, s1).copy()
    return c

