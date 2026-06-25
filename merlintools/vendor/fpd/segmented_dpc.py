
import numpy as np
import os
import warnings

from fpd.fpd_processing import rebinA, rotate_vector
from fpd.utils import nearest_int_factor


def load_gla_dpc(ds, path=None, ext='dm3'):
    '''
    Read a Glasgow DPC dataset from file and return the data, with
    the data corrected for offsets and for some small errors.
    
    Parameters
    ----------
    ds : str
        DPC dataset ID, e.g. '001_default' to be loaded from files in 'path'.
    path : string
        Directory containing data. CWD if None.
    ext : string
        DM file extension.
    
    Returns
    -------
    Tuple of: a, ds_dict, opt_dict
    a : ndarray
        3-D array of detector signals, with the images in the last axes,
        and the signals in first axis. The signals are in order of:
        inner 0->3, outer 0->3.
    ds_dict : dict
        Dictionary of hyperspy images of the segmented detector data.
    opt_dict
        Dictionary of possible additional signals in hyperspy format. If
        not acquired, the value for that key is None.
    
    See Also
    --------
    fpd.segmented_dpc.SegmentedDPC
    
    '''
    
    try:
        from hyperspy.io import load
    except ModuleNotFoundError as e:
        print("Install HyperSpy to load Digital Micrograph files.")
        raise e
    
    if path is None:
        path = os.path.curdir
    if ext.startswith('.') == False:
        ext = '.' + ext
    
    # file names to try reading
    fns = ['KE INT ' + str(t) for t in range(4)] + ['KE EXT ' + str(t) for t in range(4)]
    fn_optional = ['Gatan HAADF',
                   'JEOL NCB',
                   'Noise image']
    
    ds_dict = {}
    opt_dict = {}
    for fni in (fns + fn_optional):
        fpath = os.path.join(path, ds + '_' + fni + ext)
        try:
            t = load(fpath)
            t.change_dtype(float)
            if fni in fns:
                # dpc data
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    offset = t.original_metadata.ImageList.TagGroup0.ImageData.Calibrations.Brightness.Origin
                    if offset == 0.0:
                        warnings.warn("%s brightness origin seems to be zero. Were the trims set?" %(im_name),
                                    UserWarning)
                t.data -= offset
                ds_dict[fni] = t
            else:
                # optional data
                opt_dict[fni] = t
        except ValueError as exc:
            if fni in fn_optional:
                print("Skipping optional '%s' image." %(fni))
                opt_dict[fni] = None
            else:
                raise exc
            
    # workaround for scale info not in both noise axes
    if opt_dict['Noise image'] is not None:
        opt_dict['Noise image'].axes_manager[0].scale = ds_dict[fns[0]].axes_manager[0].scale
        opt_dict['Noise image'].axes_manager[1].scale = ds_dict[fns[0]].axes_manager[1].scale
    
    a = np.concatenate([ds_dict[fni].data[None] for fni in fns], axis=0)
    return a, ds_dict, opt_dict


class SegmentedDPC:    
    def __init__(self, ds, alpha=1.0, scan_deg=0.0, fudge=None, method='accurate', 
                 sum_method='pixel', rebin=None):
        '''
        Differential Phase Contrast (DPC) class for processing 
        4- and 8-segment DPC data.
        
        Parameters
        ----------
        ds : ndarray
            3-D array of detector signals, with the images in the last axes,
            and the signals in first axis. The signals are in order of:
            inner 0->3, outer 0->3. If the first axis length is 4, only the
            inner quadrants are used for the standard DPC analysis. Modified
            DPC analysis is also performed if there are 8 detector signals.
            See Notes.
        alpha : scalar
            Semi-convergence angle. The calculated angles will be in the 
            same units.
        scan_deg : scalar
            Scanning angle in degrees. If not zero, vectors are rotated to
            correct for the offset. Positive values rotates data clockwise.
        method : string
            DPC calculation method.
            If 'accurate', a numerical solution used.
            If 'fast', an analytical solution is used employing 
            low angle approximations.
        sum_method : string
            Method of generating sum image for DPC calculations.
            If 'pixel' (default), image is generated pixel-by-pixel.
            If 'mean', all pixels share the mean value.
            If 'percentile', all pixels share the 50 percentile value.
            if 'plane', pixel values are from a plane fitted to 'pixel' data.
        rebin : int or None
            If not 0, data is rebinned by this factor along each axis. If not
            valid, a suitable alternative will be used instead.
        fudge : str or callable
            Input data modification before calculating the DPC signal.
            If 'i0', i0 data is replaced with average of other inner datasets.
            If a callable, the signature must be function(self). The attributes
            are i0 - i3 and o1 - o3 for the inner and outer segments,
            respectively. See examples.
            
        Returns
        -------
        SegmentedDPC : class
            DPC class with input and processed dpc data as attributes in numpy
            arrays. The main DPC signals are listed below.
        
        Attributes
        ----------
        mdpc_betay : ndarray
            The y-axis deflection angle in units of `alpha` using the 'modified'
            method where the difference signal comes from the outer detectors.
        mdpc_betax : ndarray
            The x-axis deflection angle in units of `alpha` using the 'modified'
            method where the difference signal comes from the outer detectors.
        dpc_betay : ndarray
            The y-axis deflection angle in units of `alpha` using the standard
            method where the difference signal comes from all detectors.
        dpc_betax : ndarray
            The x-axis deflection angle in units of `alpha` using the standard
            method where the difference signal comes from all detectors.
        
        Notes
        -----
        Detector layout:
            0   1
            
            3   2
        
        Convention:
            X: +ve 0->1
            
            Y: +ve 0->3
        
        Examples
        --------
        >>> import numpy as np
        >>> from fpd.synthetic_data import shift_array, disk_image, shift_images
        >>> from fpd.synthetic_data import segmented_detectors, segmented_dpc_signals
        >>> from fpd import SegmentedDPC
        
        Prepare data:
        
        >>> radius = 32
        >>> sa = shift_array(scan_len=9, shift_min=-2.0, shift_max=2.0)
        >>> sa = np.asarray(sa)
        >>> im = disk_image(intensity=1e3, radius=radius, size=256, dtype=float)
        >>> data = shift_images(sa, im, noise=False)
        
        Detector geometry and signals:
        
        >>> detectors = segmented_detectors(im_shape=(256, 256), rio=(24, 128), ac_det_roll=2)
        >>> det_sigs = segmented_dpc_signals(data, detectors)
        
        DPC analysis:
        
        >>> d = SegmentedDPC(det_sigs, alpha=radius)
        
        Replace faulty channel:
        
        >>> def func(self):
        >>>     self.i0 = (self.i1 + self.i2 + self.i3) / 3.0
        >>> d = SegmentedDPC(det_sigs, alpha=radius, fudge=func)
        
        Load a Glasgow dataset
        
        >>> a = load_gla_dpc('001_default')
        >>> d = SegmentedDPC(a, alpha=radius)
        
        See Also
        --------
        fpd.segmented_dpc.load_gla_dpc, fpd.DPC_Explorer, fpd.ransac_tools.ransac_im_fit
        
        '''
        
        #TODO add opposing single quadrant analysis?
        
        
        # condition inputs 
        methods = ['accurate', 'fast']
        if method.lower() not in methods:
            msg = "Method '%s' not known; known methods: " %(method) + str(methods)
            raise ValueError(msg)
        self.method = method
        
        self.sum_methods = ['pixel', 'mean', 'percentile', 'plane']
        if sum_method.lower() not in sum_method:
            msg = "'sum_method' '%s' not known; known methods: " %sum_method + str(self.sum_methods)
            raise ValueError(msg)
        self.sum_method = sum_method.lower() 
        
        self.scan_deg = scan_deg
        self.alpha = alpha
        
        # Calculate lookup table for analytical solution to dpc value 
        # as function of beta/alpha (bar).
        bar = np.linspace(0.0, 1.0, 1000)
        self.dpc = 1.0-4.0/np.pi*(np.arctan(np.sqrt((1.0-bar)/(1.0+bar)))
                                  -bar*np.sqrt((1.0-bar)*(1.0+bar))/2.0)
        self.bar = bar
        
        # load dpc images
        self._oct_seg = True
        if not isinstance(ds, np.ndarray):
            raise ValueError("`ds` must be an ndarray.")
        if len(ds) == 4:
            self._oct_seg = False
        self._load_numpy(ds, rebin)
        
        # fudgify data
        if fudge is None:
            pass
        elif callable(fudge):
            rtn = fudge(self)
        elif fudge is 'i0':
            # replace i0 with average of other inners
            self.i0 = (self.i1 + self.i2 + self.i3) / 3.0
        else:
            raise ValueError("Fudge value '%s' not understood!" %(fudge))
        
        # calculate images
        self._calc_mdpc_ims()
        if self.scan_deg is not 0.0:
            self._correct_for_scan_angle()
    
    def _rebin_input(self, a, rebin):
        if rebin:
            n, h, w = a.shape
            fy, fsy = nearest_int_factor(h, rebin)
            fx, fsx = nearest_int_factor(w, rebin)
            
            rebina = np.array([fy, fx])
            if (rebina != rebin).any():
                print('Requested rebin (%d) changed to nearest value: (%d, %d).' %(rebin, fy, fx))
                print('Possible values are:', (fsy, fsx))
        
            ns = (n, h // rebina[0], w // rebina[1])
            a = rebinA(a, *ns)
        return a
        
    def _load_numpy(self, a, rebin): 
        a = np.array(a, dtype=float, copy=False)
        ims = ['i0',
               'i1',
               'i2',
               'i3',
               'o0',
               'o1',
               'o2',
               'o3']
        if not self._oct_seg:
            ims = ims[:4]
        
        a = self._rebin_input(a, rebin)
        
        for im_name, ai in zip(ims, a):
            setattr(self, im_name, ai)
    
    def _beta_from_dpc_sig(self, dpc_sig):
        if self.method == 'accurate':
            # interpolate lookup-table and return beta
            return np.sign(dpc_sig)*np.interp(np.abs(dpc_sig), self.dpc, self.bar)*self.alpha
        elif self.method == 'fast':
            return np.pi /4.0 * dpc_sig * self.alpha
    
    def _dpc_beta(self, mode):
        '''
        mode : str
            'std':  std dpc
            'mod':  modified dpc
            'both':  both
        '''
        
        if mode in ['std', 'both']:
            q0 = self.i0
            q1 = self.i1
            q2 = self.i2
            q3 = self.i3
            if self._oct_seg:
                q0 = q0 + self.o0
                q1 = q1 + self.o1
                q2 = q2 + self.o2
                q3 = q3 + self.o3
        
            dpcx = ((q1+q2)-(q0+q3)) / self.dpc_sum
            dpcy = ((q2+q3)-(q0+q1)) / self.dpc_sum
            
            self.dpc_betax = self._beta_from_dpc_sig(dpcx)
            self.dpc_betay = self._beta_from_dpc_sig(dpcy)
        
        if mode in ['mod', 'both']:
            # mDPC
            mdpcx = ((self.o1 + self.o2) - (self.o0 + self.o3)) / self.dpc_sum
            mdpcy = ((self.o2 + self.o3) - (self.o0 + self.o1)) / self.dpc_sum
            
            self.mdpc_betax = self._beta_from_dpc_sig(mdpcx)
            self.mdpc_betay = self._beta_from_dpc_sig(mdpcy)
    
    def _calc_mdpc_ims(self):
        # Calculate DPC and mDPC images.
        
        # (annular) BF
        self.BFi = self.i0 +self.i1 + self.i2 + self.i3
        self.BFt = self.BFi
        if self._oct_seg:
            self.BFo = self.o0 + self.o1 + self.o2 + self.o3
            self.BFt += self.BFo
        
        # DPC
        if self.sum_method == self.sum_methods[0]:
            self.dpc_sum = self.BFt
        elif self.sum_method == self.sum_methods[1]:
            self.dpc_sum = self.BFt
            self.dpc_sum[:] = self.BFt.mean()
        elif self.sum_method == self.sum_methods[2]:          
            self.dpc_sum = self.BFt
            self.dpc_sum[:] = np.percentile(self.BFt, 50)
        elif self.sum_method == self.sum_methods[3]:
            # based on
            # https://stackoverflow.com/questions/18552011/3d-curvefitting/18648210#18648210
            # https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6
            im = self.BFt

            # regular grid covering the domain of the data
            Y, X = np.indices(im.shape)
            XX = X.flatten()
            YY = Y.flatten()
            ZZ = im.flatten()
            data = np.array(list(zip(XX, YY, ZZ)))

            # best-fit linear plane
            A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
            from scipy.linalg import lstsq
            C, _, _, _ = lstsq(A, data[:, 2])

            # evaluate it on grid
            Z = C[0]*X + C[1]*Y + C[2]
            self.dpc_sum = Z
        
        if self._oct_seg:
            self._dpc_beta(mode='both')
        else:
            self._dpc_beta(mode='std')
    
    def _correct_for_scan_angle(self):
        # scan_deg < 0 means vector has to be rotated anticlockwise
        self.dpc_betay, self.dpc_betax = rotate_vector([self.dpc_betay, self.dpc_betax], self.scan_deg)
        if self._oct_seg:
            self.mdpc_betay, self.mdpc_betax = rotate_vector([self.mdpc_betay, self.mdpc_betax], self.scan_deg)



