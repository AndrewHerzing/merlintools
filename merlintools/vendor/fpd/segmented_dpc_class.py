
import numpy as np
import os
import warnings

class SegmentedDPC:
    def __init__(self, ds, path=None, ext='dm3', alpha=1.0, 
                 scan_deg=0.0, fudge=None, method='accurate', 
                 sum_method='pixel', rebin=0):
        '''
        DEPRECATED in v0.2.4. Use fpd.segmented_dpc.SegmentedDPC instead.
        
        Differential Phase Contrast (DPC) class for processing 
        4- and 8-segment DPC data.
        
        Parameters
        ----------
        ds : string or ndarray
            If a string, DPC dataset with id, e.g. '001_default' is loaded
            from files in 'path'.
            If an ndarray, the first dimension of the ndarray is taken as the 
            detector signals in order of inner 0->3, outer 0->3. If the first
            axis length is 4, only the inner quadrants are used.
        path : string
            Directory containing data. CWD if None.
        ext : string
            DM file extension.
        alpha : scalar
            Semi-convergence angle. The calculated angles will be in the 
            same units.
        scan_deg : scalar
            Scanning angle in degrees.
            If not zero, vectors are rotated to correct for offset.
            A scan_deg>0 makes image turn clockwise.
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
        rebin : integer
            If not 0, data is rebinned by factor of 2**rebin. 0 for none, 
            1 for 2 etc. This is useful for speeding up calculations.
        fudge : string or callable
            Control modification of data before calculating DPC result.
            If 'i0', i0 data is replaced with average of other inner datasets.
            If a callable, the signature must be function(self).
            
        Returns
        -------
        DPC class with input and processed dpc data.
        
        Notes
        -----
        Detector layout:
            0 1
            3 2
        
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
        
        '''
        
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            msg = 'This function is deprecated and marked for removal.Use the similar (but better) `fpd.segmented_dpc.SegmentedDPC class instead.'
            warnings.warn(msg, DeprecationWarning)
        
        
        # condition inputs        
        methods = ['accurate', 'fast']
        if method.lower() not in methods:
            print("Method '%s' not known; known methods: ", methods)
            raise Exception
        self.method = method
        
        self.sum_methods = ['pixel', 'mean', 'percentile', 'plane']
        if sum_method.lower() not in sum_method:
            print("'sum_method' '%s' not known; known methods: ", self.sum_methods)
            raise Exception
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
        if isinstance(ds, str):
            self._load_ims(ds, path, ext, rebin)
        elif isinstance(ds, np.ndarray):
            if len(ds) == 4:
                self._oct_seg = False
            self._load_numpy(ds, rebin)
        
        # fudgify data
        if fudge is None:
            pass
        elif callable(fudge):
            # fudge is function, so call it
            #print('mmm, fudgy function')
            rtn = fudge(self)
        elif fudge is 'i0':
            # replace i0 with average of other inners
            self.i0.data[:] = ((self.i1+self.i2+self.i3)/3.0).data
        else:
            raise ValueError("Fudge value '%s' not understood!" %(fudge))
        
        # calculate images
        self._calc_mdpc_ims()
        if scan_deg is not 0.0:
            self._correct_for_scan_angle(scan_deg)
    
    def _load_ims(self, ds, path, ext, rebin):
        try:
            from hyperspy.io import load
        except ModuleNotFoundError as e:
            print("Install HyperSpy to load Digital Micrograph files.")
            raise e
        
        if path is None:
            path = os.path.curdir
        if ext.startswith('.') == False:
            ext = '.'+ext
        
        fns = ['KE INT 0',
               'KE INT 1',
               'KE INT 2',
               'KE INT 3',
               'KE EXT 0',
               'KE EXT 1',
               'KE EXT 2',
               'KE EXT 3']
        
        ims = ['i0',
               'i1',
               'i2',
               'i3',
               'o0',
               'o1',
               'o2',
               'o3']
        
        fn_optional = ['Gatan HAADF',
                       'JEOL NCB',
                       'Noise image']
        
        ims_optional = ['haadf',
                        'ncb',
                        'noise']
        
        for fid, im_name in zip(fns+fn_optional, ims+ims_optional):
            fpath = os.path.join(path, ds + '_' + fid + ext)
            try:
                t = load(fpath)
                t.change_dtype(float)
                if im_name in ims:
                    with warnings.catch_warnings():
                        warnings.simplefilter("always")
                        offset = t.original_metadata.ImageList.TagGroup0.ImageData.Calibrations.Brightness.Origin
                        if offset == 0.0:
                            warnings.warn("%s brightness origin seems to be zero. Were the trims set?" %(im_name),
                                        UserWarning)
                    t.data -= offset
                setattr(self, im_name, t)
            except ValueError as exc:
                if im_name in ims_optional:
                    print("Skipping optional '%s' image." %(im_name))
                    setattr(self, im_name, None)
                else:
                    raise exc
            
        # workaround for scale info not in both noise axes
        if self.noise is not None:
            self.noise.axes_manager[0].scale = self.i0.axes_manager[0].scale
            self.noise.axes_manager[1].scale = self.i0.axes_manager[1].scale
        
        if rebin:
            rbf = 2**rebin
            sy, sx = self.i0.data.shape
            syn = sy/rbf
            sxn = sx/rbf
            
            for im_name in ims:
                exec('self.'+im_name+'=self.'+im_name+'.rebin((%d,%d))' %(sxn, syn))
    
    def _load_numpy(self, a, rebin): 
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
        
        for i, im_name in enumerate(ims):
            # try / except for hyperspy class change since 1.0  (needed for python 2 / 3)             
            try:
                try:
                    from hyperspy.signals import Image as Signal2D
                except ImportError:
                    from hyperspy.signals import Signal2D
            except ModuleNotFoundError as e:
                print("Use fpd.segmented_dpc.SegmentedDPC instead.")
                raise e
            
            t = Signal2D(a[i])
            t.change_dtype(float)
            setattr(self, im_name, t)

        if rebin:
            rbf = 2**rebin
            sy, sx = self.i0.data.shape
            syn = sy/rbf
            sxn = sx/rbf
            
            for im_name in ims:
                exec('self.'+im_name+'=self.'+im_name+'.rebin((%d,%d))' %(sxn, syn))
    
    def _beta_from_dpc_sig(self, dpc_sig):
        if self.method == 'accurate':
            # interpolate lookup-table and return beta
            return np.sign(dpc_sig)*np.interp(np.abs(dpc_sig), self.dpc, self.bar)*self.alpha
        elif self.method == 'fast':
            return np.pi/4.0*dpc_sig *self.alpha
    
    def _dpc_beta(self, mode):
        '''
        mode : string
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
        
            self.dpc_betax = dpcx.deepcopy()
            self.dpc_betay = dpcy.deepcopy()
            self.dpc_betax.data[:] = self._beta_from_dpc_sig(dpcx.data)
            self.dpc_betay.data[:] = self._beta_from_dpc_sig(dpcy.data)
            self.dpc_betax.metadata.General.title = "DPC Beta x"
            self.dpc_betay.metadata.General.title = 'DPC Beta y'
        
        if mode in ['mod', 'both']:
            # mDPC
            mdpcx = ((self.o1+self.o2)-(self.o0+self.o3)) / self.dpc_sum
            mdpcy = ((self.o2+self.o3)-(self.o0+self.o1)) / self.dpc_sum
            
            self.mdpc_betax = mdpcx.deepcopy()
            self.mdpc_betay = mdpcy.deepcopy()
            self.mdpc_betax.data[:] = self._beta_from_dpc_sig(mdpcx.data)
            self.mdpc_betay.data[:] = self._beta_from_dpc_sig(mdpcy.data)
            self.mdpc_betax.metadata.General.title = "mDPC Beta x"
            self.mdpc_betay.metadata.General.title = 'mDPC Beta y'
    
    def _calc_mdpc_ims(self):
        # Calculate DPC and mDPC images.
        
        # (annular) BF
        self.BFi = self.i0+self.i1+self.i2+self.i3
        self.BFi.metadata.General.title = "DPC BF Inner"
        if self._oct_seg:
            self.BFo = self.o0+self.o1+self.o2+self.o3
            self.BFo.metadata.General.title = "DPC BF Outer"
            self.BFt = self.BFi+self.BFo
        else:
            self.BFt = self.BFi
        self.BFt.metadata.General.title = "DPC BF Total"
        
        
        # DPC
        if self.sum_method == self.sum_methods[0]:
            self.dpc_sum = self.BFt
        elif self.sum_method == self.sum_methods[1]:
            self.dpc_sum = self.BFt.deepcopy()
            self.dpc_sum.data[:] = self.BFt.mean(0).mean(0).data
            self.dpc_sum.metadata.General.title = "DPC Sum "+self.sum_method
        elif self.sum_method == self.sum_methods[2]:          
            self.dpc_sum = self.BFt.deepcopy()
            self.dpc_sum.data[:] = np.percentile(self.BFt.data, 50)
            self.dpc_sum.metadata.General.title = "DPC Sum "+self.sum_method
            
        elif self.sum_method == self.sum_methods[3]:
            # based on
            # https://stackoverflow.com/questions/18552011/3d-curvefitting/18648210#18648210
            # https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6
            data_im = self.BFt.data

            # regular grid covering the domain of the data
            X, Y = np.meshgrid(np.arange(data_im.shape[1]), np.arange(data_im.shape[0]))
            XX = X.flatten()
            YY = Y.flatten()
            ZZ = data_im.flatten()
            #ZZ = (3*XX+0.5*YY).flatten()   # test data
            data = np.array(list(zip(XX, YY, ZZ)))

            # best-fit linear plane
            A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
            from scipy.linalg import lstsq
            C, _, _, _ = lstsq(A, data[:, 2])    # coefficients

            # evaluate it on grid
            Z = C[0]*X + C[1]*Y + C[2]

            self.dpc_sum = self.BFt.deepcopy()
            self.dpc_sum.data[:] = Z
            self.dpc_sum.metadata.General.title = "DPC Sum " + self.sum_method
        if self._oct_seg:
            self._dpc_beta(mode='both')
        else:
            self._dpc_beta(mode='std')    
        
    def _rot(self, y, x, deg):
        # rotate vector by angle
        r = np.hypot(y, x)
        t = np.arctan2(y, x)
        t2 = t+np.deg2rad(self.scan_deg)
        y2 = r*np.sin(t2)
        x2 = r*np.cos(t2)
        return y2, x2
    
    def _correct_for_scan_angle(self, scan_deg):
        # scan_deg < 0 means vector has to be rotated anticlockwise
        self.dpc_betay.data[:], self.dpc_betax.data[:] = self._rot(self.dpc_betay.data, self.dpc_betax.data, scan_deg)
        if self._oct_seg:
            self.mdpc_betay.data[:], self.mdpc_betax.data[:] = self._rot(self.mdpc_betay.data, self.mdpc_betax.data, scan_deg)
