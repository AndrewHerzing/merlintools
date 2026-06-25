
import unittest
import numpy as np

import fpd

def close_enough(a, b, rtol, atol):
    dif = np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False)
    return np.all(dif)


class TestFPDP(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.radius = 32
        sa = fpd.synthetic_data.shift_array(scan_len=9, shift_min=-2.0, 
                                            shift_max=2.0)
        self.sa = np.asarray(sa)
        self.im = fpd.synthetic_data.disk_image(intensity=128, radius=self.radius, 
                                                size=256, upscale=8, dtype='uint8')#float)
        self.sdpc_data = fpd.synthetic_data.shift_images(self.sa, self.im, origin='top')
        
        # error based on 1/100 pixel phase correlation
        self.rtol = 1e-09
        self.atol = 5e-02
        
        detectors = fpd.synthetic_data.segmented_detectors(im_shape=(256, 256),
                                                            rio=(24, 128), ac_det_roll=0)
        self.det_sigs = fpd.synthetic_data.segmented_dpc_signals(self.sdpc_data, 
                                                                 detectors)
        
        detectors_quad = fpd.synthetic_data.segmented_detectors(im_shape=(256, 256),
                                                                rio=(64, 128), ac_det_roll=0)
        self.det_sigs_quad = fpd.synthetic_data.segmented_dpc_signals(self.sdpc_data, 
                                                                      detectors_quad)[:4]
    
    
    #-----------------------------------------
    def test_segmented_dpc_class_oct(self):
        print(self.id())
        
        d = fpd.SegmentedDPC(self.det_sigs, alpha=self.radius)
        mdyx = [d.mdpc_betay.data, d.mdpc_betax.data]
        dyx = [d.dpc_betay.data, d.dpc_betax.data]
        
        self.assertTrue(close_enough(self.sa, np.asarray(mdyx), self.rtol, self.atol))
        self.assertTrue(close_enough(self.sa, np.asarray(dyx), self.rtol, self.atol))
    
    def test_segmented_dpc_class_quad(self):
        print(self.id())
        
        d = fpd.SegmentedDPC(self.det_sigs_quad, alpha=self.radius)
        dyx = [d.dpc_betay.data, d.dpc_betax.data]
        self.assertTrue(close_enough(self.sa, np.asarray(dyx), self.rtol, self.atol))

    def test_segmented_dpc_class_quad_fast_sum_methods(self):
        print(self.id())
        
        for sum_method in ['pixel', 'mean', 'percentile', 'plane']:
            d = fpd.SegmentedDPC(self.det_sigs_quad, alpha=self.radius, method='fast', sum_method=sum_method)
            dyx = [d.dpc_betay.data, d.dpc_betax.data]
            self.assertTrue(close_enough(self.sa, np.asarray(dyx), self.rtol, self.atol))
    
    
    #-----------------------------------------        
    def test_segmented_dpc_oct(self):
        print(self.id())
        
        d = fpd.segmented_dpc.SegmentedDPC(self.det_sigs, alpha=self.radius)
        mdyx = [d.mdpc_betay, d.mdpc_betax]
        dyx = [d.dpc_betay, d.dpc_betax]

        self.assertTrue(close_enough(self.sa, np.asarray(mdyx), self.rtol, self.atol))
        self.assertTrue(close_enough(self.sa, np.asarray(dyx), self.rtol, self.atol))
    
    def test_segmented_dpc_oct_scan_deg(self):
        print(self.id())
        
        d = fpd.segmented_dpc.SegmentedDPC(self.det_sigs, alpha=self.radius, scan_deg=360)
        mdyx = [d.mdpc_betay, d.mdpc_betax]
        dyx = [d.dpc_betay, d.dpc_betax]

        self.assertTrue(close_enough(self.sa, np.asarray(mdyx), self.rtol, self.atol))
        self.assertTrue(close_enough(self.sa, np.asarray(dyx), self.rtol, self.atol))
    
    def test_segmented_dpc_oct_fudge(self):
        print(self.id())
        
        def func(self):
            self.i0 = self.i1
        
        for fudge in [func, 'i0']:
            print('fudge: ', fudge)
            d = fpd.segmented_dpc.SegmentedDPC(self.det_sigs, alpha=self.radius, fudge=func)
            mdyx = [d.mdpc_betay, d.mdpc_betax]
            dyx = [d.dpc_betay, d.dpc_betax]

            self.assertTrue(close_enough(self.sa, np.asarray(mdyx), self.rtol, self.atol))
            self.assertTrue(close_enough(self.sa, np.asarray(dyx), self.rtol, self.atol))
    
    def test_segmented_dpc_oct_rebin(self):
        print(self.id())
        
        r = 3
        d = fpd.segmented_dpc.SegmentedDPC(self.det_sigs, alpha=self.radius, rebin=r)
        mdyx = [d.mdpc_betay, d.mdpc_betax]
        dyx = [d.dpc_betay, d.dpc_betax]
        
        sar_ns = [2] + [ti // r for ti in self.sa.shape[1:]]
        sar = fpd.fpd_processing.rebinA(self.sa, *sar_ns) / r**2
        
        self.assertTrue(close_enough(sar, np.asarray(mdyx), self.rtol, self.atol))
        self.assertTrue(close_enough(sar, np.asarray(dyx), self.rtol, self.atol))
        
    def test_segmented_dpc_oct_rebin_not_possible(self):
        print(self.id())
        
        r = 2
        d = fpd.segmented_dpc.SegmentedDPC(self.det_sigs, alpha=self.radius, rebin=r)
        mdyx = [d.mdpc_betay, d.mdpc_betax]
        dyx = [d.dpc_betay, d.dpc_betax]
        
        r = 1 # this should be the actual rebin used
        sar_ns = [2] + [ti // r for ti in self.sa.shape[1:]]
        sar = fpd.fpd_processing.rebinA(self.sa, *sar_ns) / r**2
        
        self.assertTrue(close_enough(sar, np.asarray(mdyx), self.rtol, self.atol))
        self.assertTrue(close_enough(sar, np.asarray(dyx), self.rtol, self.atol))
    
    def test_segmented_dpc_quad(self):
        print(self.id())
        
        d = fpd.segmented_dpc.SegmentedDPC(self.det_sigs_quad, alpha=self.radius)
        dyx = [d.dpc_betay, d.dpc_betax]
        self.assertTrue(close_enough(self.sa, np.asarray(dyx), self.rtol, self.atol))

    def test_segmented_dpc_quad_fast_sum_methods(self):
        print(self.id())
                
        for sum_method in ['pixel', 'mean', 'percentile', 'plane']:
            d = fpd.segmented_dpc.SegmentedDPC(self.det_sigs_quad, alpha=self.radius, method='fast',
                                               sum_method=sum_method)
            dyx = [d.dpc_betay, d.dpc_betax]
            self.assertTrue(close_enough(self.sa, np.asarray(dyx), self.rtol, self.atol))
    
    def test_segmented_dpc_quad_wrong(self):
        print(self.id())
        try:
            d = fpd.segmented_dpc.SegmentedDPC(self.det_sigs_quad, alpha=self.radius, method='WRONG')
        except ValueError as e:
            pass
        
        try:
            d = fpd.segmented_dpc.SegmentedDPC(self.det_sigs_quad, alpha=self.radius, sum_method='WRONG')
        except ValueError as e:
            pass
        
        try:
            d = fpd.segmented_dpc.SegmentedDPC('WRONG', alpha=self.radius)
        except ValueError as e:
            pass
        
        try:
            d = fpd.segmented_dpc.SegmentedDPC(self.det_sigs_quad, alpha=self.radius, fudge='WRONG')
        except ValueError as e:
            pass
        
        
    
    def test_load_gla_dpc(self):
        print(self.id())
        import tempfile
        from io import BytesIO
        from urllib.request import urlopen
        from zipfile import ZipFile
        import shutil

        ### gla dataset
        # Order and Disorder in the Magnetisation of the Chiral Crystal CrNb3S6
        url = 'https://researchdata.gla.ac.uk/829/1/Dataset_829.zip'
        
        try:
            tmp_d = tempfile.mkdtemp(prefix='fpd-')
            print('Connecting to repo...')
            with urlopen(url) as response:
                print('Downloading file...')
                with ZipFile(BytesIO(response.read())) as zipfile:
                    print('Extracting file...')
                    zipfile.extractall(path=tmp_d)
            print('Data acquired...')
            
            a, ds_dict, opt_dict = fpd.segmented_dpc.load_gla_dpc('201_default', path=tmp_d+'/data/')
            dpc = fpd.segmented_dpc.SegmentedDPC(a)
            #fpd.DPC_Explorer([dpc.mdpc_betay, dpc.mdpc_betay])
        finally:
            shutil.rmtree(tmp_d)


if __name__ == '__main__':
    unittest.main()

