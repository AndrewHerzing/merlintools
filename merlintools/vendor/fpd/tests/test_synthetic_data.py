
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import unittest
import numpy as np
import scipy as sp

import fpd


class TestSyntheticData(unittest.TestCase):       
    def test_fpd_data_view(self):
        print(self.id())
        im = np.ones((32,)*2)
        data = fpd.synthetic_data.fpd_data_view(im, (32,)*2)
        
        im_mem_pointer, im_read_only = im.__array_interface__['data']
        data_mem_pointer, data_read_only = data.__array_interface__['data']
        
        self.assertTrue((im_mem_pointer == data_mem_pointer) and data_read_only==True)
    
    def test_disk_image(self):
        print(self.id())
        disk = fpd.synthetic_data.disk_image(intensity=64, noise=False, ds_method='rebin')
        disk2 = fpd.synthetic_data.disk_image(intensity=64, noise=False, ds_method='interp')
        disk3 = fpd.synthetic_data.disk_image(dose=5, noise=False, ds_method='interp')
        disk4 = fpd.synthetic_data.disk_image(dose=5, noise=True, ds_method='interp')
    
    def test_poisson_noise(self):
        print(self.id())
        disk = fpd.synthetic_data.disk_image(intensity=64, noise=False, ds_method='rebin')
        noisy_disks = fpd.synthetic_data.poisson_noise(disk, samples=5)
        assert np.sum(noisy_disks[0] - noisy_disks[1]) != 0
        assert noisy_disks.shape == (5,) + disk.shape

    def test_shift_images(self):
        print(self.id())
        sa = fpd.synthetic_data.shift_array(scan_len=8, shift_min=-16.0, shift_max=16.0)
        disk_im = fpd.synthetic_data.disk_image(intensity=64)
        shifted_ims = fpd.synthetic_data.shift_images(sa, disk_im, method='linear', fill_value=0)
        shifted_ims = fpd.synthetic_data.shift_images(sa, disk_im, method='fourier', fill_value=0)
        shifted_ims = fpd.synthetic_data.shift_images(sa, disk_im, method='linear', fill_value=0,
                                                      noise=True)
    
    def test_shift_im(self):
        print(self.id())
        
        im = fpd.synthetic_data.disk_image(intensity=64)
        cyx0 = np.asarray(sp.ndimage.center_of_mass(im))
        
        shifts = [(-5, -10), (-5, +5), (5, 10)]
        for dyxi in shifts:
            for method in ['linear', 'fourier', 'pixel']:
                shifted_im = fpd.synthetic_data.shift_im(im, dyx=dyxi, method=method)
                cyx = np.asarray(sp.ndimage.center_of_mass(shifted_im))
                #plt.matshow(im)
                assert np.allclose(cyx - cyx0, dyxi)
    
    def test_array_image(self):
        print(self.id())
        im_shape = (256,)*2
        cyx = (np.array(im_shape) -1) / 2
        d0 = fpd.synthetic_data.disk_image(intensity=100, radius=10, size=im_shape[0], sigma=0.5)
        yxg = fpd.tem_tools.synthetic_lattice(cyx=cyx, ab=(50,)*2, angles=(0, np.pi/2), shape=im_shape, plot=True)
        yxg -= cyx
        im = fpd.synthetic_data.array_image(d0, yxg)
        #import matplotlib.pylab as plt
        #plt.ion()
        #plt.matshow(im)
        #plt.colorbar()
    
    def test_array_image_amps(self):
        print(self.id())
        im_shape = (256,)*2
        cyx = (np.array(im_shape) -1) / 2
        d0 = fpd.synthetic_data.disk_image(intensity=1, radius=10, size=im_shape[0], sigma=0)
        yxg = fpd.tem_tools.synthetic_lattice(cyx=cyx, ab=(50,)*2, angles=(0, np.pi/2), shape=im_shape, plot=True)
        
        im = fpd.synthetic_data.array_image(d0, yxg-cyx, amps=2)
        assert im.max() == 2
        
        amps = np.arange(len(yxg)) + 1
        im = fpd.synthetic_data.array_image(d0, yxg-cyx, amps=amps)
        yxg_ints = yxg.round(0).astype(int)
        im_amps = im[yxg_ints[:, 0], yxg_ints[:, 1]]
        print(im_amps, amps)
        assert np.allclose(im_amps, amps)
        
        #import matplotlib.pylab as plt
        #plt.ion()
        #plt.matshow(im)
        #plt.colorbar()
        
    def test_shift_array(self):
        print(self.id())
        for shift_type in [0, 1, 2]:
            sa = fpd.synthetic_data.shift_array(scan_len=8, shift_min=-16.0, shift_max=16.0,
                                                shift_type=shift_type)
        
if __name__ == '__main__':
    unittest.main()


