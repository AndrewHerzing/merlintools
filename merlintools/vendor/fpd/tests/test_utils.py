
import unittest
import numpy as np
from fpd.utils import *

class TestUtils(unittest.TestCase):
    def test_seq_unseq(self):
        print(self.id())
        a = np.random.rand(3, 4, 5, 6)

        # end axes
        axes=(-2, -1)

        av, unflat_shape = seq_image_array(a, axes=axes)
        assert av.shape == (3*4, 5, 6)
        assert (av[0] == a[0, 0]).all()
        assert (av[-1] == a[-1, -1]).all()

        a2 = unseq_image_array(av, axes, unflat_shape)
        assert a2.shape == a.shape
        assert (a2 == a).all()

        # middle axes
        axes=(-3, -2)

        av, unflat_shape = seq_image_array(a, axes=axes)
        assert av.shape == (3*6, 4, 5)
        assert (av[0] == a[0, ..., 0]).all()
        assert (av[-1] == a[-1, ..., -1]).all()

        a2 = unseq_image_array(av, axes, unflat_shape)
        assert a2.shape == a.shape
        assert (a2 == a).all()
    
    def test_gaus_im_div1d(self):
        print(self.id())
        im = np.random.rand(64, 64)
        im_div = gaus_im_div1d(im, sigma=2)
    
    def test_median_lcs(self):
        print(self.id())
        c = np.random.rand(20, 1)
        r = np.random.rand(1, 20)
        ca = np.repeat(c, 20, 1)
        ra = np.repeat(r, 20, 0)

        cac = median_lc(ca, 1)
        rac = median_lc(ra, 0)

        cacd = median_of_difference_lc(ca, 1)
        racd = median_of_difference_lc(ra, 0)

        cacd_diff = np.diff(cacd, 1, 0)
        racd_diff = np.diff(racd, 1, 1)

        #plt.matshow(ca)
        assert np.allclose(cac, cac[0, 0]) 
        assert np.allclose(rac, rac[0, 0]) 

        assert np.allclose(cacd_diff, cacd_diff[0, 0]) 
        assert np.allclose(racd_diff, racd_diff[0, 0])
        
        racs0 = median_lc(ra, 0, sigma=0)
        assert np.allclose(racs0, racs0[0, 0]) 
    
    def test_gaussian_2d(self):
        print(self.id())
        fit_hw = 7
        y, x = np.indices((fit_hw*2+1,)*2)-fit_hw
        g = gaussian_2d((y, x), 5, 0.5, 0.25, 2, 1, np.deg2rad(10), 1)
        
    def test_gaussian_2d_peak_fit_single(self):
        print(self.id())
        fit_hw = 7
        y, x = np.indices((fit_hw*2+1,)*2)
        
        args_in = (5, 5.5, 4.25, 2, 1, np.deg2rad(10), 1)
        
        g = gaussian_2d((y, x), *args_in)
        g = g.reshape(y.shape)
        
        popt, perr = gaussian_2d_peak_fit(g, yc=5, xc=4, fit_hw=2, smoothing=0, plot=False)        
        assert np.allclose(np.abs(popt), args_in)
        
        # plotting
        popt, perr = gaussian_2d_peak_fit(g, yc=5, xc=4, fit_hw=2, smoothing=0, plot=True)
    
        # smoothing
        popt, perr = gaussian_2d_peak_fit(g, yc=5, xc=4, fit_hw=2, smoothing=1, plot=False)
        
        # fit params
        popt, perr = gaussian_2d_peak_fit(g, yc=5, xc=4, fit_hw=2, smoothing=1, plot=False,
                                          maxfev=250)
        
    def test_gaussian_2d_peak_fit_single_edge_cases(self):
        print(self.id())
        fit_hw = 7
        y, x = np.indices((fit_hw*2+1,)*2)
        
        args_in = (5, 2.5, 2.25, 2, 1, np.deg2rad(10), 1)
        g = gaussian_2d((y, x), *args_in).reshape(y.shape)
        
        popt, perr = gaussian_2d_peak_fit(g, yc=2, xc=3, fit_hw=3, smoothing=0, plot=False)        
        assert (np.isnan(popt)).all()
        
        popt, perr = gaussian_2d_peak_fit(g, yc=3, xc=2, fit_hw=3, smoothing=0, plot=False)        
        assert (np.isnan(popt)).all()
        
        popt, perr = gaussian_2d_peak_fit(g, yc=12, xc=3, fit_hw=3, smoothing=0, plot=False)        
        assert (np.isnan(popt)).all()
        
        popt, perr = gaussian_2d_peak_fit(g, yc=3, xc=12, fit_hw=3, smoothing=0, plot=False)        
        assert (np.isnan(popt)).all()
        
        # plotting
        popt, perr = gaussian_2d_peak_fit(g, yc=2, xc=3, fit_hw=3, smoothing=0, plot=True)
    
    
    def test_gaussian_2d_peak_fit_multi(self):
        print(self.id())
        fit_hw = 19
        y, x = np.indices((fit_hw*2+1,)*2)
        
        g1 = gaussian_2d((y, x), 5, 5.5, 4.25, 2, 1, np.deg2rad(10), 1).reshape(y.shape)
        g2 = gaussian_2d((y, x), 4, 12.5, 22.25, 0.5, 2.1, np.deg2rad(30), -3).reshape(y.shape)
        g = g1 + g2
        
        popt, perr = gaussian_2d_peak_fit(g, yc=[5, 12], xc=[4, 22], fit_hw=[3, 3], smoothing=0, plot=False)
        assert np.allclose(popt[:, 1:3], [[5.5, 4.25], [12.5, 22.25]])
        
        # plot
        popt, perr = gaussian_2d_peak_fit(g, yc=[5, 12], xc=[4, 22], fit_hw=[3, 3], smoothing=0, plot=True)
        
        popt, perr = gaussian_2d_peak_fit(g, yc=[5, 12], xc=[4, 22], fit_hw=[3, 3], smoothing=0, plot=True, plot_mode='all')
        
        popt, perr = gaussian_2d_peak_fit(g, yc=[5, 12], xc=[4, 22], fit_hw=[3, 3], smoothing=0, plot=True, plot_mode='all', plot_log=False)
        
        # edge case
        popt, perr = gaussian_2d_peak_fit(g, yc=[5, 12], xc=[4, 22], fit_hw=[8, 3], smoothing=0, plot=False)
        assert (np.isnan(popt[0, :])).all()
        assert np.allclose(popt[1, 1:3], [12.5, 22.25])
        
        import matplotlib.pylab as plt
        plt.close('all')
    
    def test_snr_single_image(self):
        print(self.id())
        size = 128

        im = np.ones((size, size), dtype=float)*2
        im[:, size//2:] = 0
        im[size//2:, :] = im[size//2:, :][:, ::-1]
        im *= 2

        nim = (np.random.rand(size, size) -0.5)
        nim *= 1

        nsim = (im + nim)
        
        # total power from contrast
        snr_res = snr_single_image(nsim, s=1, w=2, plot=False)
        
        rtol = 0.1
        atol = 0.1 * size**2
        assert np.isclose(((im-im.mean())**2).sum(), snr_res.st, rtol=rtol, atol=atol)
        assert np.isclose((nim**2).sum(), snr_res.nt, rtol=rtol, atol=atol)
        
        # total power from contrast, per pixel
        snr_res = snr_single_image(nsim, s=1, w=2, plot=False, per_pixel=True)
        
        rtol = 0.1
        atol = 0.1
        assert np.isclose(((im-im.mean())**2).mean(), snr_res.st, rtol=rtol, atol=atol)
        assert np.isclose((nim**2).mean(), snr_res.nt, rtol=rtol, atol=atol)
        
        # total power from zero
        snr_res = snr_single_image(nsim, s=1, w=2, plot=False, signal_from='zero')
        
        rtol = 0.1
        atol = 0.1 * size**2
        assert np.isclose(((im)**2).sum(), snr_res.st, rtol=rtol, atol=atol)
        assert np.isclose((nim**2).sum(), snr_res.nt, rtol=rtol, atol=atol)
        
        # test plot
        snr_res = snr_single_image(nsim, s=1, w=2, plot=True)
        import matplotlib.pylab as plt
        plt.close('all')
    
    def test_snr_single_image_gaussian(self):
        print(self.id())
        size = 128

        im = np.ones((size, size), dtype=float)*2
        im[:, size//2:] = 0
        im[size//2:, :] = im[size//2:, :][:, ::-1]
        im *= 2

        nim = (np.random.rand(size, size) -0.5)
        nim *= 1
        nsim = (im + nim)
        
        # total power from contrast
        snr_res = snr_single_image(nsim, s=1, w=3, mode='gaussian', plot=False)
        
        rtol = 0.1
        atol = 0.1 * size**2
        assert np.isclose(((im-im.mean())**2).sum(), snr_res.st, rtol=rtol, atol=atol)
        assert np.isclose((nim**2).sum(), snr_res.nt, rtol=rtol, atol=atol)
        
        # total power from contrast, per pixel
        snr_res = snr_single_image(nsim, s=1, w=3, mode='gaussian', plot=False, per_pixel=True)
        
        rtol = 0.1
        atol = 0.1
        assert np.isclose(((im-im.mean())**2).mean(), snr_res.st, rtol=rtol, atol=atol)
        assert np.isclose((nim**2).mean(), snr_res.nt, rtol=rtol, atol=atol)
        
        # total power from zero
        snr_res = snr_single_image(nsim, s=1, w=3, mode='gaussian', plot=False, signal_from='zero')
        
        rtol = 0.1
        atol = 0.1 * size**2
        assert np.isclose(((im)**2).sum(), snr_res.st, rtol=rtol, atol=atol)
        assert np.isclose((nim**2).sum(), snr_res.nt, rtol=rtol, atol=atol)
        
        # test plot
        snr_res = snr_single_image(nsim, s=1, w=3, mode='gaussian', plot=True)
        import matplotlib.pylab as plt
        plt.close('all')
    
    def test_snr_single_image_wl(self):
        print(self.id())
        size = 128

        im = np.ones((size, size), dtype=float)*2
        im[:, size//2:] = 0
        im[size//2:, :] = im[size//2:, :][:, ::-1]
        im *= 2

        nim = (np.random.rand(size, size) -0.5)
        nim *= 1
        nsim = (im + nim)
        
        # total power from contrast
        snr_res = snr_single_image_wl(nsim)
        
        rtol = 0.1
        atol = 0.1 * size**2
        assert np.isclose(((im-im.mean())**2).sum(), snr_res.st, rtol=rtol, atol=atol)
        assert np.isclose((nim**2).sum(), snr_res.nt, rtol=rtol, atol=atol)
        
        # total power from contrast, per pixel
        snr_res = snr_single_image_wl(nsim, per_pixel=True)
        
        rtol = 0.1
        atol = 0.1
        assert np.isclose(((im-im.mean())**2).mean(), snr_res.st, rtol=rtol, atol=atol)
        assert np.isclose((nim**2).mean(), snr_res.nt, rtol=rtol, atol=atol)
        
        # total power from zero
        snr_res = snr_single_image_wl(nsim, signal_from='zero')
        
        rtol = 0.1
        atol = 0.1 * size**2
        assert np.isclose(((im)**2).sum(), snr_res.st, rtol=rtol, atol=atol)
        assert np.isclose((nim**2).sum(), snr_res.nt, rtol=rtol, atol=atol)
    
    def test_snr_two_image(self):
        print(self.id())
        from scipy.ndimage import gaussian_filter
        size = 128
        image = np.random.rand(size, size)
        ref = gaussian_filter(image, 4)
        snr = snr_two_image(image, ref)
    
    def test_decibel(self):
        print(self.id())
        assert decibel(1) == 0
        assert decibel(100) == 20
    
    
    def test_smooth7_2(self):
        print(self.id())
        n = smooth7(min_val=2, order=8, base_two=True, even=True)
        assert n==2
        assert np.issubdtype(n.dtype, np.integer)
        
        n = smooth7(min_val=255, order=8, base_two=True, even=True)
        assert n==256
        assert np.issubdtype(n.dtype, np.integer)
        
    def test_smooth7_even(self):
        print(self.id())
        n = smooth7(min_val=3, order=8, base_two=False, even=False)
        assert n==3
        assert np.issubdtype(n.dtype, np.integer)
        
        n = smooth7(min_val=3, order=8, base_two=False, even=True)
        assert n==4
        assert np.issubdtype(n.dtype, np.integer)
        
    def test_gaussian_fwhm(self):
        print(self.id())
        fwhm = gaussian_fwhm(1)
        np.allclose(fwhm, 2.3548200450309493)
    
    
    def test_Timer(self):
        print(self.id())
        with Timer('hello timer'):
            print('hello')
        
        with Timer():
            print('timeless world')
        
        with Timer('do nothing', print=False):
            print('no time')
    
    
    #-------------------------------------
    def test_int_factors_float(self):
        try:
            _ = int_factors(1.0)
        except ValueError as e:
            pass
    
    def test_int_factors_int(self):
        n = 12
        factors = int_factors(n)
        assert (factors == np.array([1, 2, 3, 4, 6, 12])).all()
        
        factors = int_factors(n, memory_efficient=True)
        assert (factors == np.array([1, 2, 3, 4, 6, 12])).all()
    
    
    #-------------------------------------
    def test_nearest_int_factor_float(self):
        try:
            _ = nearest_int_factor(1.0, 2.0)
        except ValueError as e:
            pass
    
    def test_nearest_int_factor_int(self):
        n = 12
        factor, factors = nearest_int_factor(n, 7)
        assert factor == 6
        assert (factors == np.array([1, 2, 3, 4, 6, 12])).all()
        
        factor, factors = nearest_int_factor(n, 7.0, memory_efficient=True)
        assert factor == 6
        assert (factors == np.array([1, 2, 3, 4, 6, 12])).all()
    
    
if __name__ == '__main__':
    unittest.main()
