
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import unittest
import numpy as np
import matplotlib.pylab as plt

import fpd
from fpd.ransac_tools import ransac_1D_fit
from fpd.ransac_tools import ransac_im_fit


def close_enough(a, b, rtol, atol):
    #print(np.abs(a-b).max())
    dif = np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False)
    return np.all(dif)


class TestRansacTools(unittest.TestCase):
    def setUp(self):
        im_shape = (64,)*2
        yy, xx = np.indices(im_shape)
        x0 = xx.mean()
        y0 = yy.mean()
        
        self.im_plane = (xx*2 + yy*1 -9.1)
        self.im_para = 0.16*((xx-x0)**2 + (yy-y0)**2) -200
        
        # combination of the two
        self.im_pp = np.minimum(self.im_plane, self.im_para)
        
        # 1d test data
        self.x = np.array([1, 2.5, 3.5, 4, 5, 7, 8.5])
        self.y = np.array([0.3, 1.1, 1.5, 2.0, 3.2, 6.6, 8.6])
        
    
    def test_func(self):
        print(self.id())
        im = self.im_para
        p0 = (0.1, 10, 20, 0)
        def model_f(p, *args):
            (x, y, z) = args
            m = np.abs(p[0])*((x-p[1])**2 + (y-p[2])**2) + p[3]
            return m
        
        fit, inliers, n = ransac_im_fit(im, mode=0, residual_threshold=0.01,
                                        model_f=model_f, p0=p0, 
                                        min_samples=6, max_trials=1000)
        self.assertTrue(close_enough(im, fit, rtol=1e-06, atol=1e-06))
    
    def test_func_param_pass(self):
        print(self.id())
        im = self.im_para
        p0 = (0.1, 10, 20, 0)
        def model_f(p, *args):
            (x, y, z) = args
            m = np.abs(p[0])*((x-p[1])**2 + (y-p[2])**2) + p[3]
            return m
        
        fit, inliers, n = ransac_im_fit(im, mode=0, residual_threshold=0.01,
                                        model_f=model_f, p0=p0, 
                                        min_samples=6, max_trials=1000,
                                        param_dict={'maxfev':10000})
        self.assertTrue(close_enough(im, fit, rtol=1e-06, atol=1e-06))
    
    def test_plane(self):
        print(self.id())
        im = self.im_plane
        fit, inliers, n = ransac_im_fit(im, mode=1, residual_threshold=0.01,
                                        min_samples=6, max_trials=1000)
        self.assertTrue(close_enough(self.im_plane, fit, rtol=1e-06, atol=1e-06))
    
    def test_plot(self):
        print(self.id())
        im = self.im_plane
        fit, inliers, n = ransac_im_fit(im, mode=1, residual_threshold=0.01,
                                        min_samples=6, max_trials=1000,
                                        plot=True)
        plt.close('all')
        
    def test_plane_fract(self):
        print(self.id())
        im = self.im_plane
        fit, inliers, n = ransac_im_fit(im, mode=1, residual_threshold=0.01,
                                        min_samples=6, max_trials=1000, fract=0.5)
        self.assertTrue(close_enough(self.im_plane, fit, rtol=1e-06, atol=1e-06))
    
    def test_plane_scale(self):
        print(self.id())
        im = self.im_plane
        fit, inliers, n = ransac_im_fit(im, mode=1, residual_threshold=0.1,
                                        min_samples=6, max_trials=1000, scale=True)
        self.assertTrue(close_enough(self.im_plane, fit, rtol=1e-06, atol=1e-06))
    
    def test_plane_mask(self):
        print(self.id())
        im = self.im_plane
        mask = np.random.rand(*im.shape) > 0.5
        fit, inliers, n = ransac_im_fit(im, mode=1, residual_threshold=0.01,
                                        min_samples=6, max_trials=1000, mask=mask)
        self.assertTrue(close_enough(self.im_plane, fit, rtol=1e-06, atol=1e-06))
        
    def test_quad(self):
        print(self.id())
        im = self.im_para
        fit, inliers, n = ransac_im_fit(im, mode=2, residual_threshold=0.01,
                                        min_samples=6, max_trials=1000)
        self.assertTrue(close_enough(self.im_para, fit, rtol=1e-06, atol=1e-06))
    
    def test_paraboloid(self):
        print(self.id())
        im = self.im_para
        fit, inliers, n = ransac_im_fit(im, mode=3, residual_threshold=0.01, 
                                        min_samples=6, max_trials=1000)
        self.assertTrue(close_enough(self.im_para, fit, rtol=1e-06, atol=1e-06))
    
    def test_spline(self):
        print(self.id())
        im = self.im_plane
        fit, inliers, n = ransac_im_fit(im, mode=4, residual_threshold=0.01, 
                                        min_samples=40, max_trials=1000,
                                        param_dict={'kx':1, 'ky':1})
        self.assertTrue(close_enough(self.im_plane, fit, rtol=1e-06, atol=1e-06))
        
        fit, inliers, n = ransac_im_fit(im, mode=4, residual_threshold=0.01, 
                                        min_samples=40, max_trials=1000)
        self.assertTrue(close_enough(self.im_plane, fit, rtol=1e-06, atol=1e-06))
    
    def test_full_fit(self):
        print(self.id())
        im = self.im_plane
        fit, inliers, n = ransac_im_fit(im, mode=1, residual_threshold=100,
                                        min_samples=1.0, max_trials=1)
        #self.assertTrue(close_enough(self.im_plane, fit, rtol=1e-06, atol=1e-06))
    
    def test_multi_image(self):
        print(self.id())
        im_shape = (64,)*2
        yy, xx = np.indices(im_shape)
        x0 = xx.mean()
        y0 = yy.mean()

        im_planes = np.array([(xx*p + yy*(p-1) -p**2) for p in [0.5, 1, 2, 4]])
        im_planes_mid = np.reshape(im_planes, (2, 2) + im_shape)
        im_planes_mid = np.rollaxis(im_planes_mid, 0, 4)

        # images at end
        fit, inlier, n = fpd.ransac_tools.ransac_im_fit(im_planes, plot=True)
        assert np.allclose(im_planes - fit, 0)
        plt.close('all')

        # images in middle
        fit, inlier, n = fpd.ransac_tools.ransac_im_fit(im_planes_mid, axes=(-3, -2), plot=False)
        assert np.allclose(im_planes_mid - fit, 0)
    
    # 1-D fits
    def test_full_fit_1d(self):
        print(self.id())
        def f(p, x):
            return p[0]*x**2 + p[1]*x + p[2]

        p0 = (1,1,1)
        fit, inliers, model = ransac_1D_fit(self.x, self.y, mode=0, model_f=f, p0=p0,
                                            min_samples=len(self.x))
    
    def test_full_fit_1d_sigma(self):
        print(self.id())
        def f(p, x):
            return p[0]*x**2 + p[1]*x + p[2]

        p0 = (1,1,1)
        param_dict = {'sigma' : self.x*0+5}
        fit, inliers, model = ransac_1D_fit(self.x, self.y, mode=0, model_f=f, p0=p0,
                                            param_dict=param_dict,
                                            min_samples=len(self.x))
    
    def test_linear_1d(self):
        print(self.id())
        fit, inliers, model = ransac_1D_fit(self.x, self.y, mode=1,
                                            min_samples=4)
    def test_linear_1d_mask(self):
        print(self.id())
        mask = np.random.rand(*self.x.shape) >= 0.8
        fit, inliers, model = ransac_1D_fit(self.x, self.y, mode=1,
                                            min_samples=4, mask=mask)
    
    def test_linear_1d_min_samples_float(self):
        print(self.id())
        fit, inliers, model = ransac_1D_fit(self.x, self.y, mode=1,
                                            min_samples=0.4)
        
    def test_1d_plot(self):
        print(self.id())
        fit, inliers, model = ransac_1D_fit(self.x, self.y, mode=1,
                                            plot=True, min_samples=4)
        plt.close('all')

    def test_quad_1d(self):
        print(self.id())
        fit, inliers, model = ransac_1D_fit(self.x, self.y, mode=2,
                                            min_samples=4)
    
    def test_cubic_1d(self):
        print(self.id())
        fit, inliers, model = ransac_1D_fit(self.x, self.y, mode=3,
                                            min_samples=4)

    def test_spline_1d(self):
        print(self.id())
        fit, inliers, model = ransac_1D_fit(self.x, self.y, mode=4,
                                            param_dict={'k':2, 's':600},
                                            min_samples=4)
    
    def test_spline_1d(self):
        print(self.id())
        fit, inliers, model = ransac_1D_fit(self.x, self.y, mode=4,
                                            min_samples=4)

if __name__ == '__main__':
    unittest.main()


