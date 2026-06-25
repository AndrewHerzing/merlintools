
import unittest
import numpy as np
import matplotlib.pylab as plt

import fpd


class TestFFT(unittest.TestCase):
    def setUp(self):
        self.ims = np.random.rand(2, 3, *(256,)*2) - 0.5
    
    def test_BandpassPlotter_1(self):
        print(self.id())
        bp = fpd.fft_tools.BandpassPlotter(self.ims[0,0], (0.053, 0.71), gy=2, fact=0.75, cmap='viridis')
        plt.close('all')
    
    def test_BandpassPlotter_2(self):
        print(self.id())
        ims_pass, (im_fft, mask, extent) = fpd.fft_tools.bandpass(self.ims, (0.053, 0.71), gy=2)
        plt.close('all')
    
    def test_BandpassPlotter_3(self):
        print(self.id())
        ims_pass, (im_fft, mask, extent) = fpd.fft_tools.bandpass(self.ims, (0.053, 0.71), gy=2, mode='horiz')
        plt.close('all')
    
    def test_BandpassPlotter_4(self):
        print(self.id())
        ims_pass, (im_fft, mask, extent) = fpd.fft_tools.bandpass(self.ims, (0.053, 0.71), gy=2, mode='vert')
        plt.close('all')
    
    def test_BandpassPlotter_modes(self):
        print(self.id())
        bp = fpd.fft_tools.BandpassPlotter(self.ims[0,0], (0.053, 0.71), gy=2, fact=0.75, cmap='viridis', mode='horiz')
        plt.close('all')
        
        bp = fpd.fft_tools.BandpassPlotter(self.ims[0,0], (0.053, 0.71), gy=2, fact=0.75, cmap='viridis', mode='vert')
        plt.close('all')
        
    def test_BandpassPlotter_events(self):
        print(self.id())
        bp = fpd.fft_tools.BandpassPlotter(self.ims[0,0], (0.053, 0.71), gy=2, fact=0.75, cmap='viridis')
        
        bp.update_mode('vert')
        bp.update_norm('Log')
        #bp.update_plots()
        #bp.update_mask(None)
        #bp.update_plots(None)
        bp.update_fact(None)
        
        plt.close('all')
    
    def test_fft2_diff(self):
        print(self.id())
        from scipy.integrate import cumtrapz
        
        pix=0.5
        n = 256
        yi, xi = np.indices((n,)*2)

        p = n // 1

        ## image
        imy = np.sin(2*np.pi * yi/p*1)
        imx = np.cos(2*np.pi * xi/p*2)
        im = imy + imx
        plt.matshow(im); plt.colorbar()

        # 1st and 2nd dif, and int
        gyf, gxf = fpd.fft_tools.fft2_diff(im, order=1)
        gyf, gxf = fpd.fft_tools.fft2_diff(im, order=1, ypix=pix, xpix=pix)
        gyyf, gxxf = fpd.fft_tools.fft2_diff(im, order=2, ypix=pix, xpix=pix)
        iyf, ixf = fpd.fft_tools.fft2_diff(im, order=-1, ypix=pix, xpix=pix)
        
        # numerical
        gy = np.gradient(im, pix, axis=0)
        gx = np.gradient(im, pix, axis=1)
        gyy = np.gradient(gy, pix, axis=0)
        gxx = np.gradient(gx, pix, axis=1)
        iy = cumtrapz(imy, dx=pix, axis=0, initial=0)
        ix = cumtrapz(imx, dx=pix, axis=1, initial=0)
        iy -= iy.mean()
        ix -= ix.mean()
        
        ## plots
        ##plt.matshow(gy); plt.colorbar()
        ##plt.matshow(gyf); plt.colorbar()
        #plt.matshow(gyf/gy); plt.colorbar()

        ##plt.matshow(gx); plt.colorbar()
        ##plt.matshow(gxf); plt.colorbar()
        #plt.matshow(gxf/gx); plt.colorbar()

        #plt.matshow(gyyf/gyy); plt.colorbar()
        #plt.matshow(gxxf/gxx); plt.colorbar()
        
        #plt.matshow(iy); plt.colorbar()
        #plt.matshow(iyf); plt.colorbar()

        #plt.matshow(ix); plt.colorbar()
        #plt.matshow(ixf); plt.colorbar()

        plt.matshow(iyf/iy); plt.colorbar()
        plt.matshow(ixf/ix); plt.colorbar()
        
        ## Check accuracy
        for (a, b) in zip([gyf, gxf, gyyf, gxxf, iyf, ixf],
                          [gy, gx, gyy, gxx, iy, ix]):
            #print( np.abs(a-b).max() )
            #plt.matshow( a-b )
            assert np.allclose(a, b, rtol=5e-03, atol=5e-03)
            # numerical (non-fft, and non-exact) values seem to need higher tol in checks
    
    def test_fft_diff(self):
        print(self.id())
        from scipy.integrate import cumtrapz
        
        pix=0.5
        n = 256
        xi = np.arange(n)

        p = n // 1

        ## image
        y = np.sin(2*np.pi * xi/p*1)
        #plt.figure()
        #plt.plot(xi, y)

        # 1st and 2nd dif, and int
        gyf = fpd.fft_tools.fft_diff(y, order=1)
        gyf = fpd.fft_tools.fft_diff(y, order=1, xpix=pix)
        gyyf = fpd.fft_tools.fft_diff(y, order=2, xpix=pix)
        iyf = fpd.fft_tools.fft_diff(y, order=-1, xpix=pix)
        
        # numerical
        gy = np.gradient(y, pix)
        gyy = np.gradient(gy, pix)
        iy = cumtrapz(y, dx=pix, initial=0)
        iy -= iy.mean()

        #plt.matshow(iyf/iy); plt.colorbar()
        
        ## Check accuracy
        for (a, b) in zip([gyf, gyyf, iyf],
                          [gy, gyy, iy]):
            #print( np.abs(a-b).max() )
            #plt.matshow( a-b )
            assert np.allclose(a, b, rtol=5e-03, atol=5e-03)
            # numerical (non-fft, and non-exact) values seem to need higher tol in checks
    
    
    def test_laplacians(self):
        print(self.id())
        pix = 0.2
        n = 256
        yi, xi = np.indices((n,)*2)

        p = n // 1

        ## image
        imy = np.sin(2*np.pi * yi/p*1)
        imx = np.cos(2*np.pi * xi/p*2)
        im = imy + imx
        #plt.matshow(im); plt.colorbar()

        # fft2_diff has seperate check against regular gradient calcs, so use
        # here to avoid larger errors by sticking to freq space
        gyyf, gxxf = fpd.fft_tools.fft2_diff(im, order=2, ypix=pix, xpix=pix)
        lapgf = gyyf + gxxf
        
        lapf = fpd.fft_tools.fft2_laplacian(im)
        lapf = fpd.fft_tools.fft2_laplacian(im, ypix=pix, xpix=pix)
        
        ilap = fpd.fft_tools.fft2_ilaplacian(lapf)
        ilap = fpd.fft_tools.fft2_ilaplacian(lapf, ypix=pix, xpix=pix)
        #plt.matshow(ilap/im); plt.colorbar()
        
        ## could also use regular calcs, but errors would be higher.
        #gy = np.gradient(im, pix, axis=0)
        #gx = np.gradient(im, pix, axis=1)
        #gyy = np.gradient(gy, pix, axis=0)
        #gxx = np.gradient(gx, pix, axis=1)
        #lap = gyy + gxx

        ##plt.matshow(lapgf/lap); plt.colorbar()
        ##plt.matshow(lapf/lap); plt.colorbar()
        
        ## Check accuracy
        for (a, b) in zip([lapf, ilap],
                          [lapgf, im]):
            #print( np.abs(a-b).max() )
            #plt.matshow( a-b )
            assert np.allclose(a, b, rtol=1e-05, atol=1e-08)
    
    def test_igrad(self):
        print(self.id())
        pix = 0.2
        n = 256
        yi, xi = np.indices((n,)*2)

        p = n // 1

        ## image
        imy = np.sin(2*np.pi * yi/p*1)
        imx = np.cos(2*np.pi * xi/p*2)
        im = imy + imx + yi / n *2
        #plt.matshow(im); plt.colorbar()

        # fft2_diff has seperate check against regular gradient calcs, so use
        # here to avoid larger errors by sticking to freq space
        gyyf, gxxf = fpd.fft_tools.fft2_diff(im, order=1, ypix=pix, xpix=pix)
        
        imr = fpd.fft_tools.fft2_igrad(gyyf, gxxf)
        imr = fpd.fft_tools.fft2_igrad(gyyf, gxxf, ypix=pix, xpix=pix)
        #plt.matshow(im)
        #plt.matshow(imr)
        #plt.matshow(imr - (im - im.mean())); plt.colorbar()
        assert np.allclose(im - im.mean(), imr, rtol=1e-02, atol=1e-02)
    
    def test_im2fftrdf_none(self):
        print(self.id())
        p = 32
        py = px = p
        y, x = np.indices((256,)*2)
        im = np.zeros(y.shape, dtype=float)
        im += np.sin(y/py*2*np.pi)
        im += np.sin(x/px*2*np.pi)
        #plt.matshow(im)
        
        r, i = fpd.fft_tools.im2fftrdf(im, pad_to=None)
        #plt.figure()
        #plt.plot(r, i)
        
        assert np.allclose(p, 1.0 / r[i.argmax()])
        
    def test_im2fftrdf_sq(self):
        print(self.id())
        p = 32
        py = px = p
        y, x = np.indices((256,)*2)
        im = np.zeros(y.shape, dtype=float)
        im += np.sin(y/py*2*np.pi)
        im += np.sin(x/px*2*np.pi)
        #plt.matshow(im)
        
        r, i = fpd.fft_tools.im2fftrdf(im, pad_to='square')
        #plt.figure()
        #plt.plot(r, i)
        
        assert np.allclose(p, 1.0 / r[i.argmax()])
    
    def test_fft2rdf(self):
        print(self.id())
        p = 32
        py = px = p
        y, x = np.indices((256,)*2)
        im = np.zeros(y.shape, dtype=float)
        im += np.sin(y/py*2*np.pi)
        im += np.sin(x/px*2*np.pi)
        #plt.matshow(im)
        
        from numpy.fft import fft2, fftfreq
        # freqs
        rows, cols = im.shape
        yf = fftfreq(rows, 1)
        xf = fftfreq(cols, 1)
        
        yyf, xxf = np.meshgrid(yf, xf, indexing='ij')
        rrf = np.hypot(yyf, xxf)
        
        # fft
        im_fft = np.abs(fft2(im))
        
        r, i = fpd.fft_tools.fft2rdf(rrf, im_fft)
        #plt.figure()
        #plt.plot(r, i)
        
        assert np.allclose(p, 1.0 / r[i.argmax()])
    
    def test_cepstrum2(self):
        print(self.id())
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
        #plt.matshow(im)

        # calculate cepstrum and plot
        cp = cepstrum2(im, plot=False, sigma=None)
        cp = cepstrum2(im, plot=False)
        cp = cepstrum2(im, plot=True)
    
    def test_cepstrum(self):
        print(self.id())
        from scipy.ndimage import gaussian_filter1d
        from fpd.fft_tools import cepstrum

        # generate wonky points
        a = np.arange(512)
        b = np.where((a-16) % 32 == 0)[0]
        b = np.random.choice(b, int(len(b) * 0.5), replace=False)
        b += np.round(np.random.random(b.shape), 0).astype(int)

        # envelope
        e = (a - a.mean())
        e = e / e.max() * np.pi/4
        e = np.cos(e)

        # generate data
        a = np.zeros(a.shape)
        a[b] = 1
        a *= e

        # smooth
        a = gaussian_filter1d(a, 1.0) * 4

        ## add noise
        #a = a + np.random.random(a.shape) / 10.0

        # plot input
        plt.figure()
        plt.plot(a)

        # calculate cepstrum
        cp = cepstrum(a, plot=False)
        cp = cepstrum(a, plot=True)
    
    def test_fftcorrelate(self):
        print(self.id())
        import scipy as sp
        
        ### make image
        im_shape = (256,)*2
        d0 = fpd.synthetic_data.disk_image(intensity=128, radius=10, size=im_shape[0], sigma=0.5)

        ### ensure it is centred
        comyx = sp.ndimage.center_of_mass(d0)
        dyx = -(comyx - (np.array(d0.shape)-1) / 2.0)
        if not np.allclose(dyx, 0):
            d0 = fpd.synthetic_data.shift_im(d0, dyx=dyx)
        #plt.matshow(d0)
        #plt.matshow(d0*1.0 - d0[::-1, ::-1])
        comyx0 = sp.ndimage.center_of_mass(d0)

        ### make shifted image
        d1 = fpd.synthetic_data.shift_im(d0, dyx=(-75, 50))
        #plt.matshow(d1)
        comyx1 = sp.ndimage.center_of_mass(d1)

        ### test
        atol = 0.5
        im_pairs = [(d0, d0), (d1, d1), (d1, d0)]
        ns = [0, 0.5, 1]
        cyx_refs = [comyx0, comyx0, comyx1]
        
        for (im1, im2), n, cyx_ref in zip(im_pairs, ns, cyx_refs): 
            c = fpd.fft_tools.fftcorrelate(im1, im2, phase_exponent=n)
            #plt.matshow(c)
            cyx_detected = np.unravel_index(np.argmax(c), c.shape)
            assert np.allclose(cyx_detected, cyx_ref, atol=atol)

    
if __name__ == '__main__':
    unittest.main()


