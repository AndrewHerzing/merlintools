
import unittest
import numpy as np
import matplotlib.pylab as plt

from fpd import mag_tools 


class TestMag_tools(unittest.TestCase):   
    def test_landau(self):
        print(self.id())
        my, mx = mag_tools.landau()
        my, mx = mag_tools.landau(shape=(256,)*2)
        my, mx = mag_tools.landau(plot=True)
        my, mx = mag_tools.landau(origin='top')
        my, mx = mag_tools.landau(origin='bottom')
        my, mx = mag_tools.landau(pad_width=None)
        my, mx = mag_tools.landau(pad_width=50)
        plt.close('all')
    
    def test_stripes(self):
        print(self.id())
        my, mx = mag_tools.stripes()
        my, mx = mag_tools.stripes(shape=(256,)*2)
        my, mx = mag_tools.stripes(plot=True)
        my, mx = mag_tools.stripes(origin='top')
        my, mx = mag_tools.stripes(origin='bottom')
        my, mx = mag_tools.stripes(pad_width=None)
        my, mx = mag_tools.stripes(pad_width=50)
        my, mx = mag_tools.stripes(h2h=True)
        my, mx = mag_tools.stripes(nstripes=8)
        plt.close('all')
    
    def test_uniform(self):
        print(self.id())
        my, mx = mag_tools.uniform()
        my, mx = mag_tools.uniform(shape=(256,)*2)
        my, mx = mag_tools.uniform(plot=True)
        my, mx = mag_tools.uniform(origin='top')
        my, mx = mag_tools.uniform(origin='bottom')
        my, mx = mag_tools.uniform(pad_width=None)
        my, mx = mag_tools.uniform(pad_width=50)
        plt.close('all')
    
    def test_grad(self):
        print(self.id())
        my, mx = mag_tools.grad()
        my, mx = mag_tools.grad(shape=(256,)*2)
        my, mx = mag_tools.grad(plot=True)
        my, mx = mag_tools.grad(origin='top')
        my, mx = mag_tools.grad(origin='bottom')
        my, mx = mag_tools.grad(pad_width=None)
        my, mx = mag_tools.grad(pad_width=50)
        plt.close('all')
    
    def test_divergent(self):
        print(self.id())
        my, mx = mag_tools.divergent()
        my, mx = mag_tools.divergent(shape=(256,)*2)
        my, mx = mag_tools.divergent(plot=True)
        my, mx = mag_tools.divergent(origin='top')
        my, mx = mag_tools.divergent(origin='bottom')
        my, mx = mag_tools.divergent(pad_width=None)
        my, mx = mag_tools.divergent(pad_width=50)
        plt.close('all')
    
    def test_neel(self):
        print(self.id())
        my, mx = mag_tools.neel()
        my, mx = mag_tools.neel(shape=(256,)*2)
        my, mx = mag_tools.neel(plot=True)
        my, mx = mag_tools.neel(origin='top')
        my, mx = mag_tools.neel(origin='bottom')
        my, mx = mag_tools.neel(pad_width=None)
        my, mx = mag_tools.neel(pad_width=50)
        my, mx = mag_tools.neel(width=32)
        plt.close('all')
        
    def test_vortex(self):
        print(self.id())
        my, mx = mag_tools.vortex()
        my, mx = mag_tools.vortex(shape=(256,)*2)
        my, mx = mag_tools.vortex(plot=True)
        my, mx = mag_tools.vortex(origin='top')
        my, mx = mag_tools.vortex(origin='bottom')
        my, mx = mag_tools.vortex(pad_width=None)
        my, mx = mag_tools.vortex(pad_width=50)
        my, mx = mag_tools.vortex(r_edge=100)
        my, mx = mag_tools.vortex(lambda_c=10)
        my, mx = mag_tools.vortex(lambda_c=0)
        plt.close('all')
    
    def test_cross_tie(self):
        print(self.id())
        my, mx = mag_tools.cross_tie()
        my, mx = mag_tools.cross_tie(shape=(256,)*2)
        my, mx = mag_tools.cross_tie(plot=True)
        my, mx = mag_tools.cross_tie(origin='top')
        my, mx = mag_tools.cross_tie(origin='bottom')
        my, mx = mag_tools.cross_tie(pad_width=None)
        my, mx = mag_tools.cross_tie(pad_width=50)
        my, mx = mag_tools.cross_tie(lambda_c=10)
        plt.close('all')
    
    def test_mag_phase(self):
        print(self.id())
        my, mx = mag_tools.landau()
        p = mag_tools.mag_phase(my, mx, thickness=10e-9, phase_amp=10, plot=False)
        p = mag_tools.mag_phase(my, mx, thickness=10e-9, phase_amp=10, plot=True)
        
        p = mag_tools.mag_phase(my, mx, thickness=10e-9, phase_amp=10, plot=False, origin='bottom')
        p = mag_tools.mag_phase(my, mx, thickness=10e-9, phase_amp=10, plot=False, pad_width=16)
        plt.close('all')
    
    def test_analytical(self):
        print(self.id())
        my, mx = mag_tools.landau()
        p = mag_tools.mag_phase(my, mx, thickness=10e-9, phase_amp=10, plot=False)
        
        phase_grad0 = np.percentile(p.phase_grady, 97)
        bt = mag_tools.phasegrad2bt(phase_grad0)
        phase_grad1 = mag_tools.bt2phasegrad(bt)
        assert np.isclose(phase_grad0, phase_grad1)

        beta = mag_tools.bt2beta(bt)
        phase_grady2 = mag_tools.beta2phasegrad(beta)
        assert np.isclose(phase_grady2, phase_grad0)
        
        bt2 = mag_tools.beta2bt(beta)
        assert np.allclose(bt, bt2)
        
        assert np.allclose(mag_tools.tesla2mag(mag_tools.mag2tesla(1)), 1)
        assert np.allclose(mag_tools.tesla2G(1), 1e4)
        assert np.allclose(mag_tools.tesla2Oe(1), 1e4)
        
        period = mag_tools.phasegrad2period(phase_grad0)
    
    def test_exchange_length(self):
        print(self.id())
        lex = mag_tools.exchange_length(800e3, 13e-12)
        assert lex > 0
    
    def test_fresnel_paraxial(self):
        print(self.id())
        a = np.ones((256, 256))
        phase = a
        im = mag_tools.fresnel_paraxial(a, phase, df=0, ypix=1e-9, xpix=1e-9, theta_c=0, kV=200.0)
        
    def test_fresnel(self):
        print(self.id())
        a = np.ones((256, 256))
        phase = a
        
        # anticlockwise
        my, mx = mag_tools.vortex()
        p = mag_tools.mag_phase(my, mx)
        a = p.phase * 0 + 1
        
        # centre should be bright
        cy, cx = np.fix(np.array(a.shape) / 2.0).astype(int)
        assert p.phase[cy, cx] > p.phase[0, 0]
        
        # centre should be bright
        vfren = mag_tools.fresnel(p.phase*0+1, p.phase, df=1e-3)
        assert vfren.im[cy, cx] > 1.0
    
    def test_z_tie(self):
        print(self.id())
        kV = 200.0
        my_ms = 800e3
        lex = mag_tools.exchange_length(my_ms, 13e-12) * 30
        lex_nm = lex * 1e9

        size_nm = 5000
        n_pix = 256
        pix_nm = (size_nm / n_pix)
        pix_m = pix_nm * 1e-9
        lambda_c = lex_nm / pix_nm 

        my, mx = mag_tools.cross_tie(lambda_c=n_pix//2, shape=(n_pix,)*2, pad_width=[(128*2,)*2, (0,)*2])

        ypix = xpix = pix_m
        thickness = 2e-09
        p = mag_tools.mag_phase(my, mx, ypix=ypix, xpix=xpix, thickness=thickness, plot=False)
        phase = p.phase
        a = np.ones_like(phase)
        
        c_s = 0
        df = 1e-5
        im0 = mag_tools.fresnel(a, phase, df=0,   ypix=ypix, xpix=xpix, c_s=c_s, theta_c=0, kV=kV, plot=False).im
        imp = mag_tools.fresnel(a, phase, df=+df, ypix=ypix, xpix=xpix, c_s=c_s, theta_c=0, kV=kV, plot=False).im
        imn = mag_tools.fresnel(a, phase, df=-df, ypix=ypix, xpix=xpix, c_s=c_s, theta_c=0, kV=kV, plot=False).im

        im_grad = (imp - imn) / (2 * df)
        tie_phase = mag_tools.tie(im_grad, im0, ypix=ypix, xpix=xpix, kV=kV)
        
        #plt.matshow(phase); plt.colorbar()
        #plt.matshow(tie_phase); plt.colorbar()
        #plt.matshow(tie_phase-phase); plt.colorbar()
        
        #print(np.abs(tie_phase - phase).max())
        #print(np.abs(tie_phase/ phase -1).max())
        assert np.allclose(tie_phase, phase, rtol=1e-3, atol=1e-3)
    
    def test_tem_dpc(self):
        print(self.id())
        from scipy.ndimage import gaussian_filter, gaussian_filter1d

        from fpd.tem_tools import AlignNR
        from fpd.mag_tools import cross_tie, fresnel, mag_phase

        np.random.seed(19)

        my_ms = 800e3
        thickness = 4e-09
        size_nm = 5000
        n_pix = 128
        df = 2e-3
        c_s = 0
        kV = 200

        pix_nm = (size_nm / n_pix)
        pix_m = pix_nm * 1e-9


        # cross tie
        my, mx = cross_tie(lambda_c=n_pix//2, shape=(n_pix, n_pix*2))

        # make periodic
        my = np.vstack([my, my[::-1]])
        mx = np.vstack([mx, mx[::-1]])
        s = np.s_[:n_pix, :n_pix]

        Bs = mag_tools.mu_0 * my_ms
        by =  Bs * my
        bx =  Bs * mx
        Byx_orig = np.array([by[s], bx[s]])


        # calculate phase
        p = mag_phase(by, bx, ypix=pix_m, xpix=pix_m, thickness=thickness, plot=False) 
        phase = p.phase
        phase -= phase[s].mean()

        a = np.ones_like(phase)


        # fixed noise
        p_noise = (np.random.rand(*phase.shape) - 0.5) * phase.ptp() * 11 / 100
        p_noise = gaussian_filter(p_noise, 1)

        # make images
        im_f_p_set = fresnel(a, phase + p_noise, df=df, ypix=pix_m, xpix=pix_m, c_s=c_s, theta_c=0, kV=kV, plot=False).im[s]
        im_f_p_set_nmp = fresnel(a, p_noise, df=df, ypix=pix_m, xpix=pix_m, c_s=c_s, theta_c=0, kV=kV, plot=False).im[s]

        # take gradients
        sigma = 1
        ims_rb = np.array([im_f_p_set, im_f_p_set_nmp])
        dy = gaussian_filter1d(ims_rb, sigma=sigma, order=1, axis=1, mode='wrap')
        dx = gaussian_filter1d(ims_rb, sigma=sigma, order=1, axis=2, mode='wrap')
        ims_rb_g = np.hypot(dy, dx)

        # pad for edge effects
        pad = 16
        pad_zh = np.zeros([2, ims_rb_g.shape[1], pad])
        ims_rb_gp = np.concatenate([pad_zh, ims_rb_g, pad_zh], axis=2)
        pad_zv = np.zeros([2, pad, ims_rb_gp.shape[2]])
        ims_rb_gp = np.concatenate([pad_zv, ims_rb_gp, pad_zv], axis=1)


        # run alignment
        display_align = False
        if display_align:
            A = AlignNR(ims_rb_gp, reference=ims_rb_gp[1], reg_sigma=8, alpha=10, niter=40, nrmse_min=0, nrmse_rel=0, grad_sigma=sigma)
        else:
            A = AlignNR(ims_rb_gp[0], reference=ims_rb_gp[1], reg_sigma=8, alpha=10, niter=40, nrmse_min=0, nrmse_rel=0, grad_sigma=sigma, plot=False)
        dyx = A.dyx[0, ..., pad:-pad, pad:-pad]
        
        # recover B
        Byx = mag_tools.tem_dpc(dyx, df, thickness, pix_m, kV=kV)
        #plt.matshow((Byx - Byx_orig)[0]); plt.colorbar()
        #plt.matshow((Byx - Byx_orig)[1]); plt.colorbar()
        
        assert np.allclose(Byx_orig[10:-10], Byx[10:-10], rtol=0.01, atol=0.01)



if __name__ == '__main__':
    unittest.main()






















