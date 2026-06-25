
import unittest
import numpy as np
import scipy as sp
import matplotlib.pylab as plt
import os

import fpd
import fpd.fpd_processing as fpdp


def close_enough(a, b, rtol, atol):
    dif = np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False)
    return np.all(dif)


def params_f(image, v):
    return v

def nonuniform_f(image):
    l = np.random.randint(2)+1
    return np.arange(l)


class TestFPDP(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.radius = 32
        sa = fpd.synthetic_data.shift_array(scan_len=9, shift_min=-2.0, 
                                            shift_max=2.0)
        self.sa = np.asarray(sa)
        self.im = fpd.synthetic_data.disk_image(intensity=128, radius=self.radius, 
                                                size=256, upscale=8, dtype='uint8')#float)
        self.data = fpd.synthetic_data.shift_images(self.sa, self.im, noise=False, origin='bottom')
        
        cyx = sp.ndimage.center_of_mass(self.im)
        self.cyx = np.asarray(cyx)
        
        # error based on 1/100 pixel phase correlation
        self.rtol = 1e-09
        self.atol = 5e-02
        
        # nrmse
        self.nrmse_ref_im = fpd.synthetic_data.disk_image(intensity=64, radius=32, noise=False)
        
        self.nrmse_ref_im_nans = self.nrmse_ref_im.copy()
        self.nrmse_ref_im_nans[::4, ::4] = np.nan
    
    
    def test_com(self):
        print(self.id())
        out = fpdp.center_of_mass(self.data, nr=None, nc=None, origin='bottom', progress_bar=False)
        out -= out.mean((1,2))[:, None, None]
        self.assertTrue(close_enough(self.sa, out, self.rtol, self.atol))
    
    def test_com_threshold_otsu(self):
        print(self.id())
        out = fpdp.center_of_mass(self.data, nr=None, nc=None, origin='bottom', thr='otsu', progress_bar=False)
        out -= out.mean((1,2))[:, None, None]
        self.assertTrue(close_enough(self.sa, out, self.rtol, self.atol))
    
    def test_com_threshold_fixed(self):
        print(self.id())
        thr = self.data.max() / 2.0
        out = fpdp.center_of_mass(self.data, nr=None, nc=None, origin='bottom', thr=thr, progress_bar=False)
        out -= out.mean((1,2))[:, None, None]
        self.assertTrue(close_enough(self.sa, out, self.rtol, self.atol))
    
    def test_com_threshold_callable(self):
        print(self.id())
        def func(image):
            return image >= image.max() / 2.0
        thr = func
        out = fpdp.center_of_mass(self.data, nr=None, nc=None, origin='bottom', thr=thr, progress_bar=False)
        out -= out.mean((1,2))[:, None, None]
        self.assertTrue(close_enough(self.sa, out, self.rtol, self.atol))
    
    def test_com_threshold_flat(self):
        print(self.id())
        data = self.data.copy()
        data[0, 0] = 1
        
        out = fpdp.center_of_mass(data, nr=None, nc=None, origin='bottom', thr='otsu', progress_bar=False)
        
        out = out.reshape([2, -1])[:, 1:]
        sa = self.sa.reshape([2, -1])[:, 1:]
        
        out -= out.mean(1)[:, None] # 1 missing point should not affect mean too badly that test fails
        self.assertTrue(close_enough(sa, out, self.rtol, self.atol))
        
    def test_com_nostats(self):
        print(self.id())
        print('Testing print_stats=False')
        out = fpdp.center_of_mass(self.data, nr=None, nc=None, print_stats=False, origin='bottom', progress_bar=False)
        out -= out.mean((1,2))[:, None, None]
        print('Done\n')
        self.assertTrue(close_enough(self.sa, out, self.rtol, self.atol))
    
    def test_com_ck(self):
        print(self.id())
        out = fpdp.center_of_mass(self.data, nr=3, nc=3, origin='bottom', progress_bar=False)
        out -= out.mean((1,2))[:, None, None]
        self.assertTrue(close_enough(self.sa, out, self.rtol, self.atol))
    
    def test_com_ck_nrnc_are_chunks(self):
        print(self.id())
        out = fpdp.center_of_mass(self.data, nr=3, nc=3, origin='bottom', nrnc_are_chunks=True, progress_bar=False)
        out -= out.mean((1,2))[:, None, None]
        self.assertTrue(close_enough(self.sa, out, self.rtol, self.atol))
    
    def test_com_ck_rebin(self):
        print(self.id())
        out = fpdp.center_of_mass(self.data, nr=3, nc=3, rebin=2, origin='bottom', progress_bar=False)
        out -= out.mean((1,2))[:, None, None]
        self.assertTrue(close_enough(self.sa, out, self.rtol, self.atol))
    
    def test_com_ck_non_parallel(self):
        print(self.id())
        out = fpdp.center_of_mass(self.data, nr=3, nc=3, parallel=False, origin='bottom', progress_bar=False)
        out -= out.mean((1,2))[:, None, None]
        self.assertTrue(close_enough(self.sa, out, self.rtol, self.atol))
    
    def test_com_ck_parallel_ncores(self):
        print(self.id())
        out = fpdp.center_of_mass(self.data, nr=3, nc=3, ncores=1, origin='bottom', progress_bar=False)
        out -= out.mean((1,2))[:, None, None]
        self.assertTrue(close_enough(self.sa, out, self.rtol, self.atol))
    
    def test_com_aperture(self):
        print(self.id())
        ap = fpdp.virtual_apertures(self.data.shape[-2:], cyx=(128,)*2, rio=(0, int(self.radius*1.5)), sigma=0, aaf=1)
        out = fpdp.center_of_mass(self.data, nr=3, nc=3, ncores=1, origin='bottom', progress_bar=False, aperture=ap)
        out -= out.mean((1,2))[:, None, None]
        self.assertTrue(close_enough(self.sa, out, self.rtol, self.atol))
    
    def test_com_aperture_rebin(self):
        print(self.id())
        ap = fpdp.virtual_apertures(self.data.shape[-2:], cyx=(128,)*2, rio=(0, int(self.radius*1.5)), sigma=0, aaf=1)
        out = fpdp.center_of_mass(self.data, nr=3, nc=3, ncores=1, origin='bottom', progress_bar=False, aperture=ap, rebin=2)
        out -= out.mean((1,2))[:, None, None]
        self.assertTrue(close_enough(self.sa, out, self.rtol, self.atol))
    
    def test_com_pre_func(self):
        print(self.id())
        def func(image):
            return image * 2
        
        out = fpdp.center_of_mass(self.data, nr=3, nc=3, ncores=1, origin='bottom', progress_bar=False, pre_func=func)
        out -= out.mean((1,2))[:, None, None]
        self.assertTrue(close_enough(self.sa, out, self.rtol, self.atol))
    
    
    def test_phase_2d(self):
        print(self.id())
        r1 = fpdp.phase_correlation(self.data, nr=None, nc=None, cyx=(128,)*2,
                                    crop_r=64, spf=100, origin='bottom', progress_bar=False)
        shift_yx, shift_err, shift_difp, ref = r1
        shift_yx -= shift_yx.mean((1,2))[:, None, None]
        self.assertTrue(close_enough(self.sa, shift_yx, self.rtol, self.atol))
    
    def test_phase_funcs(self):
        print(self.id())
        def pre_func(image):
            return image * 2.0
        def post_func(image):
            return image * 0.5
        
        r1 = fpdp.phase_correlation(self.data, nr=None, nc=None, cyx=(128,)*2,
                                    crop_r=64, spf=100, origin='bottom', progress_bar=False,
                                    pre_func=pre_func, post_func=post_func)
        shift_yx, shift_err, shift_difp, ref = r1
        shift_yx -= shift_yx.mean((1,2))[:, None, None]
        self.assertTrue(close_enough(self.sa, shift_yx, self.rtol, self.atol))
    
    def test_phase_truncate(self):
        print(self.id())
        r1 = fpdp.phase_correlation(self.data, nr=None, nc=None, cyx=(128,)*2,
                                    crop_r=64, spf=100, origin='bottom', progress_bar=False,
                                    truncate=64, sigma=4)
        shift_yx, shift_err, shift_difp, ref = r1
        shift_yx -= shift_yx.mean((1,2))[:, None, None]
        self.assertTrue(close_enough(self.sa, shift_yx, self.rtol, self.atol))
    
    def test_phase_2d_sigma0(self):
        print(self.id())
        r1 = fpdp.phase_correlation(self.data, nr=None, nc=None, cyx=(128,)*2,
                                    crop_r=64, spf=100, origin='bottom', sigma=0,
                                    progress_bar=False)
        shift_yx, shift_err, shift_difp, ref = r1
        shift_yx -= shift_yx.mean((1,2))[:, None, None]
        self.assertTrue(close_enough(self.sa, shift_yx, self.rtol, self.atol))
    
    def test_phase_2d_sigma_negative(self):
        print(self.id())
        r1 = fpdp.phase_correlation(self.data, nr=None, nc=None, cyx=(128,)*2,
                                    crop_r=64, spf=100, origin='bottom', sigma=-1,
                                    progress_bar=False)
        shift_yx, shift_err, shift_difp, ref = r1
        shift_yx -= shift_yx.mean((1,2))[:, None, None]
        
        self.assertTrue(close_enough(self.sa, shift_yx, self.rtol, self.atol))
    
    def test_phase_2d_clip(self):
        print(self.id())
        r1 = fpdp.phase_correlation(self.data, nr=None, nc=None, cyx=(128,)*2,
                                    crop_r=64, spf=100, origin='bottom',
                                    der_clip_fraction=0.2, der_clip_max_pct=99,
                                    progress_bar=False)
        shift_yx, shift_err, shift_difp, ref = r1
        shift_yx -= shift_yx.mean((1,2))[:, None, None]
        self.assertTrue(close_enough(self.sa, shift_yx, self.rtol, self.atol))
    
    
    def test_phase_2d_no_print(self):
        print(self.id())
        print('Testing print_stats=False')
        r1 = fpdp.phase_correlation(self.data, nr=None, nc=None, cyx=(128,)*2,
                                    crop_r=64, spf=100, print_stats=False, origin='bottom',
                                    progress_bar=False)
        shift_yx, shift_err, shift_difp, ref = r1
        shift_yx -= shift_yx.mean((1,2))[:, None, None]
        print('Done\n')
        self.assertTrue(close_enough(self.sa, shift_yx, self.rtol, self.atol))
        
    def test_phase_2d_ck(self):
        print(self.id())
        r1 = fpdp.phase_correlation(self.data, nr=3, nc=3, cyx=(128,)*2,
                                    crop_r=64, spf=100, origin='bottom',
                                    progress_bar=False)
        shift_yx, shift_err, shift_difp, ref = r1
        shift_yx -= shift_yx.mean((1,2))[:, None, None]
        self.assertTrue(close_enough(self.sa, shift_yx, self.rtol, self.atol))
        
    def test_phase_2d_ck_nrnc_are_chunks(self):
        print(self.id())
        r1 = fpdp.phase_correlation(self.data, nr=3, nc=3, cyx=(128,)*2,
                                    crop_r=64, spf=100, origin='bottom',
                                    nrnc_are_chunks=True, progress_bar=False)
        shift_yx, shift_err, shift_difp, ref = r1
        shift_yx -= shift_yx.mean((1,2))[:, None, None]
        self.assertTrue(close_enough(self.sa, shift_yx, self.rtol, self.atol))
    
    def test_phase_2d_ck_rebin(self):
        print(self.id())
        r1 = fpdp.phase_correlation(self.data, nr=3, nc=3, cyx=(128,)*2,
                                    crop_r=64, spf=100, rebin=2, origin='bottom',
                                    progress_bar=False)
        shift_yx, shift_err, shift_difp, ref = r1
        shift_yx -= shift_yx.mean((1,2))[:, None, None]
        self.assertTrue(close_enough(self.sa, shift_yx, self.rtol, self.atol))
    
    def test_phase_2d_ck_rebin_odd(self):
        print(self.id())
        # odd rebin
        r1 = fpdp.phase_correlation(self.data, nr=3, nc=3, cyx=(128,)*2,
                                    crop_r=64, spf=100, rebin=3, origin='bottom',
                                    progress_bar=False)
        shift_yx, shift_err, shift_difp, ref = r1
        shift_yx -= shift_yx.mean((1,2))[:, None, None]
        self.assertTrue(close_enough(self.sa, shift_yx, self.rtol, self.atol))
    
    def test_phase_2d_ck_ref_im(self):
        print(self.id())
        # odd rebin
        r1 = fpdp.phase_correlation(self.data, nr=3, nc=3, cyx=(128,)*2,
                                    crop_r=64, spf=100, origin='bottom',
                                    progress_bar=False, ref_im=self.data[0,0])
        shift_yx, shift_err, shift_difp, ref = r1
        shift_yx -= shift_yx.mean((1,2))[:, None, None]
        self.assertTrue(close_enough(self.sa, shift_yx, self.rtol, self.atol))
    
    def test_phase_2d_ck_non_parallel(self):
        print(self.id())
        r1 = fpdp.phase_correlation(self.data, nr=3, nc=3, cyx=(128,)*2,
                                    crop_r=64, spf=100, parallel=False, origin='bottom',
                                    progress_bar=False)
        shift_yx, shift_err, shift_difp, ref = r1
        shift_yx -= shift_yx.mean((1,2))[:, None, None]
        self.assertTrue(close_enough(self.sa, shift_yx, self.rtol, self.atol))
    
    def test_phase_2d_ck_parallel_ncores(self):
        print(self.id())
        r1 = fpdp.phase_correlation(self.data, nr=3, nc=3, cyx=(128,)*2,
                                    crop_r=64, spf=100, ncores=1, origin='bottom',
                                    progress_bar=False)
        shift_yx, shift_err, shift_difp, ref = r1
        shift_yx -= shift_yx.mean((1,2))[:, None, None]
        self.assertTrue(close_enough(self.sa, shift_yx, self.rtol, self.atol))    
        
    def test_phase_1d(self):
        print(self.id())
        r2 = fpdp.phase_correlation(self.data, nr=None, nc=None, cyx=(128,)*2,
                                    crop_r=64, mode='1d', spf=100, origin='bottom',
                                    progress_bar=False)
        shift_yx, shift_err, shift_difp, ref = r2
        shift_yx -= shift_yx.mean((1,2))[:, None, None]
        self.assertTrue(close_enough(self.sa, shift_yx, self.rtol, self.atol))
        
    def test_phase_1d_ck(self):
        print(self.id())
        r2 = fpdp.phase_correlation(self.data, nr=3, nc=3, cyx=(128,)*2,
                                    crop_r=64, mode='1d', spf=100, origin='bottom',
                                    progress_bar=False)
        shift_yx, shift_err, shift_difp, ref = r2
        shift_yx -= shift_yx.mean((1,2))[:, None, None]
        self.assertTrue(close_enough(self.sa, shift_yx, self.rtol, self.atol))
    
    def test_sum_im(self):
        print(self.id())
        sum_im = fpdp.sum_im(self.data, nr=3, nc=3, progress_bar=False)
        
        mask = np.ones(self.data.shape[-2:])
        sum_im = fpdp.sum_im(self.data, nr=3, nc=3, progress_bar=False, mask=mask)
    
    
    def test_sum_dif(self):
        print(self.id())
        dif_im = fpdp.sum_dif(self.data, nr=3, nc=3, progress_bar=False)
        
        mask = np.ones(self.data.shape[:-2])
        dif_im = fpdp.sum_dif(self.data, nr=3, nc=3, progress_bar=False, mask=mask)
    
    def test_sum_diff_dyx(self):
        print(self.id())
        
        sa = np.asarray(fpd.synthetic_data.shift_array(scan_len=3, shift_min=-16.0, shift_max=16.0))
        sa[0] += 2
        sa[1] += 4
        
        data = fpd.synthetic_data.shift_images(sa, self.im, noise=False, origin='top')
        
        sum_diff_not_aligned = fpdp.sum_dif(data, None, None)
        sum_diff_aligned_nans = fpdp.sum_dif(data, None, None, dyx=-sa)
        sum_dif_aligned_zero_filled_pix = fpdp.sum_dif(data, None, None, dyx=-sa, dyx_mode='pixel', fill_value=0) / (sa.size / 2)
        sum_dif_aligned_zero_filled_linear = fpdp.sum_dif(data, None, None, dyx=-sa, dyx_mode='linear', fill_value=0) / (sa.size / 2)
        sum_dif_aligned_zero_filled_fourier = fpdp.sum_dif(data, None, None, dyx=-sa, dyx_mode='fourier', fill_value=0) / (sa.size / 2)
        
        '''
        # plots
        plt.matshow(self.im); plt.colorbar()
        plt.matshow(sum_diff_not_aligned); plt.colorbar()
        plt.matshow(sum_diff_aligned_nans); plt.colorbar()
        plt.matshow(sum_dif_aligned_zero_filled_linear); plt.colorbar()
        plt.matshow(sum_dif_aligned_zero_filled_linear - self.im); plt.colorbar()
        '''
        assert np.allclose(sum_dif_aligned_zero_filled_linear, self.im)
        assert np.allclose(sum_dif_aligned_zero_filled_fourier, self.im)
    
    
    def test_sum_ax(self):
        print(self.id())
        for axis in [0, 1, -1]:
            ims = fpdp.sum_ax(self.data, axis=axis, n=3, progress_bar=False)
            ims_np = self.data.sum(axis)
            self.assertTrue(close_enough(ims, ims_np, self.rtol, self.atol))
    
    #def test_sum_ax_pbar(self):
        #print(self.id())
        #ims = fpdp.sum_ax(self.data, axis=0, n=3, progress_bar=False)
    
    #-----
    
    def test_synthetic_aperture(self):
        print(self.id())
        try:
            aps = fpdp.synthetic_aperture((64,)*2, (32,)*2, (0, 16), 0.5, ds_method='banana')
        except:
            pass
        else:
            raise Exception("expected exception here, but it didn't happen")
        aps = fpdp.synthetic_aperture((64,)*2, (32,)*2, (0, 16), 0.5, ds_method='interp')
        aps = fpdp.synthetic_aperture((64,)*2, (32,)*2, (0, 16), 0.5, ds_method='rebin')
        
    def test_synthetic_images(self):
        print(self.id())
        aps = fpdp.synthetic_aperture(self.data.shape[-2:], (128,)*2, np.linspace(16, 192, 4))
        synth_ims = fpdp.synthetic_images(self.data, nr=3, nc=3, apertures=aps, progress_bar=False)
    
    def test_synthetic_images_rebin(self):
        print(self.id())
        aps = fpdp.synthetic_aperture(self.data.shape[-2:], (128,)*2, np.linspace(16, 192, 4))
        synth_ims = fpdp.synthetic_images(self.data, nr=3, nc=3, apertures=aps, rebin=2, progress_bar=False)
        
        synth_ims = fpdp.synthetic_images(self.data, nr=3, nc=3, apertures=aps, rebin=3, progress_bar=False)
    
    def test_synthetic_images_nrnc_are_chunks(self):
        print(self.id())
        aps = fpdp.synthetic_aperture(self.data.shape[-2:], (128,)*2, np.linspace(16, 192, 4))
        synth_ims = fpdp.synthetic_images(self.data, nr=3, nc=3, apertures=aps, nrnc_are_chunks=True, progress_bar=False)
    
    #-----
    
    def test_virtual_apertures(self):
        print(self.id())
        try:
            aps = fpdp.virtual_apertures((64,)*2, (32,)*2, (0, 16), 0.5, ds_method='banana')
        except:
            pass
        else:
            raise Exception("expected exception here, but it didn't happen")
        aps = fpdp.virtual_apertures((64,)*2, (32,)*2, (0, 16), 0.5, ds_method='interp')
        aps = fpdp.virtual_apertures((64,)*2, (32,)*2, (0, 16), 0.5, ds_method='rebin')
    
    def test_virtual_apertures_images_single_multi(self):
        print(self.id())
        rs = np.linspace(32, 192, 9)
        aps_multi = fpdp.virtual_apertures(self.data.shape[-2:], (128,)*2, rs)
        aps_single = fpdp.virtual_apertures(self.data.shape[-2:], (128,)*2, rs[:2])
        
        assert aps_multi.ndim == 3
        assert aps_single.ndim == 2
        assert np.allclose(aps_single, aps_multi[0])
        
        virt_ims_multi = fpdp.virtual_images(self.data, nr=3, nc=3, apertures=aps_multi, progress_bar=False)
        virt_ims_single = fpdp.virtual_images(self.data, nr=3, nc=3, apertures=aps_multi[0], progress_bar=False)
        
        assert virt_ims_multi.ndim == 3
        assert virt_ims_single.ndim == 2
        assert np.allclose(virt_ims_single, virt_ims_multi[0])
        
    def test_virtual_images(self):
        print(self.id())
        aps = fpdp.virtual_apertures(self.data.shape[-2:], (128,)*2, np.linspace(16, 192, 4))
        virt_ims = fpdp.virtual_images(self.data, nr=3, nc=3, apertures=aps, progress_bar=False)
    
    def test_virtual_images_rebin(self):
        print(self.id())
        aps = fpdp.virtual_apertures(self.data.shape[-2:], (128,)*2, np.linspace(16, 192, 4))
        virt_ims = fpdp.virtual_images(self.data, nr=3, nc=3, apertures=aps, rebin=2, progress_bar=False)
        
        virt_ims = fpdp.virtual_images(self.data, nr=3, nc=3, apertures=aps, rebin=3, progress_bar=False)
    
    def test_virtual_images_nrnc_are_chunks(self):
        print(self.id())
        aps = fpdp.virtual_apertures(self.data.shape[-2:], (128,)*2, np.linspace(16, 192, 4))
        virt_ims = fpdp.virtual_images(self.data, nr=3, nc=3, apertures=aps, nrnc_are_chunks=True, progress_bar=False)
    
    def test_virtual_images_dyx(self):
        print(self.id())
        from fpd.utils import Timer
        
        intensity = 128
        im = fpd.synthetic_data.disk_image(intensity=intensity, radius=self.radius, 
                                                size=256, upscale=8, dtype='uint8', sigma=0)
        #plt.matshow(im)
        cyx = np.asarray(sp.ndimage.center_of_mass(im))
        scan_len = 5
        rtol = 1e-02
        atol = 400
        
        # two apertures, one inside and one outside unshifted disc
        pad = 5
        aps = fpdp.virtual_apertures(im.shape, cyx, np.array([[0, self.radius - pad], [self.radius + pad, self.radius+8+20]]), sigma=0)
        #for imi in aps: plt.matshow(imi)
        
        ## no shifts
        data0 = fpd.synthetic_data.shift_images(np.zeros((2, scan_len, scan_len)), im, noise=False, origin='top')
        virt_ims0 = fpdp.virtual_images(data0, nr=None, nc=None, apertures=aps, progress_bar=False, debug=False)
        #for imi in virt_ims0: plt.matshow(imi)
        
        # set up sp there are no counts in outer aperture
        assert np.allclose(virt_ims0[1]*0, virt_ims0[1], rtol=rtol)
        
        ## shifted data
        # loop over zero, negative and positive net shifts to check that efficient indexing works
        for sa_bias in [-10, 0, 10]:
            print('sa_bias: ', sa_bias)
            
            # shifts here slightly off integer so that images are very similar, but pixel and sub-pixel shift handling is tested 
            sa = np.asarray(fpd.synthetic_data.shift_array(scan_len=scan_len, shift_min=-8.1, shift_max=8.1))
            sa -= sa_bias
            data = fpd.synthetic_data.shift_images(sa, im, noise=False, origin='top')
            #plt.matshow(fpdp.sum_dif(data, None, None))
            
            for dyx_mode in ['linear', 'pixel', 'fourier']:
                with Timer(dyx_mode):
                    virt_ims_aligned = fpdp.virtual_images(data, nr=None, nc=None, apertures=aps, progress_bar=False, dyx=-sa,
                                                        dyx_mode=dyx_mode, debug=True)
                    #for imi in virt_ims_aligned: plt.matshow(imi)
                    
                    # same outside and in (when comparing aligned data with non-shifted data)
                    assert np.allclose(virt_ims_aligned[1], virt_ims0[1], rtol=rtol, atol=atol)
                    assert np.allclose(virt_ims_aligned[0], virt_ims0[0], rtol=rtol, atol=atol)
            plt.close('all')
        
        
        ### with padding
        print('With padding')
        for sa_bias in [-10, 0, 10]:
            print('sa_bias: ', sa_bias)
            # shifts here slightly off integer so that images are very similar, but pixel and sub-pixel shift handling is tested
            # and are large enough to require padding (for aperture) but not so large that disk data is missing
            sa = np.asarray(fpd.synthetic_data.shift_array(scan_len=scan_len, shift_min=-80.1, shift_max=80.1))
            sa -= sa_bias
            data = fpd.synthetic_data.shift_images(sa, im, noise=False, origin='top')
            #data = data.astype(float)
            #plt.matshow(fpdp.sum_dif(data, None, None))
            
            for dyx_mode in ['linear', 'pixel']:#, 'fourier']:
                with Timer(dyx_mode):
                    virt_ims_aligned = fpdp.virtual_images(data, nr=None, nc=None, apertures=aps, progress_bar=False,
                                                           dyx=-sa, dyx_mode=dyx_mode, fill_value=0, debug=True)
                    #for imi in virt_ims_aligned: plt.matshow(imi)
                    
                    # same outside and in (when comparing aligned data with non-shifted data)
                    assert np.allclose(virt_ims_aligned[1], virt_ims0[1], rtol=rtol, atol=atol)
                    assert np.allclose(virt_ims_aligned[0], virt_ims0[0], rtol=rtol, atol=atol)
            plt.close('all')
    
    
    def test_synthetic_images_virtual_images_speed(self):
        print(self.id())
        
        aps = fpdp.virtual_apertures(self.data.shape[-2:], (128,)*2, np.linspace(16, 192, 4))
        
        from fpd.utils import Timer
        with Timer('synthetic images'):
            synth_ims = fpdp.synthetic_images(self.data, nr=None, nc=None, apertures=aps, progress_bar=False)
        with Timer('virtual images'):
            virt_ims = fpdp.virtual_images(self.data, nr=None, nc=None, apertures=aps, progress_bar=False)
    
    #-----
    
    def test_sum_im_nrnc_are_chunks(self):
        print(self.id())
        sum_im = fpdp.sum_im(self.data, nr=3, nc=3, nrnc_are_chunks=True, progress_bar=False)
    
    def test_sum_dif_nrnc_are_chunks(self):
        print(self.id())
        dif_im = fpdp.sum_dif(self.data, nr=3, nc=3, nrnc_are_chunks=True, progress_bar=False)
    
    def test_radial_profile(self):
        print(self.id())
        cyx = (128,)*2
        im_shape = (256,)*2
        y, x = np.indices(im_shape)
        r = np.hypot(y - cyx[0], x - cyx[1])
        data = np.dstack((r**0.5, r, r**2))
        data = np.rollaxis(data, 2, 0)
        
        r_pix, radial_mean = fpdp.radial_profile(data, cyx, plot=True, spf=2)
        plt.close('all')
        
        r_pix, radial_mean = fpdp.radial_profile(data, cyx, plot=True, r_nm_pp=1)
        plt.close('all')
        
        mask = np.ones(im_shape, dtype=np.int)
        r_pix, radial_mean = fpdp.radial_profile(data, cyx, plot=True, spf=2, mask=mask)
        plt.close('all')
    
    def test_radial_profiles(self):
        print(self.id())
        from fpd.utils import Timer
        
        plot = False
        
        intensity = 128
        im = fpd.synthetic_data.disk_image(intensity=intensity, radius=16, 
                                                size=128, upscale=8, dtype='uint8', sigma=0)
        
        # data with shifts
        sa = np.asarray(fpd.synthetic_data.shift_array(scan_len=5, shift_min=-12, shift_max=12))
        data = fpd.synthetic_data.shift_images(sa, im, noise=False, origin='top')
        cyx = fpdp.center_of_mass(data, 16, 16, progress_bar=False, print_stats=False)
        
        # no shifts
        data0 = fpd.synthetic_data.shift_images(sa*0, im, noise=False, origin='top')
        cyx0 = np.asarray(sp.ndimage.center_of_mass(data0[0, 0]))
        
        if plot:
            plt.matshow(data0.sum((0, 1)))
            plt.matshow(data.sum((0, 1)))
        
        # radial profiles
        for r_lim_mode in ['corner_max', 'corner_min', 'edge_max', 'edge_min']:
            # drift corrected through cyx
            r, rm = fpdp.radial_profiles(data, 16, 16, cyx, r_lim_mode=r_lim_mode,
                                         progress_bar=False, print_stats=False)
            # drift corrected through dyx
            cyx2 = cyx[:, 0, 0]
            dyx2 = cyx-cyx[:, 0, 0][..., None, None]
            r2, rm2 = fpdp.radial_profiles(data, 16, 16, cyx2, dyx2, r_lim_mode=r_lim_mode,
                                           progress_bar=False, print_stats=False)
            # no drift correction ('wrong' result)
            rw, rmw = fpdp.radial_profiles(data, 16, 16, cyx0, r_lim_mode=r_lim_mode,
                                           progress_bar=False, print_stats=False)
            # no drift, no correction
            r0, rm0 = fpdp.radial_profiles(data0, 16, 16, cyx0, r_lim_mode=r_lim_mode,
                                           progress_bar=False, print_stats=False)
            
            rmm = rm.mean((0, 1))
            rmm2 = rm2.mean((0, 1))
            rmmw = rmw.mean((0, 1))
            rmm0 = rm0.mean((0, 1))
            
            if plot:
                plt.figure()
                plt.plot(r0, rmm0, 's')
                plt.plot(rw, rmmw, '--')
                plt.plot(r, rmm, 'k-x')
            
            # check results match
            n = min(len(r), len(r0))
            assert np.allclose(r[1:n], r2[1:n])
            assert np.allclose(rmm[1:n], rmm2[1:n])
            
            assert np.allclose(r[1:n], r0[1:n])
            assert np.allclose(rmm[1:n], rmm0[1:n])
    
    
    def test_condition_nrnc_if_chunked(self):
        print(self.id())
        rtn = fpdp._condition_nrnc_if_chunked(data=None, nr=None, nc=None, print_enabled=True)
        assert (None, None) == rtn
        
        class d:
            def __init__(self):
                self.chunks = (5,)*4
        d = d()
        rtn = fpdp._condition_nrnc_if_chunked(data=d, nr=1, nc=1, print_enabled=True)
        assert d.chunks[:2] == rtn
    
    
    # nrmse
    def test_nrmse_1image(self):
        print(self.id())
        rtn = fpdp.nrmse(self.nrmse_ref_im, self.nrmse_ref_im)
        assert (rtn==0).all()
    
    def test_nrmse_1Dimages(self):
        print(self.id())
        rtn = fpdp.nrmse(self.nrmse_ref_im, self.nrmse_ref_im[None, ...])
        assert (rtn==0).all()
    
    def test_nrmse_1image(self):
        print(self.id())
        rtn = fpdp.nrmse(self.nrmse_ref_im, self.nrmse_ref_im[None, None, ...])
        assert (rtn==0).all()
    
    def test_nrmse_nan(self):
        print(self.id())
        im = self.nrmse_ref_im_nans
        
        rtn1 = fpdp.nrmse(im, im, allow_nans=True)
        assert (rtn1==0).all()
        
        # , allow_nans=False should be false by default
        rtn2 = fpdp.nrmse(im, im)
        assert np.isnan(rtn2) == False
    
    
    # matching images
    def test_matching(self):
        print(self.id())
        from fpd.synthetic_data import disk_image, shift_array, shift_images
        
        # generate synthetic data
        disc = disk_image(radius=32, intensity=64)
        shift_array = shift_array(6, shift_min=-1, shift_max=1)
        
        # set shifts on diagonal to zero 
        diag_inds = [np.diag(x) for x in np.indices(shift_array[0].shape)]
        shift_array[0][tuple(diag_inds)] = 0
        shift_array[1][tuple(diag_inds)] = 0
        
        # shift images
        images = shift_images(shift_array, disc, noise=False)
        aperture = fpdp.virtual_apertures(images.shape[-2:], cyx=(128,)*2, rio=(0, 48), sigma=0, aaf=1)
        
        matching = fpdp.find_matching_images(images, aperture, plot=True, progress_bar=False)
        assert (matching.ims_best.std(0) == 0).all()
        assert (matching.ims_common.std(0) == 0).all()
        
        # test no plotting
        matching = fpdp.find_matching_images(images, aperture, plot=False, progress_bar=False)
        
        # test no aperture
        matching = fpdp.find_matching_images(images, aperture=None, plot=False, progress_bar=False)
    
    
    # find_circ_centre
    def test_find_circ_centre(self):
        print(self.id())
        r = 32
        image = fpd.synthetic_data.disk_image(radius=r, intensity=64, sigma=4)
        image = image + np.random.rand(*image.shape)*4

        mask = fpd.synthetic_data.disk_image(radius=r+5, intensity=5, sigma=0) > 0.5

        cyx, cr = fpdp.find_circ_centre(image, sigma=6, rmms=(2, int(image.shape[0]/2.0), 1), plot=False)
        cyx, cr = fpdp.find_circ_centre(image, sigma=6, rmms=(r-4, r+4, 1), plot=True)
        cyx, cr = fpdp.find_circ_centre(image, sigma=6, rmms=(r-4, r+4, 1), plot=False, pct=90)
        cyx, cr = fpdp.find_circ_centre(image, sigma=6, rmms=(r-4, r+4, 1), plot=False, mask=mask)
        cyx, cr = fpdp.find_circ_centre(image, sigma=6, rmms=(r-4, r+4, 1), plot=False, mask=mask, pct=90)
        cyx, cr = fpdp.find_circ_centre(image, sigma=6, rmms=(r-4, r+4, 1), plot=False, mask=mask, pct=90, spf=2)
        plt.close('all')
        
        cyx, cr = fpdp.find_circ_centre(image, sigma=6, rmms=(r-4, r+4, 1), plot=True, subpix=False)
        cyx, cr = fpdp.find_circ_centre(image, sigma=6, rmms=(r-4, r+4, 1), plot=False, fit_hw=4)
        
        # multiple discs
        from fpd.synthetic_data import shift_im
        im2 = fpd.synthetic_data.disk_image(intensity=32, radius=24, sigma=2, size=256, noise=False)
        shifted_im = shift_im(im2, dyx=(64, 64), noise=False)
        im_multidiscs = image + shifted_im
        #plt.matshow(im_multidiscs)
        cyx, cr = fpdp.find_circ_centre(im_multidiscs, sigma=6, rmms=(20, 36, 1), plot=True, max_n=2)
        #print(cyx, cr)
        cyx, cr = fpdp.find_circ_centre(im_multidiscs, sigma=6, rmms=(20, 36, 1), plot=False, max_n=2)
        #print(cyx, cr)
        plt.close('all')
    
    
    # test map_image_function
    def test_map_image_function(self):
        print(self.id())
        func = sp.ndimage.center_of_mass
        out = fpdp.map_image_function(self.data, nr=None, nc=None, func=func, progress_bar=False)
        assert out.ndim == 3
        
    def test_map_image_function_params(self):
        print(self.id())
        val = 2
        out = fpdp.map_image_function(self.data, nr=None, nc=None, func=params_f, params={'v' : val},
                                      progress_bar=False)
        assert np.isclose(val, out).all()
        assert out.ndim == 2
    
    def test_map_image_function_mapped_params(self):
        print(self.id())
        mvals = np.arange(np.prod(self.data.shape[:-2])).reshape(self.data.shape[:-2])
        val = 2
        
        def f(image, v, w):
            return w
        out = fpdp.map_image_function(self.data, nr=None, nc=None, func=f, params={'v' : val},
                                      mapped_params={'w' : mvals}, progress_bar=False)
        assert np.isclose(mvals, out).all()
        assert out.ndim == 2
        
    def test_map_image_function_nonuniform(self):
        print(self.id())
        out = fpdp.map_image_function(self.data, nr=None, nc=None, func=nonuniform_f,
                                      progress_bar=False)
        assert out.ndim == 2
    
    def test_map_image_function_print(self):
        print(self.id())
        func = sp.ndimage.center_of_mass
        print('Testing print_stats=False')
        out = fpdp.map_image_function(self.data, nr=None, nc=None, print_stats=False, func=func,
                                      progress_bar=False)
        print('Done\n')
    
    def test_map_image_function_ck(self):
        print(self.id())
        func = sp.ndimage.center_of_mass
        out = fpdp.map_image_function(self.data, nr=3, nc=3, func=func, progress_bar=False)
    
    def test_map_image_function_ck_nrnc_are_chunks(self):
        print(self.id())
        func = sp.ndimage.center_of_mass
        out = fpdp.map_image_function(self.data, nr=3, nc=3, func=func, nrnc_are_chunks=True,
                                      progress_bar=False)
    
    def test_map_image_function_ck_rebin(self):
        print(self.id())
        func = sp.ndimage.center_of_mass
        out = fpdp.map_image_function(self.data, nr=3, nc=3, rebin=2, func=func, progress_bar=False)
    
    def test_map_image_function_ck_non_parallel(self):
        print(self.id())
        func = sp.ndimage.center_of_mass
        out = fpdp.map_image_function(self.data, nr=3, nc=3, parallel=False, func=func,
                                      progress_bar=False)
    
    def test_map_image_function_ck_parallel_ncores(self):
        print(self.id())
        func = sp.ndimage.center_of_mass
        out = fpdp.map_image_function(self.data, nr=3, nc=3, ncores=1, func=func,
                                      progress_bar=False)
    
    def test_map_image_function_ck_parallel_process(self):
        print(self.id())
        func = sp.ndimage.center_of_mass
        out = fpdp.map_image_function(self.data, nr=3, nc=3, func=func,
                                      progress_bar=False, parallel_mode='process')
        
    
    def test_VirtualAnnularImages(self):
        print(self.id())
        V = fpdp.VirtualAnnularImages(self.data, nr=3, nc=3, cyx=(128,)*2, progress_bar=False)
        
        im = V.annular_slice(10, 20)
        
        V.plot()
        plt.close('all')
        
        V.plot(r1=20, r2=30, norm='log', nav_im=np.ones(self.data.shape[-2:]), alpha=0.2, cmap='gray', pct=1)
        plt.close('all')
        
        V.plot(mradpp=2.0)
        plt.close('all')
    
    def test_VirtualAnnularImages_acc(self):
        print(self.id())
        
        ims = np.ones((32,) * 4)
        for spf in [1, 2]:
            v = fpdp.VirtualAnnularImages(ims, 16, 16, cyx=(16, 16), spf=spf, progress_bar=False)

            r = np.arange(16 - 4)
            dr = 1
            i = np.array([v.annular_slice(r1, r1+dr) for r1 in r]).mean((-2, -1))
            m = i / (2 * np.pi * (r + dr/2)) / dr
            
            assert np.allclose(m, 1)
        
    def test_VirtualAnnularImages_save_load(self):
        print(self.id())
        V = fpdp.VirtualAnnularImages(self.data, nr=3, nc=3, cyx=(128,)*2, progress_bar=False)
        fn_auto = V.save_data()
        fn = V.save_data('test')
        
        V = fpdp.VirtualAnnularImages(fn)
        V.plot()
        plt.close('all')
        
        d = dict(np.load(fn))
        V = fpdp.VirtualAnnularImages(d)
        V.plot()
        plt.close('all')
        
        os.remove(fn)
        os.remove(fn_auto)
    
    def test_VirtualAnnularImages_events(self):
        print(self.id())
        V = fpdp.VirtualAnnularImages(self.data, nr=3, nc=3, cyx=(128,)*2, progress_bar=False)
        V.plot()
        
        V._update_r_from_slider(None)
        V._update_rc_from_slider(None)
        V._update_plot_r_from_val()
    
    def test_VirtualAnnularImages_dyx(self):
        print(self.id())
        
        im = fpd.synthetic_data.disk_image(intensity=128, radius=self.radius, 
                                                size=256, upscale=8, dtype='uint8', sigma=0)
        sa = np.asarray(fpd.synthetic_data.shift_array(scan_len=9, shift_min=-16.0, shift_max=16.0))
        data = fpd.synthetic_data.shift_images(sa, im, noise=False, origin='top')
        
        pad = 2
        V0 = fpdp.VirtualAnnularImages(data, nr=3, nc=3, cyx=(128,)*2, dyx=None, progress_bar=False)
        im_out0 = V0.annular_slice(self.radius + pad, 50)
        dyx = -sa
        for dyx_mode in ['linear', 'pixel', 'fourier']:
            V = fpdp.VirtualAnnularImages(data, nr=3, nc=3, cyx=(128,)*2, dyx=dyx, dyx_mode=dyx_mode, progress_bar=False)
            im_out = V.annular_slice(self.radius + pad, 50)
            #im_in = V.annular_slice(0, self.radius - pad)
            
            #plt.matshow(im_out0); plt.colorbar()
            #plt.matshow(im_out); plt.colorbar()
            self.assertFalse(np.allclose(im_out0, im_out0 * 0))
            self.assertTrue(np.allclose(im_out, im_out * 0))
    
    
    # test make_ref_im
    def test_make_ref_im(self):
        print(self.id())
        image = fpd.synthetic_data.disk_image(radius=32, intensity=64)
        cyx, cr = fpdp.find_circ_centre(image, sigma=6, rmms=(2, int(image.shape[0]/2.0), 1), plot=False)
        edge_sigma = fpdp.disc_edge_properties(image, sigma=2, cyx=cyx, r=cr, plot=False).sigma_wt_avg
        aperture = fpdp.virtual_apertures(image.shape[-2:], cyx=cyx, rio=(0, cr+32), sigma=0, aaf=1)
        
        t1 = fpdp.make_ref_im(image, edge_sigma, plot=False)
        t1 = fpdp.make_ref_im(image, edge_sigma, plot=True)
        t2 = fpdp.make_ref_im(image, edge_sigma, aperture, plot=False)
        t3 = fpdp.make_ref_im(image, edge_sigma, aperture, bin_opening=4, bin_closing=4, plot=False)
        t4 = fpdp.make_ref_im(image, edge_sigma, aperture, bin_opening=4, bin_closing=4, crop_pad=True, plot=False)
        t5 = fpdp.make_ref_im(image, edge_sigma, aperture, bin_opening=4, bin_closing=4, crop_pad=True, plot=False, threshold=32)
    
    def test_rot_vector_accuracy(self):
        print(self.id())
        yx = np.array([0, 1])
        yx_rot_real = np.array([1, 0])
        yx_rot = fpdp.rotate_vector(yx, theta=90, axis=0)
        assert np.allclose(yx_rot_real.flat, yx_rot.flat)
    
    def test_rot_vector_dims(self):
        print(self.id())
        yx = np.array([0, 1])
        yx_rot = fpdp.rotate_vector(yx, theta=90, axis=0)
        yx_rot = fpdp.rotate_vector(yx[..., None], theta=90, axis=0)
        yx_rot = fpdp.rotate_vector(yx[None, ..., None], theta=90, axis=1)
    
    def test_check_libs(self):
        print(self.id())
        fpdp._check_libs()
        
    def test_cpu_thread_lib_check(self):
        print(self.id())
        multi_thread = fpdp.cpu_thread_lib_check(n=2000)
    
    def test_print_shift_stats(self):
        print(self.id())
        print('No nans\n-------')
        scanY, scanX = 10, 3
        a = np.zeros([2, scanY, scanX])
        a[0].flat = np.random.normal(loc=5, scale=1.0, size=scanY*scanX)
        a[1].flat = np.random.normal(loc=15, scale=2.0, size=scanY*scanX)
        fpdp._print_shift_stats(a)
        
        print('With nans\n---------')
        a[:, 0, 2] = np.nan
        a[:, 5, 0] = np.nan
        fpdp._print_shift_stats(a)
    
    def test_rebinA(self):
        print(self.id())
        a = np.random.rand(6, 4, 2).astype('u2')
        
        b = fpdp.rebinA(a, *[i//2 for i in a.shape])
        #print(b.dtype)
        #uint32
        
        b = fpdp.rebinA(a, *[i//2 for i in a.shape], bitdepth=12)
        #print(b.dtype)
        #uint16
        assert b.dtype is np.dtype(np.uint16)
        
        b = fpdp.rebinA(a, *[i//2 for i in a.shape], dtype='uint64')
        #print(b.dtype)
        #uint64
        assert b.dtype is np.dtype(np.uint64)
        
    def test_shift_images(self):
        print(self.id())
        
        from fpd.utils import Timer
        with Timer('sub-pixel mode with sub-pixel shifts (linear)'):
            sa = np.asarray(fpd.synthetic_data.shift_array(scan_len=9, shift_min=-2.0, shift_max=2.0))
            data = fpd.synthetic_data.shift_images(sa, self.im, noise=False, origin='top')
            dyx_ref = np.zeros_like(sa)
        
            unshifted_data = fpdp._shift_images(data.copy(), dyx=-sa, dyx_mode='linear')
            out = fpdp.center_of_mass(unshifted_data, nr=None, nc=None, origin='top', progress_bar=False, print_stats=False)
            self.assertTrue(close_enough(out-self.cyx[..., None, None], dyx_ref, self.rtol, self.atol))
        
        with Timer('sub-pixel mode with sub-pixel shifts (fourier)'):
            sa = np.asarray(fpd.synthetic_data.shift_array(scan_len=9, shift_min=-2.0, shift_max=2.0))
            data = fpd.synthetic_data.shift_images(sa, self.im, noise=False, origin='top')
            dyx_ref = np.zeros_like(sa)
        
            unshifted_data = fpdp._shift_images(data.copy(), dyx=-sa, dyx_mode='fourier')
            out = fpdp.center_of_mass(unshifted_data, nr=None, nc=None, origin='top', progress_bar=False, print_stats=False)
            self.assertTrue(close_enough(out-self.cyx[..., None, None], dyx_ref, self.rtol, self.atol))
        
        with Timer('sub-pixel mode with pixel shifts'):
            sa = np.asarray(fpd.synthetic_data.shift_array(scan_len=9, shift_min=-4.0, shift_max=4.0))
            data = fpd.synthetic_data.shift_images(sa, self.im, noise=False, origin='top')
            dyx_ref = np.zeros_like(sa)
            
            unshifted_data = fpdp._shift_images(data.copy(), dyx=-sa, dyx_mode='linear')
            out = fpdp.center_of_mass(unshifted_data, nr=None, nc=None, origin='top', progress_bar=False, print_stats=False)
            self.assertTrue(close_enough(out-self.cyx[..., None, None], dyx_ref, self.rtol, self.atol))
        
        with Timer('pixel mode with pixel shifts'): 
            sa = np.asarray(fpd.synthetic_data.shift_array(scan_len=9, shift_min=-4.0, shift_max=4.0))
            data = fpd.synthetic_data.shift_images(sa, self.im, noise=False, origin='top')
            dyx_ref = np.zeros_like(sa)
            
            unshifted_data = fpdp._shift_images(data.copy(), dyx=-sa, dyx_mode='pixel')
            out = fpdp.center_of_mass(unshifted_data, nr=None, nc=None, origin='top', progress_bar=False, print_stats=False)
            self.assertTrue(close_enough(out-self.cyx[..., None, None], dyx_ref, self.rtol, self.atol))

if __name__ == '__main__':
    unittest.main()


