
import unittest
import numpy as np
import matplotlib.pylab as plt
plt.ion()

import fpd


class TestTEMTools(unittest.TestCase):      
    def test_rutherford_cs_scalar(self):
        print(self.id())
        z = 79
        dif_cs, mrad = fpd.tem_tools.rutherford_cs(z)
    
    def test_rutherford_cs_scalar_plot(self):
        print(self.id())
        z = 79
        dif_cs, mrad = fpd.tem_tools.rutherford_cs(z, plot=True)
        plt.close('all')
        
    def test_rutherford_cs_iter(self):
        print(self.id())
        z = [79, 10]
        dif_cs, mrad = fpd.tem_tools.rutherford_cs(z)
    
    def test_rutherford_cs_iter_plot(self):
        print(self.id())
        z = [79, 10]
        dif_cs, mrad = fpd.tem_tools.rutherford_cs(z, plot=True)
        plt.close('all')
    
    def test_hkl_cube_fcc(self):
        print(self.id())
        hkl, d, bragg_2t_mrad, p = fpd.tem_tools.hkl_cube(0.407, struct='fcc')
    
    def test_hkl_cube_no_print(self):
        print(self.id())
        hkl, d, bragg_2t_mrad, p = fpd.tem_tools.hkl_cube(0.408, print_first=0)
    
    def test_hkl_cube_bcc(self):
        print(self.id())
        hkl, d, bragg_2t_mrad, p = fpd.tem_tools.hkl_cube(0.287, struct='bcc')
    
    def test_d_from_two_theta(self):
        print(self.id())
        recip_d = fpd.tem_tools.d_from_two_theta(10.673)
    
    def test_e_lambda_rel(self):
        print(self.id())
        lam = fpd.tem_tools.e_lambda(kV=200, rel=True)
    
    def test_e_lambda_nonrel(self):
        print(self.id())
        lam = fpd.tem_tools.e_lambda(kV=200, rel=False)
    
    def test_airy_d(self):
        print(self.id())
        d = fpd.tem_tools.airy_d(0.58e-3)
    
    def test_airy_fwhm(self):
        print(self.id())
        fwhm = fpd.tem_tools.airy_fwhm(0.58e-3)
    
    def test_lambda_iak(self):
        print(self.id())
        rho, alpha, beta, kV = 19.3, 30, 30, 200
        lam = fpd.tem_tools.lambda_iak(rho, alpha, beta, kV)
    
    def test_defocus_from_ctf_crossing(self):
        print(self.id())
        df = fpd.tem_tools.defocus_from_ctf_crossing(4.745/1e-9, kV=200.0, Cs=0.0005)
    
    def test_ctf(self):
        print(self.id())
        import numpy as np
        c = fpd.tem_tools.ctf(k=np.linspace(0, 8, 1000)/1e-9, df=-35.4e-9, kV=200.0, Cs=0.0005, plot=True)
        plt.close('all')
    
    def test_scherzer_defocus(self):
        print(self.id())
        df_s = fpd.tem_tools.scherzer_defocus(Cs=0.0005, extended=False, kV=200.0)
        df_s = fpd.tem_tools.scherzer_defocus(Cs=0.0005, extended=True, kV=200.0)
    
    def test_defocus_from_image(self):
        print(self.id())
        df = 1e-3
        kV = 200.0
        pix_m = 5e-09
        s = 512

        p = np.random.rand(s, s) / 1.0
        a = p * 0 + 1

        fren = fpd.mag_tools.fresnel(a, p, df=df, ypix=pix_m, xpix=pix_m, c_s=0, theta_c=0, kV=kV)

        defocus = fpd.tem_tools.defocus_from_image(fren.im, pix_m, 5, f_max=1.0, kV=kV)
        
        assert np.allclose(df, defocus, rtol=5e-03) # error from dig
    
    def test_TranslationTransform(self):
        print(self.id())
        import skimage.transform as tr
        from fpd.tem_tools import TranslationTransform
        
        # only the estimate and properties are changed and so need testing
        src = np.array([[0, 0], [10, 10]])
        dst = src + np.array([10, 50])

        t1 = tr.EuclideanTransform()
        t1.estimate(src, dst)
        t2 = TranslationTransform()
        t2.estimate(src, dst)
        
        assert np.allclose(t1.translation, t2.translation)
    
    
    def test_TranslationTransform_methods(self):
        print(self.id())
        import skimage.transform as tr
        from fpd.tem_tools import TranslationTransform
        
        # only the estimate and properties are changed and so need testing
        src = np.array([[0, 0], [10, 10]])
        dst = src + np.array([10, 50])

        t1 = tr.EuclideanTransform()
        t1.estimate(src, dst)
        t2 = TranslationTransform(method='median')
        t2.estimate(src, dst)
        t3 = TranslationTransform(method='mean')
        t3.estimate(src, dst)
        assert np.allclose(t1.translation, t2.translation)
        assert np.allclose(t1.translation, t3.translation)
    
    
    def test_apply_image_trans_multi(self):
        print(self.id())
        import skimage.transform as tr
        from skimage.data import astronaut
        from fpd.tem_tools import apply_image_trans
        
        im = astronaut().mean(-1, dtype=np.uint8)
        trans0 = tr.EuclideanTransform()
        trans1 = tr.EuclideanTransform(translation=(-5, -10))
        
        imw = apply_image_trans([im, im], [trans0, trans1], output_mode='same')
        imw = apply_image_trans([im, im], [trans0, trans1], output_mode='expand')
        imw = apply_image_trans([im, im], [trans0, trans1], output_mode='overlap')
        
        #import matplotlib.pylab as plt
        #plt.ion()
        #plt.matshow(imw[0])
        #plt.matshow(imw[1])
        #plt.matshow(imw[0]-imw[1])


    def test_apply_image_trans_single(self):
        print(self.id())
        import skimage.transform as tr
        from skimage.data import astronaut
        from fpd.tem_tools import apply_image_trans
        
        im = astronaut().mean(-1, dtype=np.uint8)   
        trans1 = tr.EuclideanTransform(translation=(-5, -10))
        
        imw = apply_image_trans(im, trans1, output_mode='same')
        imw = apply_image_trans(im, trans1, output_mode='expand')
        imw = apply_image_trans(im, trans1, output_mode='overlap')
    
    
    def test_orb_trans(self):
        print(self.id())
        import skimage.transform as tr
        from skimage.data import astronaut
        
        from fpd.tem_tools import orb_trans, optimise_trans, apply_image_trans
        from fpd.fpd_processing import nrmse
        
        im = astronaut().mean(-1, dtype=np.uint8)
        
        trans_w = tr.AffineTransform(scale=(1.1, 0.9),
                                     translation=(50,15),
                                     rotation=5/180*np.pi,
                                     shear=0.01)
        
        imw = tr.warp(im, trans_w.inverse)
        imw = (imw*255).astype(np.uint8)
        
        trans_meas = orb_trans(im, imw, plot=True, residual_threshold=4, optimise=False)
        trans_meas = orb_trans(im, imw, plot=False, residual_threshold=4, optimise=False)
        imuw = apply_image_trans(imw, trans_meas)
        #print(nrmse(im, np.concatenate([imw[None], imuw[None]], 0)))
        
        roi_s=np.s_[100:400, 100:300]
        trans_opt = optimise_trans(im, imw, trans_meas, roi_s=roi_s)
        
        trans_meas = orb_trans(im, imw, plot=False, residual_threshold=4, optimise=True,
                               roi_s=roi_s)
        
        imuw_opt = apply_image_trans(imw, trans_opt)
        #print(nrmse(im, np.concatenate([imw[None], imuw[None], imuw_opt[None]], 0)))
        
        #print(trans_meas.params-trans_w.params)
        #print(trans_opt.params-trans_w.params)
        
        # test all transforms
        for mode in ['translation', 'euclidean', 'similarity', 'affine']:
            trans_meas = orb_trans(im, imw, ransac_trans=mode, trans=mode,
                                   plot=False, residual_threshold=4, optimise=True)
        plt.close('all')
        
        trans_meas = orb_trans(im, imw, plot=False, residual_threshold=4, optimise=True,
                               fminmax=(0, 0.5), gaus=0.5)
    
    
    def test_blob_log_detect(self):
        print(self.id())
        d1 = fpd.synthetic_data.disk_image(64, radius=9, size=64, sigma=0.5)
        d2 = fpd.synthetic_data.disk_image(64, radius=3, size=64, sigma=0.5)
        d3 = fpd.synthetic_data.shift_im(d2, (16, 16), False)
        
        image = d1+d3
        val = 64
        blobs = fpd.tem_tools.blob_log_detect(image, min_radius=5, max_radius=12, num_radii=7, threshold=0.5, overlap=0.3, log_scale=True, plot=False)
        
        assert len(blobs) == 1
        assert (np.abs(blobs[0, :2] - 32) <=2).all()
        assert (np.abs(blobs[0, 3] - val) <=1).all()
        assert blobs.shape[-1] == 6
        assert np.isnan(blobs[:, -2:]).all()
        
        blobs = fpd.tem_tools.blob_log_detect(image, min_radius=5, max_radius=12, num_radii=7, threshold=0.5, overlap=0.3, log_scale=True, plot=True)
        plt.close('all')
    
    def test_blob_log_detect_xc(self):
        print(self.id())
        d1 = fpd.synthetic_data.disk_image(64, radius=9, size=64, sigma=0.5)
        d2 = fpd.synthetic_data.disk_image(64, radius=3, size=64, sigma=0.5)
        d3 = fpd.synthetic_data.shift_im(d2, (16, 16), False)
        
        image = d1+d3
        blobs = fpd.tem_tools.blob_log_detect(image, min_radius=5, max_radius=12, num_radii=7, threshold=0.5, overlap=0.3, log_scale=True, ref_im=d1, sigma=1, log_xc_max_r=9, plot=False)
        
        assert len(blobs) == 1
        assert (np.abs(blobs[0, :2] - 32) <=2).all()
        assert blobs.shape[-1] == 6
        assert np.isnan(blobs[:, -2:]).all()
        
        blobs = fpd.tem_tools.blob_log_detect(image, min_radius=5, max_radius=12, num_radii=7, threshold=0.5, overlap=0.3, log_scale=True, ref_im=d1, sigma=1, log_xc_max_r=9, plot=True)
        plt.close('all')
    
    
    def test_blob_log_detect_xc_norm(self):
        print(self.id())
        d1 = fpd.synthetic_data.disk_image(64, radius=9, size=64, sigma=0.5)
        d2 = fpd.synthetic_data.disk_image(64, radius=3, size=64, sigma=0.5)
        d3 = fpd.synthetic_data.shift_im(d2, (16, 16), False)
        
        image = d1+d3
        blobs = fpd.tem_tools.blob_log_detect(image, min_radius=5, max_radius=12, num_radii=7, threshold=0.5, overlap=0.3, log_scale=True, ref_im=d1, sigma=1, phase_exponent=0, log_xc_max_r=3, plot=False)
        
        assert len(blobs) == 1
        assert (np.abs(blobs[0, :2] - 32) <=2).all()
        
        blobs = fpd.tem_tools.blob_log_detect(image, min_radius=5, max_radius=12, num_radii=7, threshold=0.5, overlap=0.3, log_scale=True, ref_im=d1, sigma=1, phase_exponent=0.5, log_xc_max_r=3, plot=False)
        
        assert len(blobs) == 1
        assert (np.abs(blobs[0, :2] - 32) <=2).all()
    
    
    def test_blob_log_detect_xc_sigma0(self):
        print(self.id())
        d1 = fpd.synthetic_data.disk_image(64, radius=9, size=64, sigma=0.5)
        d2 = fpd.synthetic_data.disk_image(64, radius=3, size=64, sigma=0.5)
        d3 = fpd.synthetic_data.shift_im(d2, (16, 16), False)
        
        image = d1+d3
        blobs = fpd.tem_tools.blob_log_detect(image, min_radius=5, max_radius=12, num_radii=7, threshold=0.5, overlap=0.3, log_scale=True, ref_im=d1, sigma=0, phase_exponent=0, log_xc_max_r=3, plot=False)
        
        assert len(blobs) == 1
        assert (np.abs(blobs[0, :2] - 32) <=2).all()
        
        blobs = fpd.tem_tools.blob_log_detect(image, min_radius=5, max_radius=12, num_radii=7, threshold=0.5, overlap=0.3, log_scale=True, ref_im=d1, sigma=0, phase_exponent=0, log_xc_max_r=3, plot=True)
        plt.close('all')
    
    
    def test_blob_log_detect_xc_sigma_negative(self):
        print(self.id())
        d1 = fpd.synthetic_data.disk_image(64, radius=9, size=64, sigma=0.5)
        d2 = fpd.synthetic_data.disk_image(64, radius=3, size=64, sigma=0.5)
        d3 = fpd.synthetic_data.shift_im(d2, (16, 16), False)
        
        image = d1+d3
        blobs = fpd.tem_tools.blob_log_detect(image, min_radius=5, max_radius=12, num_radii=7, threshold=0.5, overlap=0.3, log_scale=True, ref_im=d1, sigma=-2, phase_exponent=0, log_xc_max_r=3, plot=False)
        
        assert len(blobs) == 1
        assert (np.abs(blobs[0, :2] - 32) <=2).all()
        
        blobs = fpd.tem_tools.blob_log_detect(image, min_radius=5, max_radius=12, num_radii=7, threshold=0.5, overlap=0.3, log_scale=True, ref_im=d1, sigma=-2, phase_exponent=0, log_xc_max_r=3, plot=True)
        plt.close('all')
    
    
    def test_blob_log_detect_sp(self):
        print(self.id())
        d1 = fpd.synthetic_data.disk_image(64, radius=9, size=64, sigma=0.5)
        d2 = fpd.synthetic_data.disk_image(64, radius=3, size=64, sigma=0.5)
        d3 = fpd.synthetic_data.shift_im(d2, (16, 16), False)
        
        image = d1+d3
        blobs = fpd.tem_tools.blob_log_detect(image, min_radius=5, max_radius=12, num_radii=7, threshold=0.5, overlap=0.3, log_scale=True, plot=False, subpix_log=True)
        
        assert len(blobs) == 1
        assert (np.abs(blobs[0, :2] - 32) <=2).all()
        assert blobs.shape[-1] == 6
        assert (np.isnan(blobs[:, -2:]) == False).all()
        
    def test_blob_log_detect_sp_fit_hw_r(self):
        print(self.id())
        d1 = fpd.synthetic_data.disk_image(64, radius=9, size=64, sigma=0.5)
        d2 = fpd.synthetic_data.disk_image(64, radius=3, size=64, sigma=0.5)
        d3 = fpd.synthetic_data.shift_im(d2, (16, 16), False)
        
        image = d1+d3
        blobs = fpd.tem_tools.blob_log_detect(image, min_radius=5, max_radius=12, num_radii=7, threshold=0.5, overlap=0.3, log_scale=True, plot=False, subpix_log=True, fit_hw=None)
        
        assert len(blobs) == 1
        assert (np.abs(blobs[0, :2] - 32)  <= 2).all()
    
    def test_blob_log_detect_sp_fit_hw_r_float(self):
        print(self.id())
        d1 = fpd.synthetic_data.disk_image(64, radius=9, size=64, sigma=0.5)
        d2 = fpd.synthetic_data.disk_image(64, radius=3, size=64, sigma=0.5)
        d3 = fpd.synthetic_data.shift_im(d2, (16, 16), False)
        
        image = d1+d3
        blobs = fpd.tem_tools.blob_log_detect(image, min_radius=5, max_radius=12, num_radii=7, threshold=0.5, overlap=0.3, log_scale=True, plot=False, subpix_log=True, fit_hw=1.5)
        
        assert len(blobs) == 1
        assert (np.abs(blobs[0, :2] - 32) <=2).all()
        
        
    def test_blob_log_detect_xc_sp(self):
        print(self.id())
        d1 = fpd.synthetic_data.disk_image(64, radius=9, size=64, sigma=0.5)
        d2 = fpd.synthetic_data.disk_image(64, radius=3, size=64, sigma=0.5)
        d3 = fpd.synthetic_data.shift_im(d2, (16, 16), False)
        
        image = d1+d3
        blobs = fpd.tem_tools.blob_log_detect(image, min_radius=5, max_radius=12, num_radii=7, threshold=0.5, overlap=0.3, log_scale=True, ref_im=d1, sigma=1, log_xc_max_r=9, plot=False, subpix_xc=True)
        
        assert len(blobs) == 1
        assert (np.abs(blobs[0, :2] - 32) <=2).all()
        assert blobs.shape[-1] == 6
        assert (np.isnan(blobs[:, -2:]) == False).all()
    
        
    def test_blob_log_detect_xc_sp_params(self):
        print(self.id())
        d1 = fpd.synthetic_data.disk_image(64, radius=9, size=64, sigma=0.5)
        d2 = fpd.synthetic_data.disk_image(64, radius=3, size=64, sigma=0.5)
        d3 = fpd.synthetic_data.shift_im(d2, (16, 16), False)
        
        image = d1+d3
        blobs = fpd.tem_tools.blob_log_detect(image, min_radius=5, max_radius=12, num_radii=7, threshold=0.5, overlap=0.3, log_scale=True, ref_im=d1, sigma=1, log_xc_max_r=9, plot=False, subpix_xc=True, fit_hw=3, subpix_dict={'smoothing' : 2})
        
        assert len(blobs) == 1
        assert (np.abs(blobs[0, :2] - 32) <=2).all()  
    
    
    def test_friedel_filter(self):
        print(self.id())
        d0 = fpd.synthetic_data.disk_image(256, radius=5, size=64, sigma=0.5)
        d1 = fpd.synthetic_data.disk_image(64, radius=3, size=64, sigma=0.5)
        d2 = fpd.synthetic_data.shift_im(d1, (16, 16), False)
        d3 = fpd.synthetic_data.shift_im(d1, (-16, -16), False)
        d4 = fpd.synthetic_data.shift_im(d1, (-16, 16), False)
        
        image = d0 + d2 + d3 + d4
        plt.matshow(image)
        
        blobs = fpd.tem_tools.blob_log_detect(image, min_radius=1, max_radius=9, num_radii=12, threshold=0.1, overlap=0.3, log_scale=True, plot=False)
        
        blobs_filtered = fpd.tem_tools.friedel_filter(blobs, cyx=(31.5, 31.5), min_distance_radii_scale=1, min_distance_pad=2, plot=False)
        assert len(blobs_filtered) == 3
        
        blobs_filtered = fpd.tem_tools.friedel_filter(blobs, cyx=(31.5, 31.5), min_distance_radii_scale=1, min_distance_pad=2, plot=True)
        
        blobs_filtered, cyx_opt = fpd.tem_tools.friedel_filter(blobs, cyx=(31.5, 31.5), min_distance_radii_scale=1, min_distance_pad=2, optimise_cyx=True, plot=False)
        
        plt.close('all')
    
    def test_synthetic_lattice(self):
        print(self.id())
        cyx = (128,)*2
        shape = (256,256)
        ab = (40, 40)
        
        yxg = fpd.tem_tools.synthetic_lattice(cyx=cyx, ab=ab, angles=(0, np.pi/2+0.2), shape=shape, plot=False)
        yxg = fpd.tem_tools.synthetic_lattice(cyx=cyx, ab=ab, angles=(0, np.pi/2+0.2), shape=shape, plot=True)
        yxg = fpd.tem_tools.synthetic_lattice(cyx=cyx, ab=ab, angles=(0, np.pi/2+0.2), plot=True, max_r=256)
        plt.close('all')
    
    def test_synthetic_lattice_square(self):
        print(self.id())
        reps = (3,)*2
        
        cyx = (4, 5)
        ab = (1.5, 1.5)
        angles = (0.1, 0.1+np.pi/2)
        yxg = fpd.tem_tools.synthetic_lattice(cyx=cyx, ab=ab, angles=angles, reps=reps, plot=False)
        
        yxg_nom = np.array([[2.35774363, 3.65724388],
                            [2.50749375, 5.14975012],
                            [2.65724388, 6.64225637],
                            [3.85024988, 3.50749375],
                            [4.        , 5.        ],
                            [4.14975012, 6.49250625],
                            [5.34275612, 3.35774363],
                            [5.49250625, 4.85024988],
                            [5.64225637, 6.34275612]])
        assert np.allclose(yxg_nom, yxg)
    
    def test_synthetic_lattice_rectangle(self):
        print(self.id())
        reps = (3,)*2
        
        cyx = (4, 5)
        ab = (1.5, 4.5)
        angles = (0.1, 0.1+np.pi/2)
        yxg = fpd.tem_tools.synthetic_lattice(cyx=cyx, ab=ab, angles=angles, reps=reps, plot=False)
        
        yxg_nom = np.array([[-0.62726887,  3.95674413],
                            [-0.47751874,  5.44925037],
                            [-0.32776862,  6.94175662],
                            [ 3.85024988,  3.50749375],
                            [ 4.        ,  5.        ],
                            [ 4.14975012,  6.49250625],
                            [ 8.32776862,  3.05824338],
                            [ 8.47751874,  4.55074963],
                            [ 8.62726887,  6.04325587]])
        assert np.allclose(yxg_nom, yxg)

    def test_synthetic_lattice_hex(self):
        print(self.id())
        reps = (3,)*2
        
        cyx = (4, 5)
        ab = (4.5, 4.5)
        angles = (0.1, 0.1+np.pi/3)
        yxg = fpd.tem_tools.synthetic_lattice(cyx=cyx, ab=ab, angles=angles, reps=reps, plot=False)
        
        yxg_nom = np.array([[-0.55152054, -1.32721588],
                            [-0.10227017,  3.15030287],
                            [ 0.34698021,  7.62782161],
                            [ 3.55074963,  0.52248126],
                            [ 4.        ,  5.        ],
                            [ 4.44925037,  9.47751874],
                            [ 7.65301979,  2.37217839],
                            [ 8.10227017,  6.84969713],
                            [ 8.55152054, 11.32721588]])
        assert np.allclose(yxg_nom, yxg)
    
    def test_synthetic_lattice_oblique(self):
        print(self.id())
        reps = (3,)*2
        
        cyx = (4, 5)
        ab = (2.5, 4.5)
        angles = (0.1, 0.1+np.pi/2-0.5)
        yxg = fpd.tem_tools.synthetic_lattice(cyx=cyx, ab=ab, angles=angles, reps=reps, plot=False)
        
        yxg_nom = np.array([[-0.39435801,  0.76010705],
                            [-0.14477447,  3.24761746],
                            [ 0.10480907,  5.73512787],
                            [ 3.75041646,  2.51248959],
                            [ 4.        ,  5.        ],
                            [ 4.24958354,  7.48751041],
                            [ 7.89519093,  4.26487213],
                            [ 8.14477447,  6.75238254],
                            [ 8.39435801,  9.23989295]])
        assert np.allclose(yxg_nom, yxg)

    def test_vector_combinations(self):
        print(self.id())
        cyx = (128,)*2
        shape = (256,256)
        ab = (40, 40)
        yxg = fpd.tem_tools.synthetic_lattice(cyx=cyx, ab=ab, angles=(0, np.pi/2+0.2), shape=shape, plot=False)
        
        dyx, rt = fpd.tem_tools.vector_combinations(yxg, plot=False)
        dyx, rt = fpd.tem_tools.vector_combinations(yxg, plot=True)
        plt.close('all')
    
    def test_lattice_angles(self):
        print(self.id())
        cyx = (128,)*2
        shape = (256,256)
        angles = (0, np.pi/2)
        ab = (40, 40)
        
        yxg = fpd.tem_tools.synthetic_lattice(cyx=cyx, ab=ab, angles=angles, shape=shape, plot=False)
        dyx, rt = fpd.tem_tools.vector_combinations(yxg, plot=False)
        
        (a1, a2) = fpd.tem_tools.lattice_angles(rt, nfold=None, bin_deg=1.0, hist_gaus=2.0, weight_hist=True, trim_r_max_pct=79, min_distance=None, plot=False)
        assert np.allclose((a1, a2), angles, rtol=1e-02, atol=np.pi/180)
        
        (a1, a2) = fpd.tem_tools.lattice_angles(rt, nfold=2, bin_deg=1.0, hist_gaus=2.0, weight_hist=True, trim_r_max_pct=79, min_distance=None, plot=False)
        assert np.allclose((a1, a2), angles, rtol=1e-02, atol=np.pi/180)
        
        (a1, a2) = fpd.tem_tools.lattice_angles(rt, nfold=None, bin_deg=1.0, hist_gaus=2.0, weight_hist=True, trim_r_max_pct=79, min_distance=None, plot=True)
        plt.close('all')
        
    def test_lattice_magnitudes(self):
        print(self.id())
        cyx = (128,)*2
        shape = (256,256)
        angles = (0, np.pi/2)
        ab = (40, 40)
        
        yxg = fpd.tem_tools.synthetic_lattice(cyx=cyx, ab=ab, angles=angles, shape=shape, plot=False)
        dyx, rt = fpd.tem_tools.vector_combinations(yxg, plot=False)
        
        lattice_constants = fpd.tem_tools.lattice_magnitudes(rt, angles, window_deg=5, bin_pix=1.0, hist_gaus=2.0, min_vector_mag=None, max_vector_mag=None, mode='peaks', peak_min_distance=None, plot=False)
        assert np.allclose(lattice_constants, ab, rtol=1e-02, atol=2)
        
        lattice_constants = fpd.tem_tools.lattice_magnitudes(rt, angles, window_deg=5, bin_pix=1.0, hist_gaus=2.0, min_vector_mag=None, max_vector_mag=None, mode='fft', peak_min_distance=None, plot=False)
        assert np.allclose(lattice_constants, ab, rtol=1e-02, atol=2)
        
        
        lattice_constants = fpd.tem_tools.lattice_magnitudes(rt, angles, window_deg=5, bin_pix=1.0, hist_gaus=2.0, min_vector_mag=None, max_vector_mag=None, mode='peaks', peak_min_distance=None, plot=True)
        
        lattice_constants = fpd.tem_tools.lattice_magnitudes(rt, angles, window_deg=5, bin_pix=1.0, hist_gaus=2.0, min_vector_mag=None, max_vector_mag=None, mode='fft', peak_min_distance=None, plot=True)
        plt.close('all')
    
    def test_lattice_resolver(self):
        print(self.id())
        cyx = (128,)*2
        shape = (256,256)
        angles = (0, np.pi/2)
        ab = (40, 40)
        
        yxg = fpd.tem_tools.synthetic_lattice(cyx=cyx, ab=ab, angles=angles, shape=shape, plot=False)
        dyx, rt = fpd.tem_tools.vector_combinations(yxg, plot=False)
        
        (a1, a2) = fpd.tem_tools.lattice_angles(rt, nfold=None, bin_deg=1.0, hist_gaus=2.0, weight_hist=True, trim_r_max_pct=79, min_distance=None, plot=False)
        assert np.allclose((a1, a2), angles, rtol=1e-02, atol=np.pi/180)
        
        lattice_constants = fpd.tem_tools.lattice_magnitudes(rt, angles, window_deg=5, bin_pix=1.0, hist_gaus=2.0, min_vector_mag=None, max_vector_mag=None, mode='peaks', peak_min_distance=None, plot=False)
        assert np.allclose(lattice_constants, ab, rtol=1e-02, atol=2)
        
        angles_in = np.array((a1, a2))[:, None, None]
        ab_in = np.array(lattice_constants)[:, None, None]
        abs_res, angles_res = fpd.tem_tools.lattice_resolver(ab_in, angles_in, (45, 135))
        
        assert np.allclose(abs_res[:, 0, 0], np.array(ab)*2**0.5, rtol=2e-2)
        assert np.allclose(angles_res[:, 0, 0], np.array([np.pi/4, np.pi*3/4]), rtol=2e-2)
    
    def test_lattice_from_inliers(self):
        print(self.id())
        cyx = (128,)*2
        shape = (256,256)
        angles = (0, np.pi/2)
        ab = (40, 40)
        
        yx = fpd.tem_tools.synthetic_lattice(cyx=cyx, ab=ab, angles=angles, shape=shape, plot=False)
        
        ab_in, angles_in = fpd.tem_tools.lattice_from_inliers(yx, cyx, r=2, max_n=10, r_min=None, r_max=None, 
                         min_dangle=10, max_dangle=170, degen_mode='max_geom_r',
                         degen_func=None, reps=41, plot=False)
        
        abs_res, angles_res = fpd.tem_tools.lattice_resolver(ab_in[:, None, None], angles_in[:, None, None], (0, 90))
        abs_res = np.squeeze(abs_res)
        angles_res = np.squeeze(angles_res)
        
        assert np.allclose(abs_res, np.array(ab), rtol=2e-2)
        assert np.allclose(angles_res, angles, rtol=1e-02, atol=np.pi/180)
        
        ab2, angles2 = fpd.tem_tools.lattice_from_inliers(yx, cyx, r=8, max_n=10, r_min=None, r_max=None, 
                         min_dangle=10, max_dangle=170, degen_mode='max_geom_r',
                         degen_func=None, reps=41, plot=True)
        plt.close('all')
        
        
        t = lambda r1, r2, a1, a2: r1*r2
        ab_in, angles_in = fpd.tem_tools.lattice_from_inliers(yx, cyx, r=8, max_n=np.inf, r_min=5, r_max=80, 
                         min_dangle=10, max_dangle=170, degen_mode='max_geom_r',
                         degen_func=t, reps=41, plot=True)
        abs_res, angles_res = fpd.tem_tools.lattice_resolver(ab_in[:, None, None], angles_in[:, None, None], (0, 90))
        abs_res = np.squeeze(abs_res)
        angles_res = np.squeeze(angles_res)
        
        assert np.allclose(abs_res, np.array(ab), rtol=2e-2)
        assert np.allclose(angles_res, angles, rtol=1e-02, atol=np.pi/180)
        plt.close('all')
        
    
    def test_lattice_inlier(self):
        print(self.id())
        cyx = (128,)*2
        shape = (256,256)
        angles = (0, np.pi/2)
        ab = (40, 40)
        
        yxg1 = fpd.tem_tools.synthetic_lattice(cyx=cyx, ab=ab, angles=angles, shape=shape, plot=False)
        yxg2 = fpd.tem_tools.synthetic_lattice(cyx=cyx, ab=[x*1.3 for x in ab], angles=angles, shape=shape, plot=False)
        
        inliers = fpd.tem_tools.lattice_inlier(yxg1, yxg1, r=4, plot=False)
        assert inliers.sum() == len(yxg1)
        
        inliers = fpd.tem_tools.lattice_inlier(yxg1, yxg2, r=4, plot=False)
        assert inliers.sum() == 1
        
        inliers = fpd.tem_tools.lattice_inlier(yxg1, yxg1, r=4, plot=True)
        plt.close('all')
    
    def test_optimise_lattice(self):
        print(self.id())
        cyx = (128,)*2
        shape = (256,256)
        angles = (0, np.pi/2)
        ab = (40, 40)
        
        yxg = fpd.tem_tools.synthetic_lattice(cyx=cyx, ab=ab, angles=angles, shape=shape, plot=False)
        yx = yxg
        cyx_opt, ab_opt, angles_opt, error = fpd.tem_tools.optimise_lattice(yx, cyx, ab=ab,
                                                                     angles=angles, shape=shape,
                                                                     constraints=None, plot=False)
    def test_optimise_lattice_plot(self):
        print(self.id())
        cyx = (128,)*2
        shape = (256,256)
        angles = (0, np.pi/2)
        ab = (40, 40)
        
        yxg = fpd.tem_tools.synthetic_lattice(cyx=cyx, ab=ab, angles=angles, shape=shape, plot=False)
        yx = yxg       
        cyx_opt, ab_opt, angles_opt, error = fpd.tem_tools.optimise_lattice(yx, cyx, ab=ab,
                                                                     angles=angles, shape=shape,
                                                                     constraints=None, plot=True)
        plt.close('all')
        
    def test_optimise_lattice_constraint_bounds(self):
        print(self.id())
        cyx = (128,)*2
        shape = (256,256)
        angles = (0, np.pi/2)
        ab = (40, 40)
        
        yxg = fpd.tem_tools.synthetic_lattice(cyx=cyx, ab=ab, angles=angles, shape=shape, plot=False)
        yx = yxg
        
        cons = {'type': 'eq', 'fun': lambda x:  x[3] - x[2] - np.pi/2}
        bounds = [(120, 140), (120, 140), (None, None), (None, None), (None, None), (None, None)]
        cyx_opt, ab_opt, angles_opt, error = fpd.tem_tools.optimise_lattice(yx, cyx, ab=ab,
                                                                     angles=angles, shape=shape,
                                                                     constraints=cons, plot=False,
                                                                     bounds=bounds)
        plt.close('all')
    
    def test_background_erosion(self):
        print(self.id())
        im_shape = (256,)*2
        cyx = (np.array(im_shape) -1) / 2
        
        ### background
        yy, xx = np.indices(im_shape)
        im_bk = -((xx-cyx[1])**2 + (yy-cyx[0])**2)
        im_bk -= im_bk.min()
        im_bk /= im_bk.max() / 100
        #plt.matshow(im_bk); plt.colorbar()
        
        ### lattice
        r = 4
        d0 = fpd.synthetic_data.disk_image(intensity=100, radius=r, size=im_shape[0], sigma=0.5)
        yxg = fpd.tem_tools.synthetic_lattice(cyx=cyx, ab=(50,)*2, angles=(0, np.pi/2), shape=im_shape, plot=False)
        yxg -= cyx
        im_peaks = fpd.synthetic_data.array_image(d0, yxg)
        #plt.matshow(im_peaks); plt.colorbar()
        
        ### image
        im = im_bk + im_peaks
        #plt.matshow(im); plt.colorbar()
        
        im_bk_ext = fpd.tem_tools.background_erosion(im, radius=r, plot=False)
        im_bk_ext = fpd.tem_tools.background_erosion(im, radius=r, plot=True)
        
        #plt.matshow(im_bk_ext - im_bk); plt.colorbar()
        assert np.allclose(im_bk, im_bk_ext, rtol=15e-02, atol=15)
        
        #plt.matshow(im - im_bk_ext); plt.colorbar()
        assert np.allclose(im_peaks, im - im_bk_ext, rtol=10e-02, atol=10)
    
    def test_nc_correct(self):
        print(self.id())
        from fpd.tem_tools import nc_correct
        sum_im, nc = np.random.rand(2, 256, 256)
        
        cim = nc_correct(sum_im, nc, plot=False)
        
        cim = nc_correct(sum_im, nc, plot=True)
        plt.close('all')
    
    def test_strain(self):
        print(self.id())
        # reference
        rt_ref = [np.array([50, 50]), np.array([0, -np.pi/2])]
        r_ref, t_ref = rt_ref
        
        ## test data
        # scale x
        # scale y
        # scale yx
        # rotate (ac w/ y0 at bot)
        # shear ll x
        # shear ll y
        # shear in y and x
        rts = [[r_ref * np.array([1.1, 1]), t_ref],
            [r_ref * np.array([1, 1.1]), t_ref],
            [r_ref * np.array([1.1, 1.1]), t_ref],
            [r_ref, t_ref + np.deg2rad(-15)],
            [r_ref * np.array([1, 1.0 / np.cos(np.deg2rad(15))]), t_ref + np.deg2rad([0, 15])],
            [r_ref * np.array([1.0 / np.cos(np.deg2rad(-15)), 1]), t_ref + np.deg2rad([-15, 0])],
            [r_ref * np.array([1.0 / np.cos(np.deg2rad(-15)), 1.0 / np.cos(np.deg2rad(15))]), t_ref + np.deg2rad([-15, 15])]]
        rts = np.array(rts)
        
        ### plot synthetic lattices
        #imsize = 256
        #cyx = np.array((imsize//2,) * 2)
        #yxg_ref = fpd.tem_tools.synthetic_lattice(cyx, r_ref, t_ref, shape=(imsize,)*2)
        
        #yxgs = [fpd.tem_tools.synthetic_lattice(cyx, ri, ti, shape=(imsize,)*2) for ri, ti in rts]
        
        #import matplotlib.pylab as plt
        #plt.ion()
        #plt.figure()
        #ax = plt.gca()
        #ax.set_aspect(1)
        #ax.invert_yaxis()
        
        #plt.plot(yxg_ref[:, 1], yxg_ref[:, 0], 'ko')
        #for yxgi in yxgs:
            #plt.plot(yxgi[:, 1], yxgi[:, 0], 'o')
        
        ### test single
        ri, ti = rts[-1]
        es = fpd.tem_tools.strain(r_ref=r_ref, t_ref=t_ref, r=ri, t=ti,
                                  invert_space=False, origin='top', plot=True)
        assert es.shape == (4,)
        
        
        ### test line
        # rts is [example #, [r, t], [vector 1, v2]
        # swap to 2xM
        ri = np.moveaxis(rts[:, 0], -1, 0)
        ti = np.moveaxis(rts[:, 1], -1, 0)
        
        es = fpd.tem_tools.strain(r_ref=r_ref, t_ref=t_ref, r=ri, t=ti,
                                  invert_space=False, origin='top', plot=True)
        assert es.shape == (4,) + ri.shape[1:]
        
        # results should be independent of the order of input vector
        es2 = fpd.tem_tools.strain(r_ref=r_ref[::-1], t_ref=t_ref[::-1], r=ri[::-1], t=ti[::-1],
                                   invert_space=False, origin='top', plot=True)
        assert(np.allclose(es, es2))
        
        #print(np.round(es, 6))
        es_ref = np.array([[ 0.1,       0.,        0.1,      -0.034074,  0.,        0.,        0.      ],
                           [ 0.,        0.1,       0.1,      -0.034074,  0.,        0.,        0.      ],
                           [ 0.,        0.,        0.,        0.,        0.133975,  0.133975,  0.267949],
                           [ 0.,        0.,        0.,        0.258819, -0.133975,  0.133975,  0.      ]])
        assert np.allclose(es, es_ref)
        
        
        ### test area
        ny = 3
        ri = np.array((rts[:, 0],)*ny)
        ti = np.array((rts[:, 1],)*ny)
        # swap to 2xMxN
        ri = np.moveaxis(ri, -1, 0)
        ti = np.moveaxis(ti, -1, 0)
        es = fpd.tem_tools.strain(r_ref=r_ref, t_ref=t_ref, r=ri, t=ti,
                                  invert_space=False, origin='top', plot=True)
        assert es.shape == (4,) + ri.shape[1:]
        
        # test plots
        es = fpd.tem_tools.strain(r_ref=r_ref, t_ref=t_ref, r=ri, t=ti,
                                  invert_space=False, origin='top', plot=True,
                                  pct=1, centred_cmap=False)
        
        # test invert space
        es = fpd.tem_tools.strain(r_ref=r_ref, t_ref=t_ref, r=ri, t=ti,
                                  invert_space=True, origin='top', plot=True,
                                  pct=1, centred_cmap=False)
        
    def test_strain_basis(self):
        print(self.id())
        
        from fpd import tem_tools as fpdtt
        from fpd import fpd_processing as fpdp
        from itertools import product, combinations

        # make ref and non-ref coords
        yxg_ref = fpdtt.synthetic_lattice(cyx=(0,0), ab=(1, 1), angles=(np.pi/2, 0), reps=(3,)*2, plot=True)
        yxg = fpdtt.synthetic_lattice(cyx=(0,0), ab=(1.1, 1), angles=(np.pi/2+0.1, 0), reps=(3,)*2, plot=True)

        # add some rotation
        deg = 5
        yxg = fpdp.rotate_vector(yxg, deg, axis=-1).T

        # convert to polar
        t_refs = np.arctan2(*yxg_ref.T)
        r_refs = np.linalg.norm(yxg_ref, axis=1)
        ts = np.arctan2(*yxg.T)
        rs = np.linalg.norm(yxg, axis=1)

        # generate all combinations of vectors
        inds = range(len(yxg_ref))
        inds_pair = combinations(inds, 2)

        # loop over inds to calculate es
        ess = []
        for i in inds_pair:
            # index to get local variables
            i = list(i)
            r_ref, t_ref = r_refs[i], t_refs[i]
            r, t = rs[i], ts[i]
            
            # skip any combination with a zero-length vector
            if np.isclose([r_ref, r], 0, atol=1e-3).any():
                continue
            
            # skip any combination with near-parallel vectors 
            dt_ref = np.diff(t_ref)[0] % np.pi
            dt = np.diff(t)[0] % np.pi
            if np.isclose([dt_ref, dt], 0, atol=np.deg2rad(5)).any():
                continue
            
            # calculate es
            es = fpdtt.strain(r_ref, t_ref, r, t)
            ess.append(es)
        ess = np.array(ess)
        #print(np.round(ess, 3))
        assert np.allclose(ess, ess[0])
        #print(ess)

        
if __name__ == '__main__':
    unittest.main()


