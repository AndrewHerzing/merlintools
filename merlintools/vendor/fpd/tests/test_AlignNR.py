
import unittest
import numpy as np
import matplotlib.pylab as plt
plt.ion()

from skimage.data import gravel
from skimage.transform import PiecewiseAffineTransform, warp

from fpd.tem_tools import AlignNR
from fpd.fpd_processing import nrmse
from fpd.fpd_processing import rotate_vector


class TestAlignNR(unittest.TestCase):   
    @classmethod
    def setUpClass(self):
        image = gravel() / 255.0
        image = image[:128, :128]    # use subset for speed
        # edge
        edge = 3
        image[:edge,:] = 0
        image[-edge:,:] = 0
        image[:, :edge] = 0
        image[:, -edge:] = 0
        #plt.matshow(image)
        
        
        rows, cols = image.shape

        src_cols = np.linspace(0, cols, 20)
        src_rows = np.linspace(0, rows, 20)
        src_rows, src_cols = np.meshgrid(src_rows, src_cols)
        src = np.dstack([src_cols.flat, src_rows.flat])[0]

        # add sinusoidal oscillations to coordinates
        dy0 = np.sin(np.linspace(0, 3 * np.pi, src.shape[0])) * 3
        dx0 = np.cos(np.linspace(0, 3 * np.pi, src.shape[0])) * 3

        # form stack with displacement field rotation (in z)
        r = np.linspace(0, 2*np.pi, 8, endpoint=False)

        warped = np.empty(r.shape + image.shape)
        dys = np.empty(r.shape + dy0.shape)
        dxs = np.empty(r.shape + dx0.shape)
        for i in range(len(r)):
            dy, dx = rotate_vector(np.array([dy0, dx0]), theta=np.rad2deg(r[i]), axis=0)

            dst_rows = src[:, 1] + dy
            dst_cols = src[:, 0] + dx
            dst = np.vstack([dst_cols, dst_rows]).T

            tform = PiecewiseAffineTransform()
            tform.estimate(src, dst)

            out = warp(image, tform)
            warped[i] = out
            dys[i] = dy
            dxs[i] = dx
            
            #fig, ax = plt.subplots()
            #ax.imshow(out)
            #ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.r')
            #ax.axis((0, out.shape[1], out.shape[0], 0))
            #plt.show()
        #plt.matshow(image - warped[0])
        
        self.image = image
        self.warped_stack = warped
        self.edge = edge
        self.nrmse_min = 0.04 #0.0275
    
    def err(self, images, ref, edge=None):
        if images.ndim == 2:
            images = images[None]
        
        if edge is not None:
            s = np.s_[edge:-edge, edge:-edge]
        else:
            s = np.s_[:]
        return np.array([nrmse(ref[s], imi[s], allow_nans=True) for imi in images])
        
    def test_stack_mean(self):
        print(self.id())
        A = AlignNR(self.warped_stack, reference='mean',
                    alpha=40, niter=200, nrmse_min=self.nrmse_min, nrmse_rel=0, nrmse_mode='mean',
                    pct=0, grad_sigma=1, reg_mode='gaussian', reg_sigma=8, nthreads=1)
        err = self.err(A.images_aligned, self.image, self.edge + 1)
        plt.close('all')
        assert np.all(err <= 0.1)
        
    def test_stack_median(self):
        print(self.id())
        A = AlignNR(self.warped_stack, reference='median',
                    alpha=40, niter=200, nrmse_min=self.nrmse_min, nrmse_rel=0, nrmse_mode='mean',
                    pct=0, grad_sigma=1, reg_mode='gaussian', reg_sigma=8, nthreads=1)
        err = self.err(A.images_aligned, self.image, self.edge + 1)
        plt.close('all')
        assert np.all(err <= 0.1)
        
    def test_stack_int(self):
        print(self.id())
        i = 4
        A = AlignNR(self.warped_stack, reference=i,
                    alpha=40, niter=200, nrmse_min=self.nrmse_min, nrmse_rel=0, nrmse_mode='mean',
                    pct=0, grad_sigma=1, reg_mode='gaussian', reg_sigma=8, nthreads=1)
        err = self.err(A.images_aligned, self.warped_stack[i], self.edge + 1)
        plt.close('all')
        assert np.all(err <= 0.1)
    
    def test_stack_im(self):
        print(self.id())
        A = AlignNR(self.warped_stack, reference=self.image,
                    alpha=40, niter=200, nrmse_min=self.nrmse_min+0.02, nrmse_rel=0, nrmse_mode='mean',
                    pct=0, grad_sigma=1, reg_mode='gaussian', reg_sigma=8, nthreads=1)
        err = self.err(A.images_aligned, self.image, self.edge + 1)
        plt.close('all')
        assert np.all(err <= 0.1)
    
    def test_single_image(self):
        print(self.id())
        A = AlignNR(self.warped_stack[0], reference=self.image,
                    alpha=40, niter=200, nrmse_min=self.nrmse_min, nrmse_rel=0, nrmse_mode='mean',
                    pct=0, grad_sigma=1, reg_mode='gaussian', reg_sigma=8, nthreads=1)
        err = self.err(A.images_aligned, self.image, self.edge + 1)
        plt.close('all')
        assert np.all(err <= 0.1)
    
    def test_single_image_dyx0(self):
        print(self.id())
        # same as test_single_image w/o plot
        A = AlignNR(self.warped_stack[0], reference=self.image,
                    alpha=40, niter=200, nrmse_min=0.0275, nrmse_rel=0, nrmse_mode='mean',
                    pct=0, grad_sigma=1, reg_mode='gaussian', reg_sigma=8,
                    plot=False, block=True)
        err = self.err(A.images_aligned, self.image, self.edge + 1)
        plt.close('all')
        assert np.all(err <= 0.1)
        
        # single iteration using output of above
        dyx0 = np.squeeze(A.dyx)
        
        Ab = AlignNR(self.warped_stack[0], reference=self.image,
                    alpha=40, niter=2, nrmse_min=0.0275, nrmse_rel=0, nrmse_mode='mean',
                    pct=0, grad_sigma=1, reg_mode='gaussian', reg_sigma=8,
                    plot=False, dyx0=dyx0, block=True)
        err = self.err(Ab.images_aligned, self.image, self.edge + 1)
        plt.close('all')
        assert np.all(err <= 0.1)
        
    def test_stack_im_rel(self):
        print(self.id())
        A = AlignNR(self.warped_stack, reference=self.image,
                    alpha=40, niter=200, nrmse_min=0, nrmse_rel=1e-5, nrmse_mode='mean',
                    pct=0, grad_sigma=1, reg_mode='gaussian', reg_sigma=8, nthreads=1)
        err = self.err(A.images_aligned, self.image, self.edge + 1)
        plt.close('all')
        assert np.all(err <= 0.1)
    
    def test_single_image_niter(self):
        print(self.id())
        A = AlignNR(self.warped_stack[0], reference=self.image,
                    alpha=40, niter=200, nrmse_min=0, nrmse_rel=0, nrmse_mode='mean',
                    pct=0, grad_sigma=1, reg_mode='gaussian', reg_sigma=8, nthreads=1)
        err = self.err(A.images_aligned, self.image, self.edge + 1)
        plt.close('all')
        assert np.all(err <= 0.1)
    
    def test_single_image_print_stats(self):
        print(self.id())
        A = AlignNR(self.warped_stack[0], reference=self.image,
                    alpha=40, niter=200, nrmse_min=self.nrmse_min, nrmse_rel=0, nrmse_mode='mean',
                    pct=0, grad_sigma=1, reg_mode='gaussian', reg_sigma=8,
                    print_stats=True)
        err = self.err(A.images_aligned, self.image, self.edge + 1)
        plt.close('all')
        assert np.all(err <= 0.1)
    
    def test_single_image_cval(self):
        print(self.id())
        A = AlignNR(self.warped_stack[0], reference=self.image,
                    alpha=40, niter=200, nrmse_min=self.nrmse_min, nrmse_rel=0, nrmse_mode='mean',
                    pct=0, grad_sigma=1, reg_mode='gaussian', reg_sigma=8,
                    cval=0)
        err = self.err(A.images_aligned, self.image, self.edge + 1)
        plt.close('all')
        assert np.all(err <= 0.1)
    
    def test_single_image_pct(self):
        print(self.id())
        A = AlignNR(self.warped_stack[0], reference=self.image,
                    alpha=40, niter=200, nrmse_min=self.nrmse_min, nrmse_rel=0, nrmse_mode='mean',
                    pct=1, grad_sigma=1, reg_mode='gaussian', reg_sigma=8,
                    cval=0)
        err = self.err(A.images_aligned, self.image, self.edge + 1)
        plt.close('all')
        assert np.all(err <= 0.1)
        
        A = AlignNR(self.warped_stack[0], reference=self.image,
                    alpha=40, niter=200, nrmse_min=self.nrmse_min, nrmse_rel=0, nrmse_mode='mean',
                    pct=None, grad_sigma=1, reg_mode='gaussian', reg_sigma=8,
                    cval=0)
        err = self.err(A.images_aligned, self.image, self.edge + 1)
        plt.close('all')
        assert np.all(err <= 0.1)
    
    def test_single_image_ishift(self):
        print(self.id())
        A = AlignNR(self.warped_stack[0], reference=self.image,
                    alpha=40, niter=200, nrmse_min=self.nrmse_min, nrmse_rel=0, nrmse_mode='mean',
                    pct=0, ishift=True, grad_sigma=1, reg_mode='gaussian', reg_sigma=8,
                    cval=0)
        err = self.err(A.images_aligned, self.image, self.edge + 1)
        #print(err)
        plt.close('all')
        #assert np.all(err <= 0.2) # source images don't have distortion based intensity modulations
        
        print(self.id())
        A = AlignNR(self.warped_stack[0], reference=self.image,
                    alpha=40, niter=200, nrmse_min=self.nrmse_min, nrmse_rel=0, nrmse_mode='mean',
                    pct=None, ishift=True, grad_sigma=1, reg_mode='gaussian', reg_sigma=8,
                    cval=0)
        err = self.err(A.images_aligned, self.image, self.edge + 1)
        #print(err)
        plt.close('all')
        #assert np.all(err <= 0.2) # source images don't have distortion based intensity modulations
    
    def test_stack_im_nrmse_modes(self):
        print(self.id())
        
        A = AlignNR(self.warped_stack, reference=self.image,
                    alpha=40, niter=200, nrmse_min=self.nrmse_min+0.02, nrmse_rel=0, nrmse_mode='median',
                    pct=0, grad_sigma=1, reg_mode='gaussian', reg_sigma=8, plot=False, block=True)
        err = self.err(A.images_aligned, self.image, self.edge + 1)
        plt.close('all')
        assert np.all(err <= 0.1)
    
        A = AlignNR(self.warped_stack, reference=self.image,
                    alpha=40, niter=200, nrmse_min=self.nrmse_min+0.02, nrmse_rel=0, nrmse_mode='max',
                    pct=0, grad_sigma=1, reg_mode='gaussian', reg_sigma=8, plot=False, block=True)
        err = self.err(A.images_aligned, self.image, self.edge + 1)
        plt.close('all')
        assert np.all(err <= 0.1)
        
        A = AlignNR(self.warped_stack, reference=self.image,
                    alpha=40, niter=200, nrmse_min=self.nrmse_min+0.02, nrmse_rel=0, nrmse_mode='quad',
                    pct=0, grad_sigma=1, reg_mode='gaussian', reg_sigma=8, plot=False, block=True)
        err = self.err(A.images_aligned, self.image, self.edge + 1)
        plt.close('all')
        assert np.all(err <= 0.1)
    
    def non_block(self):
        A = AlignNR(self.warped_stack[0], reference=self.image,
                    alpha=40, niter=200, nrmse_min=self.nrmse_min, nrmse_rel=0, nrmse_mode='mean',
                    pct=1, grad_sigma=1, reg_mode='gaussian', reg_sigma=8,
                    cval=0, plot=False, block=False)
        
        err = self.err(A.images_aligned, self.image, self.edge + 1)
        plt.close('all')
        assert np.all(err <= 0.1)
        
    def test_single_image_non_block(self):
        print(self.id())
        self.assertRaises(AttributeError, self.non_block)
    
    def test_single_image_non_block_interupt(self):
        print(self.id())
        A = AlignNR(self.warped_stack[0], reference=self.image,
                    alpha=40, niter=200, nrmse_min=self.nrmse_min, nrmse_rel=0, nrmse_mode='mean',
                    pct=1, grad_sigma=1, reg_mode='gaussian', reg_sigma=8,
                    cval=0, plot=False, block=False)
        inter = A.interrupt
        assert inter == False
        A.interrupt = True
    
    def test_stack_im_reg_modes_1func(self):
        print(self.id())
        
        from scipy.ndimage import map_coordinates, gaussian_filter, gaussian_filter1d
        def reg(d):
            return gaussian_filter(d, (20, 1))
        reg_mode = reg
        
        A = AlignNR(self.warped_stack, reference=self.image,
                    alpha=40, niter=200, nrmse_min=self.nrmse_min+0.01, nrmse_rel=0, nrmse_mode='median',
                    pct=0, grad_sigma=1, reg_mode=reg_mode, reg_sigma=8, plot=False, block=True)
        err = self.err(A.images_aligned, self.image, self.edge + 1)
        plt.close('all')
        assert np.all(err <= 0.1)
    
    def test_stack_im_reg_modes_1func_kwd_args(self):
        print(self.id())
        
        from scipy.ndimage import map_coordinates, gaussian_filter, gaussian_filter1d
        def reg(d, axis):
            return gaussian_filter(d, axis)
        reg_mode = reg
        reg_kwargs = {'axis':(20, 1)}
        
        A = AlignNR(self.warped_stack, reference=self.image,
                    alpha=40, niter=200, nrmse_min=self.nrmse_min+0.01, nrmse_rel=0, nrmse_mode='median',
                    pct=0, grad_sigma=1, reg_mode=reg_mode, reg_kwargs=reg_kwargs, 
                    reg_sigma=8, plot=False, block=True)
        err = self.err(A.images_aligned, self.image, self.edge + 1)
        plt.close('all')
        assert np.all(err <= 0.1)
    
    def test_stack_im_reg_modes_2func(self):
        print(self.id())
        
        from scipy.ndimage import map_coordinates, gaussian_filter, gaussian_filter1d
        def regx(dx):
            return gaussian_filter(dx, (20, 1))
        def regy(dy):
            dy = gaussian_filter(dy, (20, 1))
            return dy
        reg_mode = [regy, regx]
        
        A = AlignNR(self.warped_stack, reference=self.image,
                    alpha=40, niter=200, nrmse_min=self.nrmse_min+0.01, nrmse_rel=0, nrmse_mode='median',
                    pct=0, grad_sigma=1, reg_mode=reg_mode, reg_sigma=8, plot=False, block=True)
        err = self.err(A.images_aligned, self.image, self.edge + 1)
        plt.close('all')
        assert np.all(err <= 0.1)
    
    def test_single_image_grad_sigma(self):
        print(self.id())
        A = AlignNR(self.warped_stack[0], reference=self.image,
                    alpha=40, niter=200, nrmse_min=self.nrmse_min, nrmse_rel=0, nrmse_mode='mean',
                    pct=0, grad_sigma=0, reg_mode='gaussian', reg_sigma=8, nthreads=1)
        err = self.err(A.images_aligned, self.image, self.edge + 1)
        plt.close('all')
        assert np.all(err <= 0.1)
    
    def test_single_image_warp(self):
        print(self.id())
        wim = self.warped_stack[0]
        rim = self.image
        A = AlignNR(wim, reference=rim,
                    alpha=40, niter=200, nrmse_min=self.nrmse_min, nrmse_rel=0, nrmse_mode='mean',
                    pct=0, grad_sigma=1, reg_mode='gaussian', reg_sigma=8, nthreads=1)
        err = self.err(A.images_aligned, self.image, self.edge + 1)
        plt.close('all')
        assert np.all(err <= 0.1)
        
        err = self.err(A.images_aligned, A.warp(wim, A.dyx), self.edge + 1)
        assert np.all(err <= 0.1)
        
        err = self.err(A.images_aligned, A.warp(wim), self.edge + 1)
        assert np.all(err <= 0.1)
    
    def test_stack_im_single_multithread(self):
        print(self.id())
        A = AlignNR(self.warped_stack, reference=self.image,
                    alpha=40, niter=200, nrmse_min=self.nrmse_min+0.02, nrmse_rel=0, nrmse_mode='mean',
                    pct=0, grad_sigma=1, reg_mode='gaussian', reg_sigma=8, nthreads=None)
        err = self.err(A.images_aligned, self.image, self.edge + 1)
        plt.close('all')
        assert np.all(err <= 0.1)
    
    @classmethod
    def tearDownClass(self):
        pass
    
if __name__ == '__main__':
    unittest.main()

