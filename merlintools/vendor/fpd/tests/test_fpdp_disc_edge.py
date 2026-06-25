
import unittest
import numpy as np
import matplotlib.pylab as plt

import fpd
import fpd.fpd_processing as fpdp
import fpd.fpd_file as fpdf


class TestFPDP_disc_edge(unittest.TestCase):    
    def test_disc_edge_properties(self):
        print(self.id())
        sigma = 5.0
        
        im = fpd.synthetic_data.disk_image(intensity=1024, radius=32, sigma=sigma, size=256, noise=False)
        cyx, r = fpd.fpd_processing.find_circ_centre(im, 2, (22, int(256/2.0), 1), spf=1, plot=False)
        
        dp = fpdp.disc_edge_properties(im, sigma=6, cyx=cyx, r=r, plot=False)
        # sigma_wt_avg, sigma_wt_std, sigma_vals, sigma_avg, sigma_std, sigma_stds, r_vals, r_stds, d_vals, d_avg, d_std)
        sigma_wt_avg = dp.sigma_wt_avg
        self.assertTrue(np.abs(sigma_wt_avg/sigma - 1) <= 0.02)
        
        dp = fpdp.disc_edge_properties(im, sigma=6, cyx=cyx, r=r, n_angles=91, plot=False)
        
        dp = fpdp.disc_edge_properties(im, sigma=6, plot=True)
        plt.close('all')

if __name__ == '__main__':
    unittest.main()


