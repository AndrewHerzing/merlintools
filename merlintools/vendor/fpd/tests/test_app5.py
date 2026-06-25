
import unittest
import os
import zipfile
import tempfile
import shutil
import matplotlib.pylab as plt
plt.ion()
#import numpy as np
#import h5py

from fpd.fpd_io import topspin_app5_to_hdf5
from fpd.fpd_file import DataBrowser


class TestFile(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tmp_d = tempfile.mkdtemp(prefix='fpd-')
        self.bp = os.path.dirname(__file__)
        
        # multiple images per file
        z_fn = os.path.join(self.bp, 'file_datasets', 'app5_test_data.zip')
        with zipfile.ZipFile(z_fn, 'r') as z:
            z.extractall(path=self.tmp_d)
            z_fns = z.namelist()
        self.z_app5_fns = [x for x in z_fns if x.split('.')[-1].lower() in ['app5']]
        self.z_app5_fns.sort()
    
    def test_area(self):
        print(self.id())
        # 2d scan of 10 y by 12 x
        
        fns = [t for t in self.z_app5_fns if 'area' in t.lower()]
        
        ofns = []
        for fn in fns:
            h5fn = topspin_app5_to_hdf5(os.path.join(self.tmp_d, fn), h5fn=None, TEM='ARM200cF', prec_angle=0.0, chunk_dim_size=6, compression_opts=4, progress_bar=False)
            ofns.append(h5fn)
        
        B = DataBrowser(os.path.join(self.tmp_d, ofns[0]), fpd_check=True)

    @classmethod
    def tearDownClass(self):
        #print(self.tmp_d)
        shutil.rmtree(self.tmp_d)
        #pass


if __name__ == '__main__':
    unittest.main()

