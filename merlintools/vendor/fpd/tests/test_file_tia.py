
import unittest
import os
import tarfile
import tempfile
import shutil
import filecmp

import matplotlib.pylab as plt
plt.ion()

import fpd.fpd_file as fpdf

class TestFile(unittest.TestCase):
    @classmethod
    def setUpClass(self):        
        self.tmp_d = tempfile.mkdtemp(prefix='fpd-')
        self.bp = os.path.dirname(__file__)
        
        # One 16x16 scan
        a_fn = os.path.join(self.bp, 'file_datasets', 'TIA_Scan_Test_Data.tar.xz')
    
        with tarfile.open(a_fn, 'r') as a:
            a.extractall(path=self.tmp_d)
            a_fns = [i.name for i in a.getmembers() if i.isfile()]
        a_fns = [os.path.join(self.tmp_d, t) for t in a_fns]
        
        self.mib_fn = [t for t in a_fns if t.endswith('.mib')][0]
        self.hdr_fn = [t for t in a_fns if t.endswith('.hdr')][0]
        self.ser_fn = [t for t in a_fns if t.endswith('.ser')][0]
        self.emi_fn = [t for t in a_fns if t.endswith('.emi')][0]
        
        self.h5fn = os.path.join(self.tmp_d, 'TIA.hdf5')
    
    def test_1_convert(self):
        print(self.id())        
        M = fpdf.MerlinBinary(binfns=self.mib_fn,
                              hdrfn=self.hdr_fn,
                              dmfns=[],
                              tiafns=[self.ser_fn],
                              ds_start_skip=0,
                              row_end_skip=0)
        M.write_hdf5(self.h5fn, progress_bar=False)
    
    def test_2_read(self):
        print(self.id())
        with fpdf.fpd_nt_cm(self.h5fn) as fpd_nt:
            d = fpd_nt.fpd_data.data[()]
            
            # all TIA data is currently only 2D
            # postscript numeral is separate files
            tia_nt = fpd_nt.TIA0
            tia = tia_nt.data[()]
            assert(tia.ndim == 2)
            assert(tia_nt.dim1.units == 'nm')
    
    def test_3_browse(self):
        print(self.id())
        B = fpdf.DataBrowser(self.h5fn, fpd_check=True)
    
    def test_4_tools(self):
        print(self.id())
        
        rtn = fpdf.hdf5_tia_tags_to_dict(self.h5fn, fpd_check=True)
        tagds, dm_group_paths, dm_filenames = rtn
        assert dm_group_paths[0] == 'fpd_expt/TIA0'
        ser_name = os.path.basename(self.ser_fn)
        assert dm_filenames[0] == ser_name
    
    def test_5_tia_extract(self):
        print(self.id())
        
        out_filenames = [os.path.join(self.tmp_d, t.split('.')[0]+'_extracted') for t in [self.ser_fn]]
        fpdf.hdf5_tia_to_bin(self.h5fn,
                             tiafns=out_filenames,
                             fpd_check=True,
                             ow=True)
        
        # binary comparison
        assert filecmp.cmp(out_filenames[0] + '.ser', self.ser_fn, shallow=False)
        
    @classmethod
    def tearDownClass(self):
        #print(self.tmp_d)
        shutil.rmtree(self.tmp_d)
        #pass


if __name__ == '__main__':
    unittest.main()

