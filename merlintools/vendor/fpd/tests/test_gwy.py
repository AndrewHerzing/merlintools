
import unittest
import numpy as np
import tempfile, os

import fpd


class TestGWY(unittest.TestCase):
    def setUp(self):
        self.data = np.random.random((10, 20))
        self.sy, self.sx = self.data.shape
        self.tmp_d = tempfile.mkdtemp(prefix='fpd-')
        
    def test_write(self):
        print(self.id())
        filename = fpd.gwy.writeGSF(filename=None, data=self.data, XReal=1.0*self.sx, 
                                    YReal=1.0*self.sy, Title=None, XYUnits='m', 
                                    ZUnits=None, open_file=False)        
        
        filename = os.path.join(self.tmp_d, 'test')
        filename = fpd.gwy.writeGSF(filename=filename, data=self.data, XReal=1.0*self.sx, 
                                    YReal=1.0*self.sy, Title=None, XYUnits='m', 
                                    ZUnits=None, open_file=False)

    def test_write_read(self):
        print(self.id())
        filename = fpd.gwy.writeGSF(filename=None, data=self.data, XReal=1.0*self.sx, 
                                    YReal=1.0*self.sy, Title=None, XYUnits='m', 
                                    ZUnits=None, open_file=False)
        
        rtn = fpd.gwy.readGSF(filename=filename)
        self.assertTrue(np.allclose(rtn[0], self.data))
        
if __name__ == '__main__':
    unittest.main()


