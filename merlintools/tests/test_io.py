import merlintools
import os
import numpy as np
from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D

merlin_path = os.path.dirname(merlintools.__file__)


class TestReadMIB:
    def test_load_mib_with_dm(self):
        dmfilename = os.path.join(merlin_path, "tests",
                                  "test_data", "HAADF.dm3")
        mibfile = os.path.join(merlin_path, "tests", "test_data", "merlin.mib")
        hdrfile = os.path.join(merlin_path, "tests", "test_data", "merlin.hdr")
        s = merlintools.io.get_merlin_data(mibfile, hdrfile, dmfilename,
                                           use_fpd=False, show_progressbar=False)
        assert type(s) is ElectronDiffraction2D
        assert s.axes_manager.signal_shape == (256, 256)
        assert s.axes_manager.navigation_shape == (20, 20)
    
    def test_load_mib_with_dm_fpd(self):
        dmfilename = os.path.join(merlin_path, "tests",
                                  "test_data", "HAADF.dm3")
        mibfile = os.path.join(merlin_path, "tests", "test_data", "merlin.mib")
        hdrfile = os.path.join(merlin_path, "tests", "test_data", "merlin.hdr")
        s = merlintools.io.get_merlin_data(mibfile, hdrfile, dmfilename,
                                           use_fpd=True, show_progressbar=False)
        assert type(s) is ElectronDiffraction2D
        assert s.axes_manager.signal_shape == (256, 256)
        assert s.axes_manager.navigation_shape == (20, 20)

    def test_load_mib_no_dm(self):
        mibfile = os.path.join(merlin_path, "tests", "test_data", "merlin.mib")
        hdrfile = os.path.join(merlin_path, "tests", "test_data", "merlin.hdr")
        s = merlintools.io.get_merlin_data(mibfile, hdrfile,
                                           use_fpd=False, show_progressbar=False)
        assert type(s) is ElectronDiffraction2D
        assert s.axes_manager.signal_shape == (256, 256)
        assert s.axes_manager.navigation_shape == (405, 1)
    
    def test_load_mib_no_dm_fpd(self):
        mibfile = os.path.join(merlin_path, "tests", "test_data", "merlin.mib")
        hdrfile = os.path.join(merlin_path, "tests", "test_data", "merlin.hdr")
        s = merlintools.io.get_merlin_data(mibfile, hdrfile,
                                           use_fpd=True, show_progressbar=False)
        assert type(s) is ElectronDiffraction2D
        assert s.axes_manager.signal_shape == (256, 256)
        assert s.axes_manager.navigation_shape == (405, 1)
    
    def test_load_mib_manual_scan(self):

        mibfile = os.path.join(merlin_path, "tests", "test_data", "merlin.mib")
        hdrfile = os.path.join(merlin_path, "tests", "test_data", "merlin.hdr")
        s = merlintools.io.get_merlin_data(mibfile, hdrfile, scanX=[20,'x','nm'],
                                           scanY=[20,'y','nm'], use_fpd=False,
                                           show_progressbar=False)
        assert type(s) is ElectronDiffraction2D
        assert s.axes_manager.signal_shape == (256, 256)
        assert s.axes_manager.navigation_shape == (20, 20)
    
    def test_load_mib_manual_scan_fpd(self):
        mibfile = os.path.join(merlin_path, "tests", "test_data", "merlin.mib")
        hdrfile = os.path.join(merlin_path, "tests", "test_data", "merlin.hdr")
        s = merlintools.io.get_merlin_data(mibfile, hdrfile, scanX=[20,'x','nm'],
                                           scanY=[20,'y','nm'], use_fpd=True,
                                           show_progressbar=False)
        assert type(s) is ElectronDiffraction2D
        assert s.axes_manager.signal_shape == (256, 256)
        assert s.axes_manager.navigation_shape == (20, 20)

    class TestHeaderParsing:

        def test_hdr_parser(self):
            hdrfile = os.path.join(merlin_path, "tests", "test_data", "merlin.hdr")
            header = merlintools.io.parse_hdr(hdrfile)
            assert type(header) is dict
            assert 'TimeStamp' in header.keys()
            assert 'SoftwareVersion' in header.keys()
            assert np.int(header['TotalFrames']) == 11007
        
        def test_mib_parser(self):
            mibfile = os.path.join(merlin_path, "tests", "test_data", "merlin.mib")
            header = merlintools.io.parse_mib_header(mibfile)
            assert type(header) is dict
            assert 'HeaderID' in header.keys()
            assert 'ExtCounterDepth' in header.keys()
            assert header['DataOffset'] == 384