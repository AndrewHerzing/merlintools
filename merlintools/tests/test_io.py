# -*- coding: utf-8 -*-
#
# This file is part of MerlinTools

"""
test_io module for MerlinTools package.

@author: Andrew Herzing
"""

import merlintools
import os
import numpy as np
from hyperspy.signals import Signal2D
import glob

merlin_path = os.path.dirname(merlintools.__file__)

class TestMIBConversion:
    """Test MIB conversion functionality."""
    def test_no_scan_file(self):
        datadir = os.path.join(merlin_path, "tests", "test_data", "No-Scan")
        mib = glob.glob(datadir + '/*.mib')
        hdr = glob.glob(datadir + '/*.hdr')
        emi = glob.glob(datadir + '/*.emi')
        ser = glob.glob(datadir + '/*.ser')
        dm = glob.glob(datadir + '/*.dm*')
        mb = merlintools.preprocessing.get_merlin_binary(mib, hdr, emi, ser, dm)
        assert mb.shape == (1,256,256,256)
    
    def test_dm_scan_missing_frames(self):
        datadir = os.path.join(merlin_path, "tests", "test_data", "DM-Scan_MissingFrames")
        mib = glob.glob(datadir + '/*.mib')
        hdr = glob.glob(datadir + '/*.hdr')
        emi = glob.glob(datadir + '/*.emi')
        ser = glob.glob(datadir + '/*.ser')
        dm = glob.glob(datadir + '/*.dm*')
        mb = merlintools.preprocessing.get_merlin_binary(mib, hdr, emi, ser, dm)
        assert mb.shape == (16,16,256,256)
    
    def test_dm_scan_extra_frames(self):
        datadir = os.path.join(merlin_path, "tests", "test_data", "DM-Scan_ExtraFrames")
        mib = glob.glob(datadir + '/*.mib')
        hdr = glob.glob(datadir + '/*.hdr')
        emi = glob.glob(datadir + '/*.emi')
        ser = glob.glob(datadir + '/*.ser')
        dm = glob.glob(datadir + '/*.dm*')
        mb = merlintools.preprocessing.get_merlin_binary(mib, hdr, emi, ser, dm)
        assert mb.shape == (16,16,256,256)
    
    def test_tia_scan(self):
        datadir = os.path.join(merlin_path, "tests", "test_data", "TIA-Scan")
        mib = glob.glob(datadir + '/*.mib')
        hdr = glob.glob(datadir + '/*.hdr')
        emi = glob.glob(datadir + '/*.emi')
        ser = glob.glob(datadir + '/*.ser')
        dm = glob.glob(datadir + '/*.dm*')
        mb = merlintools.preprocessing.get_merlin_binary(mib, hdr, emi, ser, dm)
        assert mb.shape == (16,16,256,256)

class TestHeaderParsing:
    """Test header parsing functionality."""

    def test_hdr_parser(self):
        """Parse HDR file."""
        hdrfile = os.path.join(merlin_path, "tests", "test_data",
                               "TestData_12bit", "merlin.hdr")
        header = merlintools.io.parse_hdr(hdrfile)
        assert type(header) is dict
        assert 'TimeStamp' in header.keys()
        assert 'SoftwareVersion' in header.keys()
        assert np.int(header['TotalFrames']) == 11007

    def test_mib_parser(self):
        """Parse MIB file."""
        mibfile = os.path.join(merlin_path, "tests", "test_data",
                               "TestData_12bit", "merlin.mib")
        header = merlintools.io.parse_mib_header(mibfile)
        assert type(header) is dict
        assert 'HeaderID' in header.keys()
        assert 'ExtCounterDepth' in header.keys()
        assert header['DataOffset'] == 384
