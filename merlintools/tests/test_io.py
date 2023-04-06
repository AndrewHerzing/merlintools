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
import glob
from tempfile import TemporaryDirectory
import h5py
import fpd.fpd_file as fpdf
import pandas

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
        assert mb.shape == (1, 256, 256, 256)

    def test_dm_scan_missing_frames(self):
        datadir = os.path.join(merlin_path, "tests", "test_data", "DM-Scan_MissingFrames")
        mib = glob.glob(datadir + '/*.mib')
        hdr = glob.glob(datadir + '/*.hdr')
        emi = glob.glob(datadir + '/*.emi')
        ser = glob.glob(datadir + '/*.ser')
        dm = glob.glob(datadir + '/*.dm*')
        mb = merlintools.preprocessing.get_merlin_binary(mib, hdr, emi, ser, dm)
        assert mb.shape == (16, 16, 256, 256)

    def test_dm_scan_extra_frames(self):
        datadir = os.path.join(merlin_path, "tests", "test_data", "DM-Scan_ExtraFrames")
        mib = glob.glob(datadir + '/*.mib')
        hdr = glob.glob(datadir + '/*.hdr')
        emi = glob.glob(datadir + '/*.emi')
        ser = glob.glob(datadir + '/*.ser')
        dm = glob.glob(datadir + '/*.dm*')
        mb = merlintools.preprocessing.get_merlin_binary(mib, hdr, emi, ser, dm)
        assert mb.shape == (16, 16, 256, 256)

    def test_tia_scan(self):
        datadir = os.path.join(merlin_path, "tests", "test_data", "TIA-Scan")
        mib = glob.glob(datadir + '/*.mib')
        hdr = glob.glob(datadir + '/*.hdr')
        emi = glob.glob(datadir + '/*.emi')
        ser = glob.glob(datadir + '/*.ser')
        dm = glob.glob(datadir + '/*.dm*')
        mb = merlintools.preprocessing.get_merlin_binary(mib, hdr, emi, ser, dm)
        assert mb.shape == (1, 256, 256, 256)


class TestGetMicroscopeParameters:
    def test_tia_scope_parameters_filename(self):
        temp_dir = TemporaryDirectory()
        h5filename = temp_dir.name + "tempfile.hdf5"
        datadir = os.path.join(merlin_path, "tests", "test_data", "TIA-Scan")
        mib = glob.glob(datadir + '/*.mib')
        hdr = glob.glob(datadir + '/*.hdr')
        emi = glob.glob(datadir + '/*.emi')
        ser = glob.glob(datadir + '/*.ser')
        dm = glob.glob(datadir + '/*.dm*')
        mb = merlintools.preprocessing.get_merlin_binary(mib, hdr, emi, ser, dm)
        mb.write_hdf5(h5filename)
        params = merlintools.io.get_microscope_parameters(h5filename)
        temp_dir.cleanup()
        assert params['HT'] == 200.0

    def test_tia_scope_parameters_h5(self):
        temp_dir = TemporaryDirectory()
        h5filename = temp_dir.name + "tempfile.hdf5"
        datadir = os.path.join(merlin_path, "tests", "test_data", "TIA-Scan")
        mib = glob.glob(datadir + '/*.mib')
        hdr = glob.glob(datadir + '/*.hdr')
        emi = glob.glob(datadir + '/*.emi')
        ser = glob.glob(datadir + '/*.ser')
        dm = glob.glob(datadir + '/*.dm*')
        mb = merlintools.preprocessing.get_merlin_binary(mib, hdr, emi, ser, dm)
        mb.write_hdf5(h5filename)
        with h5py.File(h5filename, 'r') as h5:
            params = merlintools.io.get_microscope_parameters(h5)
        temp_dir.cleanup()
        assert params['HT'] == 200.0

    def test_tia_scope_parameters_nt(self):
        temp_dir = TemporaryDirectory()
        h5filename = temp_dir.name + "tempfile.hdf5"
        datadir = os.path.join(merlin_path, "tests", "test_data", "TIA-Scan")
        mib = glob.glob(datadir + '/*.mib')
        hdr = glob.glob(datadir + '/*.hdr')
        emi = glob.glob(datadir + '/*.emi')
        ser = glob.glob(datadir + '/*.ser')
        dm = glob.glob(datadir + '/*.dm*')
        mb = merlintools.preprocessing.get_merlin_binary(mib, hdr, emi, ser, dm)
        mb.write_hdf5(h5filename)
        nt = fpdf.fpd_to_tuple(h5filename)
        params = merlintools.io.get_microscope_parameters(nt)
        temp_dir.cleanup()
        assert params['HT'] == 200.0

    def test_dm_scope_parameters_filename(self):
        temp_dir = TemporaryDirectory()
        h5filename = temp_dir.name + "tempfile.hdf5"
        datadir = os.path.join(merlin_path, "tests", "test_data", "DM-Scan_ExtraFrames")
        mib = glob.glob(datadir + '/*.mib')
        hdr = glob.glob(datadir + '/*.hdr')
        emi = glob.glob(datadir + '/*.emi')
        ser = glob.glob(datadir + '/*.ser')
        dm = glob.glob(datadir + '/*.dm*')
        mb = merlintools.preprocessing.get_merlin_binary(mib, hdr, emi, ser, dm)
        mb.write_hdf5(h5filename)
        params = merlintools.io.get_microscope_parameters(h5filename)
        temp_dir.cleanup()
        assert params['HT'] == 200.0

    def test_dm_scope_parameters_h5(self):
        temp_dir = TemporaryDirectory()
        h5filename = temp_dir.name + "tempfile.hdf5"
        datadir = os.path.join(merlin_path, "tests", "test_data", "DM-Scan_ExtraFrames")
        mib = glob.glob(datadir + '/*.mib')
        hdr = glob.glob(datadir + '/*.hdr')
        emi = glob.glob(datadir + '/*.emi')
        ser = glob.glob(datadir + '/*.ser')
        dm = glob.glob(datadir + '/*.dm*')
        mb = merlintools.preprocessing.get_merlin_binary(mib, hdr, emi, ser, dm)
        mb.write_hdf5(h5filename)
        with h5py.File(h5filename, 'r') as h5:
            params = merlintools.io.get_microscope_parameters(h5)
        temp_dir.cleanup()
        assert params['HT'] == 200.0

    def test_dm_scope_parameters_nt(self):
        temp_dir = TemporaryDirectory()
        h5filename = temp_dir.name + "tempfile.hdf5"
        datadir = os.path.join(merlin_path, "tests", "test_data", "DM-Scan_ExtraFrames")
        mib = glob.glob(datadir + '/*.mib')
        hdr = glob.glob(datadir + '/*.hdr')
        emi = glob.glob(datadir + '/*.emi')
        ser = glob.glob(datadir + '/*.ser')
        dm = glob.glob(datadir + '/*.dm*')
        mb = merlintools.preprocessing.get_merlin_binary(mib, hdr, emi, ser, dm)
        mb.write_hdf5(h5filename)
        nt = fpdf.fpd_to_tuple(h5filename)
        params = merlintools.io.get_microscope_parameters(nt)
        temp_dir.cleanup()
        assert params['HT'] == 200.0


class TestGetMerlinParameters:
    def test_get_merlin_parameters_string(self):
        h5file = os.path.join(merlin_path, "tests", "test_data",
                              "FPD_HDF5.hdf5")
        params = merlintools.io.get_merlin_parameters(h5file)
        assert params['Detector shape'] == [256, 256]

    def test_get_merlin_parameters_h5(self):
        h5file = os.path.join(merlin_path, "tests", "test_data",
                              "FPD_HDF5.hdf5")
        with h5py.File(h5file, 'r') as h5:
            params = merlintools.io.get_merlin_parameters(h5)
        assert params['Detector shape'] == [256, 256]


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


class TestReadSingleMIB:

    def test_load_single_mib(self):
        mibfile = os.path.join(merlin_path, "tests", "test_data",
                               "SingleMIB.mib")
        mibfile = merlintools.io.read_single_mib(mibfile)
        assert isinstance(mibfile, np.ndarray)


class TestOtherIOFunctions:
    def test_get_experimental_parameters(self):
        rootpath = os.path.join(merlin_path, "tests", "test_data", "Archive-Directory")
        h5files = glob.glob(rootpath + "/*/*.hdf5")
        df = merlintools.io.get_experimental_parameters(h5files)
        assert isinstance(df, pandas.DataFrame)
        assert df['Beam Energy'][0] == 200.0

    def test_get_axes_dict(self):
        h5filename = os.path.join(merlin_path, "tests", "test_data", "FPD_HDF5.hdf5")
        nt = fpdf.fpd_to_tuple(h5filename)
        axes_dict = merlintools.io.get_spatial_axes_dict(nt)
        assert isinstance(axes_dict, dict)
        assert axes_dict['axis-0']['units'] == 'nm'
        assert {"axis-0", "axis-1"} <= axes_dict.keys()
