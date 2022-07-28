# -*- coding: utf-8 -*-
#
# This file is part of MerlinTools

"""
test_io module for MerlinTools package.

@author: Andrew Herzing
"""

import merlintools
import os
import glob
import numpy as np
from tempfile import TemporaryDirectory

merlin_path = os.path.dirname(merlintools.__file__)

class TestPreprocessing:
    """Test MIB conversion functionality."""
    def test_not_full_aligned(self):
        test_data_dir = os.path.join(merlin_path, "tests", "test_data")
        temp_dir = TemporaryDirectory()
        merlintools.preprocessing.preprocess(test_data_dir, temp_dir.name)
        h5file = glob.glob(temp_dir.name + '/**/*.hdf5', recursive=True)[0]
        h5 = merlintools.io.read_h5_results(h5file)
        temp_dir.cleanup()
        assert 'radial_profile' in h5.keys()

    def test_not_full_aligned(self):
        test_data_dir = os.path.join(merlin_path, "tests", "test_data")
        temp_dir = TemporaryDirectory()
        merlintools.preprocessing.preprocess(test_data_dir, temp_dir.name, full_align=True)
        h5file = glob.glob(temp_dir.name + '/**/*.hdf5', recursive=True)[0]
        h5 = merlintools.io.read_h5_results(h5file)
        temp_dir.cleanup()
        assert type(h5['aligned']) is np.ndarray