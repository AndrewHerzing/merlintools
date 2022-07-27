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

merlin_path = os.path.dirname(merlintools.__file__)

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
