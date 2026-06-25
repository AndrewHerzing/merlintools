# file module version (separate from fpd version)
__version__ = "0.2.0"
_min_version = "0.1.0"

import mmap
import h5py
import numpy as np
from collections import OrderedDict, namedtuple
from collections.abc import MutableMapping
import time
from dateutil.parser import parse
import codecs
import re
import os
from functools import partial
import inspect
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging
from packaging.version import Version
import io
from tqdm import tqdm
from itertools import product
import sys
from scipy.interpolate import griddata
from scipy.interpolate import CloughTocher2DInterpolator
import copy
import numbers
import psutil, platform
import hdf5plugin
import contextlib

from . import __version__ as fpd_pkg_version


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
# _handler.setFormatter(logging.Formatter(fmt='%(levelname)s:%(name)s:%(message)s'))
_handler.setFormatter(logging.Formatter(fmt="%(levelname)s: %(message)s"))
_logger.addHandler(_handler)


_mpl_non_adjust = False
import matplotlib as mpl

_mplv = mpl.__version__
from packaging.version import Version

if Version(_mplv) >= Version("2.2.0"):
    _mpl_non_adjust = True


def _flatten(d, parent_key="", sep="_", include_empty=False):
    """
    Flatten a dictionary or other nested structure into a dictionary.

    Parameters
    ----------
    d : instance of MutableMapping
        Object to be flattened.
    parent_key : string
        String to a which all keys are appended.
    sep : string
        String seperating flattened keys.
    include_empty : bool
        boolean controlling if empty keys are retained.

    Returns
    -------
    Dictionary of flattened structure.

    """

    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(_flatten(v, new_key, sep).items())
            if include_empty:
                items.append((new_key, v))
        else:
            items.append((new_key, v))
    return dict(items)


def _unflatten(fd, sep="_"):
    """
    Unflatten a dictionary or other nested structure into a nested
    dictionary.

    Parameters
    ----------
    fd : dictionary
        Flattend dictionary.
    sep : string
        String at which keys are split.

    Returns
    -------
    Nested dictionary of unflattened structure.

    """

    ud = dict()
    for k, v in fd.items():
        new_ks = k.split(sep)
        d = ud
        for new_k in new_ks[:-1]:
            if new_k not in d:
                d[new_k] = dict()
            d = d[new_k]
        d[new_ks[-1]] = v
    return ud


class Merlin_hdr_file_parser:
    """
    Converts Merlin header file into ordered key : value dictionary,
    patching the file formatting and extracting specific parameters
    along the way.

    The keys exclude any lists of possible values (in parentheses).
    Strings are converted to lower case and words are separated by
    underscores.

    Versions handled are:
        0   0.76                format differs from documentation.
        1   0.65.8 - 0.67.0.6   format matches documentation.

    Additional keys in 1 are:
        humidity, dac_file, gap_fill_mode

    Attributes
    ----------
    fn : string
        Merlin header filename.
    strict : bool
        boolean of whether strict version check control took place.
    fmt_ver : integer
        Internal version numbering of known systems.
    hs : string
        Header file contents as a string.
    hd : ordered dictionary
        Dictionary of keys and values parsed from header file.
    CM : bool
        boolean describing if colour mode is in operation.
    AC : integer
        Number of active counters.
    mode : string
        String describing detector's overall mode of operation.

    Methods
    -------
    _print_testing_notes

    """

    def __init__(self, fn, strict=True):
        """
        Parameters
        ----------
        fn : string
            Merlin header filename.
        strict : bool
            boolean controlling if strict version check takes place.

        Notes
        -----
        If `strict` is False, the most recent version will be tried.

        """

        self._ver_str_min = "0.65.8"
        self._ver_str_max = "0.77.0.16"

        self.fn = fn
        self.strict = strict

        self._read_file_and_version()
        self._parse_header()
        self._extract_data()

    def _read_file_and_version(self):
        with codecs.open(self.fn, "r", encoding="iso 8859-15") as f:
            self.hs = f.read().strip()

        # Parse versions
        p = "Software Version:[\s]*([\d\.]+)"
        s = re.search(p, self.hs)
        try:
            self._ver_str = s.groups()[0]
        except:
            # This should never happen in newer versions of the file.
            m = "hdr version string not found!"
            _logger.error(m)
            raise Exception(m)

        p = "Readout System:[\s]*([^\n\r]+)"
        s = re.search(p, self.hs)
        try:
            self._read_str = s.groups()[0]
        except:
            # This should never happen in newer versions of the file.
            m = "hdr readout system string not found!"
            _logger.error(m)
            raise Exception(m)

        # Check versions. Could write as dict if many versions.
        fmt_ver1 = (
            Version(self._ver_str_min)
            <= Version(self._ver_str)
            <= Version(self._ver_str_max)
        )

        fmt_ver0_string = "0.76"
        if self._ver_str == fmt_ver0_string and self._read_str == "Merlin vs 1":
            # 1st hardware / software
            _logger.info("Detected 1st version: '%s'" % (self._ver_str))
            self.fmt_ver = 0
        elif fmt_ver1 and self._read_str == "Merlin Quad":
            _logger.info("Detected 2nd version: '%s'" % (self._ver_str))
            self.fmt_ver = 1
        else:
            m = "hdr software '%s' and readout '%s' combination not understood" % (
                self._ver_str,
                self._read_str,
            )
            dev_m = (
                "Consider sharing test data with developers for support to be added."
            )
            if self.strict:
                _logger.info(
                    "Try adding 'strict=False' to input options to assume the latest known format."
                )
                _logger.info(dev_m)
                _logger.error(m)
                raise Exception(m)
            else:
                self.fmt_ver = 1
                _logger.warning(m)
                _logger.warning(
                    "You've set strict==False, so assuming latest known version. Fingers crossed!"
                )
                _logger.warning(
                    "If it works, check that your data has been converted correctly!"
                )
                _logger.warning(dev_m)

    def _parse_header(self):
        # Convert header string into ordered key : value dictionary.

        # split at header delims: End is optional (it is absent in 0.76)
        p = "^HDR,(.+?)(End)?$"
        s = re.search(p, self.hs, re.S)
        hsp = s.groups()[0].strip()

        # patch header
        # add ':' to end of Sensor Bias line key if needed.
        # It is absent in 0.76 and 0.65.8.
        p = "(Sensor Bias \(V, .?A\))[^:][\s ]*"
        s = re.search(p, hsp)
        if s is not None:
            k = s.groups()[0]
            hsp = hsp.replace(k, k + ":")

        # split into keys and values
        p1 = "[ ]*:[\s ]+"  # matches 'spaces : newlines' for v0.76
        p2 = "[\n\r]+"  # matches newlines for end of lines
        s = re.split(p1 + "|" + p2, hsp)
        ks = s[0::2]
        vs = [v.strip() for v in s[1::2]]

        # Match and remove last parentheses with contents,
        # then strip and swap underscore for space to form keys
        ks_strip = [
            re.sub(r" *\([^)]*\)$", "", k.strip()).replace(" ", "_").lower() for k in ks
        ]
        # make dictionary
        self.hd = OrderedDict(zip(ks_strip, vs))

        # print key vals
        for i, k in enumerate(ks):
            m = "%50s\t%25s\t%s" % (k, ks_strip[i], vs[i])
            _logger.debug(m)

    def _extract_data(self):
        # read parsed dictionary for important parameters
        self.mode = self.hd["chip_mode"]
        # Colour mode activated if CM in one of (SPM, CSM, CM, CSCM)
        self.CM = "CM" in self.hd["chip_mode"]
        # number of active counters (1 or 2)
        if self.fmt_ver == 0:
            AC = self.hd["active_counters"].lower().count("on")
        elif self.fmt_ver == 1:
            AC = len(re.findall("\d", self.hd["active_counters"]))
            if AC == 0 and self.hd["active_counters"].lower() == "alternating":
                AC = 1
        self.AC = AC

    def _print_testing_notes(self):
        m = """
        To see what keys are different:
        
        from merlintools.vendor.fpd import fpd_file as fpdf
        m1 = fpdf.Merlin_hdr_file_parser('./fpd_test_data/bin_257.hdr')
        m2 = fpdf.Merlin_hdr_file_parser('./fpd_test_data/spm100ms.hdr')
        
        # 0.75.4.84
        m3 = fpdf.Merlin_hdr_file_parser('NIST_FPDTestData.hdr', strict=False)
        
        # '0.67.0.6'
        m4 = fpdf.Merlin_hdr_file_parser('007_12bit_32x1.hdr')

        d1 = m1.hd
        d2 = m2.hd
        d3 = m3.hd
        d4 = m4.hd
        
        print( set(d4.keys()) - set(d3.keys()) )
        # set()
        print( set(d3.keys()) - set(d4.keys()) )
        # set()

        print( set(d1.keys()) - set(d2.keys()) )
        #[]
        print( set(d2.keys()) - set(d1.keys()) )
        #[u'humidity', u'dac_file', u'gap_fill_mode']
        
        
        for x,y in d3.items():
            print(x, '\t', y)
        
        time_and_date_stamp      12/18/2020 10:49:15 AM
        chip_id          W509_H5, - , - , -
        chip_type        Medipix 3RX
        assembly_size    1x1
        chip_mode        SPM
        counter_depth    12
        gain     SLGM
        active_counters          Alternating
        thresholds       3.200000E+1,5.110000E+2,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0
        dacs     059,511,000,000,000,000,000,000,100,255,200,125,100,100,098,100,104,050,128,004,255,084,128,135,125,511,511; ; ;
        bpc_file         c:\MERLIN_Quad_Config\W509_H5\W509_H5_SPM.bpc,,,
        dac_file         c:\MERLIN_Quad_Config\W509_H5\W509_H5_SPM_300kV.dacs,,,
        gap_fill_mode    None
        flat_field_file          None
        dead_time_file   Dummy (C:\<NUL>\)
        acquisition_type         Normal
        frames_in_acquisition    265
        frames_per_trigger       265
        trigger_start    Rising Edge
        trigger_stop     Rising Edge
        sensor_bias      115 V
        sensor_polarity          Positive
        temperature      Board Temp 0.000000 Deg C
        humidity         Board Humidity 0.000000
        medipix_clock    120MHz
        readout_system   Merlin Quad
        software_version         0.75.4.84
        
        
        Version 0.77.0.16 is the same as 0.75.4.84.

        for x,y in d1.items():
            print(x, '\t', y)
        
            chip_id                 W113_L6
            chip_type               Medipix3RX
            assembly_size           1 X 1
            chip_mode               SPM
            counter_depth           12B
            gain                    High
            active_counters         On Off
            thresholds              2.700000E+1  3.600000E+0
            dacs                    150,20,0,0,0,0,0,0,100,10,125,125,100,100,80,100,120,50,128,4,255,108,120,160,148,401,401
            bpc_file                C:\MediPix\Config\default\Mask default.bpc
            flat_field_file         Dummy (C:\Temp\Temp.ffc)
            dead_time_file          Dummy (C:\Temp\Temp.dtc)
            acquisition_type        Normal
            frames_in_acquisition   65792
            trigger_start           Rising Edge
            trigger_stop            Internal
            frames_per_trigger      1
            time_and_date_stamp     11/08/201517:55:07
            sensor_bias             90 V
            sensor_polarity         Positive
            temperature             0.000000
            medipix_clock           120MHz
            readout_system          Merlin vs 1
            software_version        0.76


        for x,y in d2.items():
            print(x, '\t', y)
            
            time_and_date_stamp     27/08/2015 17:54:11
            chip_id                 W113_L6,-,-,-
            chip_type               Medipix 3RX
            assembly_size           2x2
            chip_mode               SPM
            counter_depth           12
            gain                    SLGM
            active_counters         Counter 0
            thresholds              0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0
            dacs                    000,000,000,000,000,000,000,000,175,010,200,125,100,100,093,100,099,030,128,004,255,127,128,177,167,000,000
            bpc_file                C:\Merlin_Quad_Config\Default\Default_SPM.bpc,,,
            dac_file                c:\MERLIN_Quad_Config\default\Default_SPM.dacs,,,
            gap_fill_mode           Distribute
            flat_field_file         None
            dead_time_file          Dummy (C:\<NUL>\)
            acquisition_type        Normal
            frames_in_acquisition   1
            frames_per_trigger      1
            trigger_start           Internal
            trigger_stop            Internal
            sensor_bias             15 V
            sensor_polarity         Positive
            temperature             Board Temp 0.000000 Deg C
            humidity                Board Humidity 0.000000
            medipix_clock           120MHz
            readout_system          Merlin Quad
            software_version        0.65.8
        
        
        from merlintools.vendor.fpd import fpd_file as fpdf
        v0p67p0p8 = fpdf.Merlin_hdr_file_parser('test_small_16/test_small_16.hdr', strict=False)
        for x,y in v0p67p0p8.hd.items():
            print(x, '\t', y)
        
            time_and_date_stamp      01/03/2018 15:09:59
            chip_id          W529_E5,-,-,-
            chip_type        Medipix 3RX
            assembly_size    1x1
            chip_mode        SPM
            counter_depth    24
            gain             SLGM
            active_counters  Counter 0
            thresholds       0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0
            dacs             040,511,000,000,000,000,000,000,175,010,200,125,100,100,096,100,093,030,128,004,255,054,128,104,100,511,511
            bpc_file         c:\MERLIN_Quad_Config\W529_E5\W529_E5_SPM.bpc,,,
            dac_file         c:\MERLIN_Quad_Config\W529_E5\W529_E5_SPM.dacs,,,
            gap_fill_mode    Distribute
            flat_field_file  None
            dead_time_file   Dummy (C:\<NUL>\)
            acquisition_type         Normal
            frames_in_acquisition    16
            frames_per_trigger       1
            trigger_start    Internal
            trigger_stop     Internal
            sensor_bias      110 V
            sensor_polarity          Positive
            temperature      Board Temp 0.000000 Deg C
            humidity         Board Humidity 0.000000
            medipix_clock    120MHz
            readout_system   Merlin Quad
            software_version         0.67.0.8
        """
        print(m)


class Merlin_binary_header_parser:
    """
    Parses a Merlin binary header to extract parameters.

    Attributes
    ----------
    hs : string
        Header string, trimmed of padding.
    bitdepth_data : integer
        Data bitdepth: 1, 6, 12, 24.
    bitdepth_bin : integer
        Binary bitdepth (non-raw: 8, 16 or 32, raw: 64).
    params : dict of dicts
        Parameters extracted from header.
        The key is used internally. The value is a dict used for
        dataset creation and has keys:
        'name' : str
            Dataset name.
        'unit' : str
            Data unit.
        'data' : obj
            Object data to be stored.
        'isim' : bool
            True is stored with image tag.
    raw_mode : bool
        If True, data was recorded in raw mode.
    header_bytesize : int
        Binary header size in bytes

    Notes
    -----
    If `fh` is an opened file, the file position after parsing is at
    the data block, and Exception('EOF') is raised at end of file.

    """

    def __init__(self, fh, fmt_ver):
        """
        Parameters
        ----------
        fh : bytes object or file / io.BufferedReader
            Object to parse.
        fmt_ver : integer
            Internal format version, determined from Merlin header file.

        """

        self._fh = fh
        self._fmt_ver = fmt_ver

        self._read_header()
        self._parse_header()

    def _read_header(self):
        # Read header
        if type(self._fh) is str:
            # header is provided directly
            self.header = self._fh
        elif type(self._fh) == io.BufferedReader:
            # read correct number of bytes from file
            if self._fmt_ver == 0:
                self.header_bytesize = 256
            elif self._fmt_ver == 1:
                # Length is variable and contained in 3rd csv in binary hdr
                nc = 0  # number of commas read
                ns = 0  # number of bytes read
                sl = []
                while True:
                    c = self._fh.read(1)
                    if c == b"":
                        raise Exception("EOF")  # eof
                    ns += 1
                    if c == b",":
                        nc += 1
                        if nc == 3:
                            break
                    elif nc > 1:
                        sl.append(c)

                self.header_bytesize = int(b"".join(sl))
                self._fh.seek(-ns, 1)  # reset to start of header
            else:
                m = "Format '%s' versions not understood" % (self._fmt_ver)
                _logger.error(m)
                raise Exception(m)

            self.header = self._fh.read(self.header_bytesize)
            if self.header == b"":
                # reached end of file
                raise Exception("EOF")
        else:
            m = "fh must be of type 'str' or 'file', got %s." % (str(type(self._fh)))
            _logger.error(m)
            raise Exception(m)

    def _parse_header(self):
        self.hs = self.header.split(b"\x00", 1)[0]
        hv = self.hs.split(b",")
        self.detY, self.detX = np.array(hv[4:6], int)

        if self._fmt_ver == 0:
            Threshold_keV = np.array(hv[5:7], dtype="f4")
            DAC_val = np.array(hv[7:], dtype="u1")
            Exposure_nS = int(float(hv[4]) * 1e6)

            t = time.mktime(parse(hv[3]).timetuple())
            Unixtime_nS = int(t // 1) * 1000000000 + int(
                hv[3].rsplit(".")[-1].ljust(9, "0")
            )

            bitdepth_data = int(hv[0][:2])
            if bitdepth_data == 12:
                bitdepth_bin = 16
            else:
                bitdepth_bin = 32

        elif self._fmt_ver == 1:
            bitdepth_bin = int(hv[6][1:])
            self.CM = bool(hv[12])
            bitdepth_data = int(hv[-2])  # 53
            Exposure_nS = int(hv[-3].lower().split(b"ns")[0])  # 52
            t = time.mktime(parse(hv[-4]).timetuple())  # 51
            # s = t//1 + float(hv[51].rsplit('.')[-1][:-1])/1e9
            Unixtime_nS = int(t // 1) * 1000000000 + int(
                hv[-4].rsplit(b".")[-1][:-1]
            )  # 51

            if bitdepth_bin == 64:
                self.raw_mode = True
            else:
                self.raw_mode = False

                Threshold_keV = np.array(hv[14:22], dtype="f4")
                DAC_val = np.array(hv[23:50], dtype="u2")

        # these always exist
        self.bitdepth_data = bitdepth_data
        self.bitdepth_bin = bitdepth_bin

        # make dictionary of data to be added to file
        self.params = {
            "exposure": {
                "name": "Exposure",
                "unit": "nS",
                "data": Exposure_nS,
                "isim": True,
            },
            "unixtime": {
                "name": "Unixtime",
                "unit": "nS",
                "data": Unixtime_nS,
                "isim": True,
            },
        }

        # add non-raw params
        if self.raw_mode == False:
            nonraw_params = {
                "threshold": {
                    "name": "Threshold",
                    "unit": "keV",
                    "data": Threshold_keV,
                    "isim": False,
                },
                "DAC": {"name": "DAC", "unit": "val", "data": DAC_val, "isim": False},
            }
            self.params.update(nonraw_params)

    def _print_testing_notes(self):
        m = """
            
            #--------------------------------------------------
            # 0.77.0.16
            'MQ1,000001,00384,01,0256,0256,U16,   1x1,01,2021-10-04 12:57:48.710465,0.001000,0,0,0,5.700000E+1,5.110000E+2,0.000000+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,3RX,114,511,000,000,000,000,000,000,100,255,200,125,100,100,098,100,104,050,128,004,255,084,128,135,125,511,511,MQ1A,2021-10-04T16:57:48.710465438Z,1000000ns,12,'
            
            # 0.75.4.84
            'MQ1,000001,00384,01,0256,0256,U16,   1x1,01,2020-12-18 10:49:31.280266,0.039191,0,0,0,3.200000E+1,5.110000E+2,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,3RX,059,511,000,000,000,000,000,000,100,255,200,125,100,100,098,100,104,050,128,004,255,084,128,135,125,511,511,MQ1A,2020-12-18T15:49:31.280266304Z,39191060ns,12,'
                        
            #--------------------------------------------------
            # VERSION 0.76 (no documentation)
            Format is 256 bytes, zero byte padded.
            20150825 - currently 34 values, eg:
            
            '12B,000001,0,2015-08-11 17:55:11.682,1.000000E+0,2.700000E+1,3.600000E+0,150,020,000,000,000,000,000,000,100,010,125,125,100,100,080,100,120,050,128,004,255,108,120,160,148,401,401'
            
                bit depth                               12B|24B
                frame number (6 wide)                   000001
                counter or colour (in CM)?              0    # TODO check value
                date and time stamp (to ms)             2015-08-11 17:55:11.682
                exposure (ms)                           1.000000E+0
                threshold0                              2.700000E+1
                threshold1                              3.600000E+0
                DAC (27 of)                             150
            
            #--------------------------------------------------
            # VERSION 0.65.8 (differs from documentation)
            Variable length csv, zero byte padded. eg:
            
            'MQ1,000001,00384,01,0128,0128,U16,   2x2,01,2015-08-27 18:10:36.025832,0.010000,0,1,0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,3RX,000,000,000,000,000,000,000,000,175,010,200,125,100,100,093,100,099,030,128,004,255,127,128,177,167,000,000,MQ1A,2015-08-27T17:10:36.025832396Z,10000000nS,12,'
            
                id                                      MQ1
                frame number                            000001
                header length                           00384
                number of chips                         01
                image size x                            0128
                image size y                            0128
                binary no. fmt                          U16
                chip layout                             2x2
                hexadecimal of active chips             01
                date and time stamp (to us)             2015-08-27 18:10:36.025832
                exposure (s w/ us resolution)           0.010000
                counter number (0,1) or colour (0-7)    0
                colour mode bool                        1
                gain mode 0=SLGM, 1=LGM, 2=HGM, 3=SHGM  0
                TH0 (keV)                               0.000000E+0
                TH1 (keV)                               0.000000E+0
                TH2 (keV)                               0.000000E+0
                TH3 (keV)                               0.000000E+0
                TH4 (keV)                               0.000000E+0
                TH5 (keV)                               0.000000E+0
                TH6 (keV)                               0.000000E+0
                TH7 (keV)                               0.000000E+0
                
                REPEATED ONCE PER CHIP
                    chip version                        3RX
                    DAC (27 of)                         000
                
                    UNDOCUMENTED
                        more detailed id?               MQ1A
                        ns timestamp                    2015-08-27T17:10:36.025832396Z
                        exposure (ns)                   10000000nS
                        bit depth?                      12
            
            #--------------------------------------------------
            # VERSION 0.67.0.6 as 0.65.8
            
            Raw mode status is not recorded in hdr file. Comparison of binary headers:
            
            
            head -c 2000 12bit_multi_file/12bit_multi_file1.mib; echo 
            MQ1,000001,00384,01,0256,0256,U16,   1x1,01,2017-11-10 15:35:28.419324,0.005000,0,0,0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,3RX,060,510,000,000,000,000,000,000,100,010,100,125,100,100,064,100,070,030,128,004,255,125,128,179,167,511,511,MQ1A,2017-11-10T15:35:28.419324545Z,5000000ns,12,
            
            
            head -c 2000 1bit_raw.mib; echo 
            MQ1,000001,00384,01,0256,0256,R64,   1x1,01,2017-11-10 15:38:15.030531,0.005000,0,0,0,,MQ1A,2017-11-10T15:38:15.030531785Z,5000000ns,1,

            head -c 2000 6bit_raw.mib; echo 
            MQ1,000001,00384,01,0256,0256,R64,   1x1,01,2017-11-10 15:39:58.616402,0.005000,0,0,0,,MQ1A,2017-11-10T15:39:58.616402345Z,5000000ns,6,

            head -c 2000 12bit_raw.mib; echo 
            MQ1,000001,00384,01,0256,0256,R64,   1x1,01,2017-11-10 15:40:37.414250,0.005000,0,0,0,,MQ1A,2017-11-10T15:40:37.414250055Z,5000000ns,12,

            head -c 2000 24bit_raw.mib; echo 
            MQ1,000001,00384,01,0512,0256,R64,   1x1,01,2017-11-10 15:41:20.880248,0.005000,0,0,0,,MQ1A,2017-11-10T15:41:20.880248165Z,5000000ns,24,
            
            
            Differences:
                raw mode documented by 7th csv as R64
                after counter, color, gain, is empty string.
                missing are thresholds, chip version and dacs
                next values are same last 4 values begining with MQ1A
                last of these seems to indicate the bit depth
            
            Examples of missing:
                0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,3RX,060,510,000,000,000,000,000,000,100,010,100,125,100,100,064,100,070,030,128,004,255,125,128,179,167,511,511,
            
            
            
            ----------------------------------------------------------------------------------------------------
            # V 0.67.0.8 appears the same as 0.67.0.6
            
            with open('test_small_16/test_small_16.mib', 'rb') as f:
                t = fpdf.Merlin_binary_header_parser(f, fmt_ver=1)
            
            t.header
            
            MQ1,000001,00384,01,0256,0256,U32,   1x1,01,2018-03-01 15:10:00.059524,0.002000,0,0,0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,3RX,040,511,000,000,000,000,000,000,175,010,200,125,100,100,096,100,093,030,128,004,255,054,128,104,100,511,511,MQ1A,2018-03-01T15:10:00.059524623Z,2000000ns,24,
            
            """
        print(m)


def _parse_dm_tia_file(fn):
    """
    Parse a DM / TIA file to extract axes properties, tags, image data, and
    binary.

    Parameters
    ----------
    fn : string
        Input file name.

    Returns
    -------
    (pixX, axisX, unitsX) : tuple
        Image x-axis information where:
            pixX : int
                Number of pixels.
            axisX : ndarray
                Pixel axis.
            unitsX : string
                Axis units.
    (pixY, axisY, unitsY) : tuple
        y-axis information, as above.
    tag_dict : dictionary
        Dictionary of image tags.
    im_data : ndarray
        Image data array.
    bin_file : binary object
        Complete binary file.

    Examples
    --------
    xaxis, yaxis, tag_dict, im_data, bin_file = _parse_dm_tia_file('1.dm3')
    pixX, axisX, unitsX = xaxis
    pixY, axisY, unitsY = yaxis

    """
    try:
        from hyperspy.io import load
    except ModuleNotFoundError as e:
        m = "To process Digital Micrograph or TIA files, install HyperSpy. Alternatively, specify the data axes through the 'scanYalu' and 'scanXalu' parameters in 'MerlinBinary'."
        _logger.warn(m)
        raise e

    with open(fn, "rb") as f:
        bin_file = f.read()

    im = load(fn)

    # TIA files can apparently be 3D, with multiple signals in the 1st dimension
    # only store 1st image as EMD, but raise warning if there are more valid images
    # TODO allow 3D data in EMD in future?
    ext = fn.rsplit(".", 1)[-1]
    if "dm" not in ext.lower():
        # we have a TIA file
        valid_n_ims = im.original_metadata.ser_header_parameters.ValidNumberElements
        if valid_n_ims > 1:
            _logger.info(
                "Only the first of %d valid TIA image data in '%s' will be stored as an EMD."
                % (valid_n_ims, fn)
            )
            _logger.info("However, the entire TIA file is stored as a binary blob.")
    if im.data.ndim > 2:
        im_data = im.inav[0].data
    else:
        im_data = im.data

    tagd = im.original_metadata.as_dictionary()
    unitsX = im.axes_manager.signal_axes[0].units
    axisX = im.axes_manager.signal_axes[0].axis
    pixX = im.axes_manager.signal_axes[0].size
    if len(im.axes_manager.signal_axes) == 1:
        pixY = 1
        unitsY = im.axes_manager.signal_axes[0].units
        axisY = axisX[:1]
    else:
        pixY = im.axes_manager.signal_axes[1].size
        unitsY = im.axes_manager.signal_axes[1].units
        axisY = im.axes_manager.signal_axes[1].axis

    return (pixX, axisX, unitsX), (pixY, axisY, unitsY), tagd, im_data, bin_file


def _create_emd(
    fgp,
    gname,
    im_class,
    dim_names,
    dim_axes,
    dim_units,
    data_name="",
    data_units="",
    *args,
    **kwargs,
):
    """
    Create an EMD compatible dataset, with additional attributes.

    Parameters
    ----------
    fgp : one of [filename, hdf5 group, h5 file]
        Object in which EMD group will be made.
    gname : string
        EMD group name. If empty (''), EMD dataset will be in `fgp`.
    im_class : bool
        booling controlling if data is an hdf5 image class.
    dim_names : list of strings
        Dimension names.
    dim_axes : list of ndarrays
        Dimension axes.
    dim_units : list of strings
        Dimension units.
    data_name : string
        Attribute for data group.
    data_units : string
        Attribute for data group.

    Additional parameters are passed to `create_dataset`. See h5py for
    details.

    Returns
    -------
    ds : hd5f dataset
        The created dataset. This will be in a closed file if `fgp` is
        a filename.

    """

    for x in [dim_names, dim_axes, dim_units]:
        assert type(x) == list

    # should file be closed at end?
    closef = False
    fpgn_type = type(fgp)

    if fpgn_type is str:
        # fgp is filename
        h5f = h5py.File(fgp, "w")
        closef = True
    elif fpgn_type == h5py._hl.files.File:
        h5f = fpgn
    elif fpgn_type == h5py._hl.group.Group:
        h5f = fgp

    if gname:
        grp = h5f.create_group(name=gname)
    else:
        grp = h5f

    # Group attributes
    grp.attrs["emd_group_type"] = 1

    # This makes the HyperSpy file loader identify the N-D datasets
    # as Signal2D, making it easier to navigate and use
    if len(dim_names) >= 3:
        grp.attrs["signal_type"] = "Signal2D"

    ds = grp.create_dataset(name="data", *args, **kwargs)

    if isinstance(data_name, bytes):
        data_name = data_name.decode("utf-8")
    if isinstance(data_units, bytes):
        data_units = data_units.decode("utf-8")

    ds.attrs["name"] = data_name
    ds.attrs["units"] = data_units
    if im_class == True:
        ds.attrs["CLASS"] = np.bytes_("IMAGE")
        ds.attrs["IMAGE_VERSION"] = np.bytes_("1.2")

    # add axes
    dim_dataset_names = ["dim" + str(x + 1) for x in range(len(dim_names))]
    for i, dim_dataset_name in enumerate(dim_dataset_names):
        try:
            dim_ds = grp.create_dataset(
                name=dim_dataset_name,
                data=dim_axes[i],
                compression="gzip",
                compression_opts=4,
            )
        except Exception as e:
            # print(e)
            # Scalar datasets don't support chunk/filter options
            dim_ds = grp.create_dataset(name=dim_dataset_name, data=dim_axes[i])

        dim_name = dim_names[i]
        dim_unit = dim_units[i]
        if isinstance(dim_name, bytes):
            dim_name = dim_name.decode("utf-8")
        if isinstance(dim_unit, bytes):
            dim_unit = dim_unit.decode("utf-8")

        dim_ds.attrs["name"] = dim_name
        dim_ds.attrs["units"] = dim_unit

    if closef:
        h5f.close()
    return ds


def _create_dm_tia_grp(
    fgp,
    gname,
    im_class,
    dim_names,
    dim_axes,
    dim_units,
    dmtia_fn,
    dmtia_tagd,
    dmtia_bin,
    *args,
    **kwargs,
):
    """
    Create a DM or TIA group containing an EMD compatible dataset, a
    dictionary of tags, and the binary file.

    Parameters
    ----------
    fgp : one of [filename, hdf5 group, h5 file]
        Object in which EMD group will be made.
    gname : string
        Group name. If empty (''), DM or TIA dataset will be in `fgp`.
    im_class : bool
        boolean controlling if data is an hdf5 image class.
    dim_names : list of strings
        Dimension names.
    dim_axes : list of ndarrays
        Dimension axes.
    dim_units : list of strings
        Dimension units.

    dmtia_fn : string
        DM / TIA filename.
    dmtia_tagd : dictionary
        DM / TIA tag dictionary.
    dmtia_bin : binary
        DM / TIA file binary.

    Additional parameters are passed to `create_dataset` for EMD. See
    h5py for details.

    Returns
    -------
    grp : hd5f group
        The created DM or TIA group. This will be in a closed file if `fgp` is
        a filename.

    """

    for x in [dim_names, dim_axes, dim_units]:
        assert type(x) == list

    closef = False
    fpgn_type = type(fgp)

    if fpgn_type is str:
        h5f = h5py.File(fgp, "w")
        closef = True
    elif fpgn_type == h5py._hl.files.File:
        h5f = fpgn
    elif fpgn_type == h5py._hl.group.Group:
        h5f = fgp

    if gname:
        grp = h5f.create_group(name=gname)
    else:
        grp = h5f

    # get original filename
    dmtia_fn_dir, dmtia_fn_fn = os.path.split(dmtia_fn)

    # simple decision of group type from file extension
    ext = dmtia_fn_fn.rsplit(".", 1)[-1]
    if "dm" in ext.lower():
        grp_attrs_type = "dm_group_type"
        grp_attrs_fn = "dm_filename"
    else:
        grp_attrs_type = "tia_group_type"
        grp_attrs_fn = "tia_filename"

    # internal group 'version'
    grp.attrs[grp_attrs_type] = 1
    # add DM / TIA filename with extension as attribute
    grp.attrs[grp_attrs_fn] = dmtia_fn_fn

    # add EMD as subgroup
    _create_emd(
        grp,
        "",
        im_class,
        dim_names,
        dim_axes,
        dim_units,
        data_name="Intensity",
        data_units="Counts",
        *args,
        **kwargs,
    )

    # add binary file
    DMTIA_bin = grp.create_dataset(name="bin", data=np.void(dmtia_bin))

    # DM / TIA tags
    DMTIA_tags_grp = grp.create_group(name="tags")
    # exclude empty dicts which raise error when writing to hdf5
    tagdf = _flatten(dmtia_tagd, sep="/", include_empty=False)
    for k, v in tagdf.items():
        # Some TIA tags return a None value and cause h5py to return an error.
        # In this case, set the values to 'None' to prevent this.
        if v is None:
            v = "None"
        try:
            DMTIA_tags_grp.create_dataset(
                name=k, data=v, compression="gzip", compression_opts=4
            )
        except Exception as e:
            # print(e)
            # Scalar datasets don't support chunk/filter options
            DMTIA_tags_grp.create_dataset(name=k, data=v)

    if closef:
        h5f.close()
    return grp


def _sort_multiple_binary_file_input(filename_list):
    """
    Function for sorting multiple binary files written by the Merlin
    Medipix readout system in the acquisition order.

    Parameters
    ----------
    filename_list : list of strings

    Notes
    -----
    One of the settings on the Merlin readout system saves each frame
    as a separate file. This function will work if there are other
    numbers in the filename, but there must be a non-numeric character
    between the file counter and any other numbers.

    """

    index_list = []
    # Extract index numbers from the filenames.
    # Makes sure the last number in the filename is chosen, in case there are several
    # numbers in the filename
    for filename in filename_list:
        temp_filename = filename.split(".")[-2]
        index_list.append(int(re.search("\d+", temp_filename[::-1]).group()[::-1]))

    # Sort the filename_list using the index_list
    new_filename_list = [
        filename_list
        for (index_list, filename_list) in sorted(zip(index_list, filename_list))
    ]
    return new_filename_list


def _embed_source_in_hdf5_group(fpd_expt_grp):
    # embed source
    src_path = inspect.stack()[0][1]
    with open(src_path, "r") as f:
        src_str = f.read()

    src_ds = fpd_expt_grp.create_dataset(name="src", data=src_str)
    src_ds.attrs["src_path"] = src_path


def _embed_header_in_hdf5_group(fpd_expt_grp, header):
    fpd_hdr_grp = fpd_expt_grp.create_group(name="merlin_hdr")
    fpd_hdr_grp.attrs["string"] = header.hs

    fhdf = _flatten(header.hd, sep="/")
    for k, v in fhdf.items():
        try:
            fpd_hdr_grp.create_dataset(
                name=k, data=v, compression="gzip", compression_opts=4
            )
        except:
            fpd_hdr_grp.create_dataset(name=k, data=v)


def _create_sum_im_hdf5_dataset(
    group, base_kwd, det_pix_max_val, dim_lens, dim_names, dim_axes, dim_units
):
    """
    Create sum_im dataset, returning dataset object.
    Sum of all detector signals at each scan pixel.
    """

    sum_im_dtype = "u8"
    assert det_pix_max_val * np.prod(dim_lens[-2:]) < np.iinfo(sum_im_dtype).max

    kwd = base_kwd.copy()
    shp = tuple(dim_lens[:-2])
    kwd.update({"shape": shp, "dtype": sum_im_dtype, "maxshape": shp})
    fpd_sum_im_dataset = _create_emd(
        group,
        "fpd_sum_im",
        True,
        dim_names[:-2],
        dim_axes[:-2],
        dim_units[:-2],
        data_name="Intensity",
        data_units="Counts",
        **kwd,
    )
    return fpd_sum_im_dataset


def _create_sum_dif_hdf5_dataset(
    group, base_kwd, det_pix_max_val, dim_lens, dim_names, dim_axes, dim_units
):
    """
    Create sum_dif dataset, returning dataset object.
    Sum of each detector pixel across all scan pixels.
    """

    sum_dif_dtype = "u8"
    assert det_pix_max_val * np.prod(dim_lens[:2]) < np.iinfo(sum_dif_dtype).max

    kwd = base_kwd.copy()
    shp = tuple(dim_lens[2:])
    kwd.update({"shape": shp, "dtype": sum_dif_dtype, "maxshape": shp})
    fpd_sum_dif_dataset = _create_emd(
        group,
        "fpd_sum_dif",
        True,
        dim_names[2:],
        dim_axes[2:],
        dim_units[2:],
        data_name="Intensity",
        data_units="Counts",
        **kwd,
    )

    fpd_sum_dif_dataset[...] = np.zeros_like(
        fpd_sum_dif_dataset, dtype=fpd_sum_dif_dataset.dtype
    )
    return fpd_sum_dif_dataset


def _create_mask_hdf5_dataset(
    group, base_kwd, dim_lens, dim_names, dim_axes, dim_units
):
    """
    Create mask dataset, returning dataset object.
    """

    dtype = bool

    kwd = base_kwd.copy()
    shp = tuple(dim_lens[-2:])
    kwd.update({"shape": shp, "dtype": dtype, "maxshape": shp})
    kwd.pop("fillvalue")

    fpd_mask_dataset = _create_emd(
        group,
        "fpd_mask",
        True,
        dim_names[-2:],
        dim_axes[-2:],
        dim_units[-2:],
        data_name="Mask",
        data_units="Boolean",
        **kwd,
    )
    return fpd_mask_dataset


def _write_dm_files_to_hdf5(
    group, base_kwd, dmfns, dim_lens, dim_names, dim_axes, dim_units
):
    """
    Write DM file(s)
    """

    for i, dmfn in enumerate(dmfns):
        kwd = base_kwd.copy()

        # TODO move to write dm and have _create_dm_tia_grp handle all processing?
        # This could allow it to handle more filetypes.
        _, _, tagd, dm_im, dm_bin = _parse_dm_tia_file(dmfns[i])
        DM_dtype = "u%d" % (tagd["ImageList"]["TagGroup0"]["ImageData"]["PixelDepth"])
        kwd.update({"data": dm_im, "dtype": DM_dtype})

        _create_dm_tia_grp(
            group,
            "DM%d" % (i),
            True,
            dim_names[:2],
            dim_axes[:2],
            dim_units[:2],
            dmfn,
            tagd,
            dm_bin,
            **kwd,
        )


def _write_tia_files_to_hdf5(
    group, base_kwd, tiafns, dim_lens, dim_names, dim_axes, dim_units
):
    """
    Write TIA file(s)
    """

    for i, tiafn in enumerate(tiafns):
        kwd = base_kwd.copy()

        # TODO move to write tia and have _create_dm_tia_grp handle all processing?
        # This could allow it to handle more filetypes.
        _, _, tagd, tia_im, tia_bin = _parse_dm_tia_file(tiafns[i])
        TIA_dtype = "u%d" % (tagd["ser_header_parameters"]["DataType"])
        kwd.update({"data": tia_im, "dtype": TIA_dtype})

        _create_dm_tia_grp(
            group,
            "TIA%d" % (i),
            True,
            dim_names[:2],
            dim_axes[:2],
            dim_units[:2],
            tiafn,
            tagd,
            tia_bin,
            **kwd,
        )


def _create_bin_header_str_dataset(
    group, base_kwd, dim_lens, dim_names, dim_axes, dim_units
):
    """
    Create FPD binary header string dataset, returning dataset object.
    """

    kwd = base_kwd.copy()
    shp = tuple(dim_lens[:-2])
    kwd.update({"shape": shp, "dtype": h5py.special_dtype(vlen=bytes), "maxshape": shp})
    kwd.pop("fillvalue")

    filter_keys = ["compression", "compression_opts", "chunks", "shuffle", "fletcher32"]
    for key in filter_keys:
        kwd.pop(key, None)

    fpd_binary_hdr_dataset = _create_emd(
        group,
        "binary_hdr",
        False,
        dim_names[:-2],
        dim_axes[:-2],
        dim_units[:-2],
        data_name="binary_hdr",
        data_units="",
        **kwd,
    )
    return fpd_binary_hdr_dataset


def _create_fpd_data_hdf5_dataset(
    group, base_kwd, det_dtype, dim_lens, dim_names, dim_axes, dim_units, chunks
):
    """
    Create FPD data dataset, returning dataset object.
    """

    kwd = base_kwd.copy()
    shp = tuple(dim_lens)
    kwd.update({"shape": shp, "dtype": det_dtype, "chunks": chunks, "maxshape": shp})

    fpd_dataset = _create_emd(
        group,
        "fpd_data",
        True,
        dim_names,
        dim_axes,
        dim_units,
        data_name="Intensity",
        data_units="Counts",
        **kwd,
    )
    return fpd_dataset


def _create_bin_header_param_hdf5_dataset(
    group, base_kwd, det_dtype, dim_lens, dim_names, dim_axes, dim_units, params
):
    """
    Create FPD binary header datasets, returning list of datasets.
    """

    bhd_dss = []
    for pk in params.keys():
        p = params[pk]
        data = p["data"]
        data_dim_name_s = p["name"]
        data_unit_s = p["unit"]
        isim = p["isim"]

        kwd = base_kwd.copy()
        try:
            # arrays
            data_len = len(data)
            data_dtype = data.dtype
            shp = tuple(dim_lens[:-2])
            kwd.update(
                {
                    "shape": shp + (data_len,),
                    "dtype": data_dtype,
                    "maxshape": shp + (data_len,),
                }
            )
            bhd_ds = _create_emd(
                group,
                data_dim_name_s,
                isim,
                dim_names[:-2] + [data_dim_name_s],
                dim_axes[:-2] + [np.arange(data_len)],
                dim_units[:-2] + [data_unit_s],
                data_name=data_dim_name_s,
                data_units=data_unit_s,
                **kwd,
            )
        except TypeError:
            # scalar
            data_len = 0
            data_dtype = type(data)
            shp = tuple(dim_lens[:-2])
            kwd.update({"shape": shp, "dtype": data_dtype, "maxshape": shp})
            bhd_ds = _create_emd(
                group,
                data_dim_name_s,
                isim,
                dim_names[:-2],
                dim_axes[:-2],
                dim_units[:-2],
                data_name=data_dim_name_s,
                data_units=data_unit_s,
                **kwd,
            )
        bhd_dss.append(bhd_ds)
    return bhd_dss


def _read_header_and_data(fp, fmt_ver, im_bytesize, return_bin_data=True):
    """
    Helper function to read binary header and data.
    """

    try:
        Mbhp = Merlin_binary_header_parser(fp, fmt_ver)
        if return_bin_data:
            bin_data = fp.read(im_bytesize)
        else:
            bin_data = None
            fp.seek(im_bytesize, 1)
    except Exception as e:
        if str(e) == "EOF":
            return ""
        else:
            raise
    return (Mbhp.hs, Mbhp.params, bin_data)


def _raw_merlin_to_array(
    data_buffer, det_dtype, bitdepth_data, bitdepth_bin, raw_mode, detY=256, detX=256
):
    """
    Convert Merlin data from string to array.

    Parameters
    ----------
    data_buffer : binary string to convert
        Binary string to convert.
    det_dtype : numpy dtype
        numpy dtype in which data is stored in string
    bitdepth_data : integer
        Data bitdepth: 1, 6, 12, 24.
    bitdepth_bin : integer
        Binary bitdepth (non-raw: 8, 16 or 32, raw: 64).
    raw_mode : bool
        If True, data_buffer is in raw mode form.
    detY : int
        Number of detector y pixels. See notes.
    detX : int
        Number of detector x pixels. See notes.

    Returns
    -------
    im_array : ndarray
        2-D array of detector data of appropriate type.

    Notes
    -----
    `detY` and `detX` are the modified from the real pixel count by running
    in colour mode.

    """

    raw_1bit = raw_mode and (bitdepth_data == 1)

    # process bin_data
    if raw_1bit:
        # raw 1bit is really 1 bit
        # this is all really slow!
        im_array = np.frombuffer(data_buffer, dtype=">u1")

        im_array = np.array([np.unpackbits(byte) for byte in im_array], dtype=det_dtype)
        im_array = im_array.flatten().reshape(detY, detX)
    else:
        im_array = np.frombuffer(data_buffer, dtype=det_dtype).reshape(detY, detX)

    if raw_mode:
        # raw mode needs data flipped in column sections.
        # n = 64  # 1 bit
        # n = 8   # 6 bit
        # n = 4   # 12 bit
        # n = 2   # 24 bit (untested)

        # overwrite bitdepth_bin which is always 64 with real bitdepth
        bitdepth_bin = int(max(1, 2 ** np.ceil(np.log2(bitdepth_data)) // 8)) * 8
        n = 64 // bitdepth_bin
        if raw_1bit:
            n = 64

        indices = np.arange(detX / n + 1, dtype=int) * n
        im_array_temp = np.empty_like(im_array)
        for start, stop in zip(indices[:-1], indices[1:]):
            im_array_temp[:, start:stop] = im_array[:, start:stop][:, ::-1]
        im_array = im_array_temp
    return np.ascontiguousarray(im_array[::-1])


def _image_array_properties(detY, detX, Mbhp):
    """
    Determine image array properties
    """
    # detector data storage bitdepth
    # min x in 2**x bytes. The max makes 1bit -> 8bit
    nbytes = int(max(1, 2 ** np.ceil(np.log2(Mbhp.bitdepth_data)) // 8))
    det_dtype = np.dtype(">u%d" % (nbytes))

    im_bytesize = detX * detY * nbytes
    # raw 1bit is really 1 bit
    raw_1bit = Mbhp.raw_mode and (Mbhp.bitdepth_data == 1)
    if raw_1bit:
        im_bytesize = int(detX * detY / 8)
    return nbytes, det_dtype, im_bytesize


class CubicImageInterpolator:
    def __init__(self, mask):
        """
        CloughTocher2DInterpolator for 2-D arrays at locations where mask is True.

        Parameters
        ----------
        mask : 2-D bool array
            Mask indicating which values are to be interpolated.
        """

        self.mask = np.asarray(mask, dtype=bool)
        self.keep_inds = np.where(self.mask == False)
        self.interp_inds = np.where(self.mask == True)
        self.points = np.asarray(self.keep_inds).T
        self.xi = np.asarray(self.interp_inds).T

        # dummy values for initialisation
        values = np.ones(self.points.shape[0], dtype=float)
        self.ip = CloughTocher2DInterpolator(
            self.points, values, fill_value=np.nan, rescale=False
        )

    def interpolate_image(self, image, clip_min=None, clip_max=None, in_place=True):
        """
        Interpolate image

        Parameters
        ----------
        image : 2-D array
            Image to be processed
        clip_min : None or scalar
            Minimum value to clip interpolated values to.
        clip_max : None or scalar
            Maximum value to clip interpolated values to.
        in_place : bool
            If True, the image is processed in place. Otherwise, a new image is returned.

        Returns
        -------
        image : 2-D array or None
            Interpolated input array or None. See `in_place` for details.
        """

        self.ip.values[:] = image[self.keep_inds].astype(
            self.ip.values.dtype, copy=False
        )[:, None]
        interp_vals = self.ip(self.xi)
        if clip_min is not None or clip_max is not None:
            interp_vals = interp_vals.clip(clip_min, clip_max)
        if not in_place:
            image = image.copy()
            image[self.interp_inds] = interp_vals.astype(image.dtype, copy=False)
            return image
        else:
            image.setflags(write=1)  # make sure image is writable
            image[self.interp_inds] = interp_vals.astype(image.dtype, copy=False)


def determine_read_chunks(fpd_dataset, MiB_max, det_im_processing=True):
    """
    Determine chunk size for reading a dataset.

    Parameters
    ----------
    fpd_dataset : hdf5 dataset
        The dataset to assess the chunk size.
    MiB_max : scalar
        Maximum number of mebibytes (1024**2 bytes) allowed for file access. This is
        a soft limit and controls how many chunks are read for calculating image sums.
        The minimum number of chunks is currently 1, so this limit might be breached
        for very large chunk sizes.
    det_im_processing : bool
        If True, the chunks are for the scan axes.
        If False, they are for the detector axes.

    Returns
    -------
    read_chunk_y, read_chunk_x : int
        The size of chunks to read.

    """

    byte_limit = MiB_max * 1024**2

    # chunk logic
    scan_chunks = fpd_dataset.chunks[:2]
    non_scan_chunks = fpd_dataset.chunks[-2:]

    scan_chunk_npix = np.prod(scan_chunks)
    non_scan_chunk_npix = np.prod(non_scan_chunks)

    scan_area = np.prod(fpd_dataset.shape[:2])
    non_scan_area = np.prod(fpd_dataset.shape[-2:])

    ds_nbytes = fpd_dataset.dtype.itemsize

    scan_chunk_bytes = scan_chunk_npix * non_scan_area * ds_nbytes
    non_scan_chunk_bytes = non_scan_chunk_npix * scan_area * ds_nbytes

    if det_im_processing:
        nb = non_scan_chunk_bytes
        c = scan_chunks
    else:
        nb = scan_chunk_bytes
        c = non_scan_chunks

    n_chunks = np.floor(byte_limit / scan_chunk_bytes)
    n_chunks = max(1, n_chunks)
    # could regard as hard limit and set for reading integer divisions of chunks
    # to avoid exceeding limit for large chunk sizes

    cax = np.arange(n_chunks) + 1
    cm = cax[None] * cax[:, None]
    cm[cm > n_chunks] = 0

    # weight chunk magnitude array to prefer squarish chunks
    r = (np.indices(cm.shape) ** 2).sum(0) ** 0.5  #
    r[0, 0] = 1e6
    t = cm / r**0.5
    read_chunk_y, read_chunk_x = np.array(np.unravel_index(t.argmax(), cm.shape)) + 1
    read_chunk_y *= c[0]
    read_chunk_x *= c[1]

    return read_chunk_y, read_chunk_x


class MerlinBinary:
    def __init__(
        self,
        binfns,
        hdrfn,
        dmfns=[],
        tiafns=[],
        ds_start_skip=0,
        row_end_skip=1,
        scanYalu=None,
        scanXalu=None,
        detYalu=None,
        detXalu=None,
        detY=256,
        detX=256,
        sort_binary_file_list=True,
        strict=True,
        array_interface_progress_bar=False,
        *args,
        **kwargs,
    ):
        """

        Class for reading data produced by the Merlin Medipix 3 readout system.
        The class exposes an array-like interface to allow reading directly
        from the file to memory, and can convert the detector data and metadata
        into an hdf5 file.

        The data order c-order: scanY, scanX, [colour,] detY, detX
        The colour axis is omitted if singular.

        Parameters
        ----------
        binfns : str, or list of str
            Binary FPD filenames to process.
        hdrfn : str
            FPD text header filename.
        dmfns : str, or list of str
            DM filenames to process. Use [] for no files. The scan
            dimensions are parsed from the first file. If no files are
            supplied, scanXpau and scanYalu control scan size. See Notes.
        tiafns : str, or list of str
            TIA filenames to process. Use [] for no files. The scan
            dimensions are parsed from the first file. If no files are
            supplied, scanXpau and scanYalu control scan size. See Notes.
        ds_start_skip : int
            Number of scan pixels skipped at start of input file.
        row_end_skip : int
            Number of pixels to skip after end of each row.
            This can be used to omit false triggers during beam flyback.
        scanXalu : tuple or None
            Determines scan size if no `dmfns` or `tiafns` are specified. The
            tuple must be in the form (axis, label, units). See Notes.
            scanXaxis : int or iterable
                If an integer, this is taken as the axis length and an axis is
                generated. Otherwise, scanX is converted to an axis directly.
            scanXlabel : str
                Name of x-axis.
            scanXunits : str
                Units of the x-axis.
        scanYalu : tuple or None
            As `scanXalu`, but for y-axis.
        detYalu : tuple or None
            As `scanXalu`, but for detector y-axis. The axis size should be
            the size of the data axis. See also `detY`.
        detXalu : tuple or None
            As `scanXalu`, but for detector x-axis.
        detY : int
            The number of hardware pixels is SPM mode.
            This value will be automatically modified in colour mode.
        detX : int
            As `detY`, but for detector x-axis.
        sort_binary_file_list : bool
            If True, and if binfns is a list, the filenames will be sorted
            by frame number.
        strict : bool
            If True, file format matching is strict and only known versions
            will be processed. If False, the most recent known version will
            be tried.
        array_interface_progress_bar : bool
            If True, a progress bar is displayed.

        Notes
        -----
        If 'scanXalu' or 'scanYalu' not None, these take precedence over
        scan parameters from DM or TIA files. If None, the scan parameters
        are parsed from the first DM or TIA file, with the DM parameters
        taking priority.

        TIA files are read using HyperSpy, which can apparently read .emi
        and .ser files. However, versions upto 1.7.5 used on test data only
        return data for .ser files.

        For any other files, the axes information can be parsed with:

        >>> from hyperspy.io import load
        >>> im = load('image_filename')
        >>> scanXalu, scanYalu = [(m.axis, m.name, m.units) for m in im.axes_manager.signal_axes]

        Non-scan data can be processed by setting the y-axis size as 1, and
        the x-axis size the same as the number of images. In this case,
        `ds_start_skip` and `row_end_skip` should be set to 0.

        If the scanXalu axis is None, it is automatically generated from the header
        file with axis being the index. Note that the header file contains the
        maximum number of frames and the real number of frames could be smaller by
        design or error.

        If no scan information is passed, and a single binary file is provided, all
        images in the file are processed as though the shape was [nimages, 1, detY, detX].
        In this case, `row_end_skip` should be set to 0 unless there are a known
        number of unwanted images at the end of the dataset.

        The class exposes an array-like interface that may be sliced and indexed,
        with ndims, size, nbytes, shape, dtype etc attributes. Since the data are
        read from disc, these should not be changed after the class is initialised.

        When the class is indexed, the file is parsed from the beginning, since
        the file format allows each header length to be of a different size, and
        so multiple single files may be converted.

        For detY and detX the number of pixels registered by the Merlin software, which
        depends on operational mode rather than the physical number of pixels,
        should be used.
        """

        self._binfns = binfns
        self._hdrfn = hdrfn
        self._dmfns = dmfns
        self._tiafns = tiafns
        self._ds_start_skip = ds_start_skip
        self._row_end_skip = row_end_skip
        self._scanYalu = scanYalu
        self._scanXalu = scanXalu
        self._detYalu = detYalu
        self._detXalu = detXalu
        self._detY = detY
        self._detX = detX
        self._sort_binary_file_list = sort_binary_file_list
        self._strict = strict
        # self._memmap = memmap

        self._condition_params()
        self._parse_headers()
        self._determine_axes()

        # data and image dimensions / size
        self._ims_shape = self._dim_lens[:-2]
        self._total_no_ims = np.prod(self._ims_shape)

        # set getitem attributes
        self.shape = tuple(self._dim_lens)
        self.size = np.prod(self._dim_lens)
        self.ndim = len(self._dim_lens)
        self.dtype = self._det_dtype
        self.nbytes = (self.dtype.itemsize * 8) * self.size

        self.array_interface_progress_bar = array_interface_progress_bar

    def _condition_params(self):
        # if binfns is string, convert to list for looping
        if type(self._binfns) is str:
            self._binfns = [self._binfns]
        else:
            if self._sort_binary_file_list:
                m = "Sorting binfns by frame number, deactivite sorting by 'setting sort_binary_file_list=False'."
                _logger.info(m)
                self._binfns = _sort_multiple_binary_file_input(self._binfns)
        # if dmfns is string, convert to list for looping over 1 or more DM files.
        if type(self._dmfns) is str:
            self._dmfns = [self._dmfns]
        # if tiafns is string, convert to list for looping over 1 or more TIA files.
        if type(self._tiafns) is str:
            self._tiafns = [self._tiafns]

    def _determine_colourN(self):
        # determine the number of colours in the image.
        if self._hdrfn:
            if self._Mhfp.mode == "SPM":
                colourN = self._Mhfp.AC
            elif self._Mhfp.mode == "CSM":
                colourN = self._Mhfp.AC
            elif self._Mhfp.mode == "CM":
                colourN = self._Mhfp.AC * 4
            elif self._Mhfp.mode == "CSCM":
                colourN = self._Mhfp.AC * 4
        else:
            # Check maximum number of colours used in binary header
            with open(self._binfns[0], "rb") as f:
                hdr = f.read(self._Mbhp.header_bytesize)
                hdr = hdr.split(b",")
                colour0 = int(hdr[11])

                frame_size = int(
                    self._Mbhp.header_bytesize
                    + (self._Mbhp.bitdepth_bin * self._detX * self._detY) / 8
                )
                f.seek(-frame_size, 2)
                hdr = f.read(self._Mbhp.header_bytesize)
                hdr = hdr.split(b",")
                colour1 = int(hdr[11])

                if colour0 == colour1:
                    colourN = 1
                else:
                    colourN = np.max(np.array([colour0, colour1])) + 1
        return colourN

    def _parse_headers(self):
        # parse text header file and first binary header if text header file exists
        if self._hdrfn:
            self._Mhfp = Merlin_hdr_file_parser(self._hdrfn, strict=self._strict)
            with open(self._binfns[0], "rb") as f:
                self._Mbhp = Merlin_binary_header_parser(f, self._Mhfp.fmt_ver)
        # parse first binary header if no text header exists
        # assume most recent binary header file format
        else:
            with open(self._binfns[0], "rb") as f:
                self._Mbhp = Merlin_binary_header_parser(f, 1)
        """
        if self._hdrfn:
            if  self._Mhfp.CM and self._detP == 55:
                self._detX //= 2
                self._detY //= 2
            elif not self._Mhfp.CM and self._detP == 110:
                self._detX *= 2
                self._detY *= 2
        else:
            if  self._Mbhp.CM and self._detP == 55:
                self._detX //= 2
                self._detY //= 2
            elif not self._Mbhp.CM and self._detP == 110:
                self._detX *= 2
                self._detY *= 2
        """
        # set detector size and colour depth from hdr file
        self._colourN = self._determine_colourN()

        rtn = _image_array_properties(self._detY, self._detX, self._Mbhp)
        self._nbytes, self._det_dtype, self._im_bytesize = rtn

        # check detector size (set as parameter, but it is in newer bin header files)
        if self._Mbhp.detY != self._detY:
            m = "Binary header detY (%d) != detY supplied (%d)." % (
                self._Mbhp.detY,
                self._detY,
            )
            _logger.info(m)
        if self._Mbhp.detX != self._detX:
            m = "Binary header detX (%d) != detX supplied (%d)." % (
                self._Mbhp.detX,
                self._detX,
            )
            _logger.info(m)

        rtn = _image_array_properties(self._detY, self._detX, self._Mbhp)
        self._nbytes, self._det_dtype, self._im_bytesize = rtn

    def _determine_axes(self):
        # Determine scanX/Y and their axes and units.
        # If non None, use these over those in DM or TIA file
        if (
            len(self._binfns) == 1
            and self._scanXalu is None
            and self._scanYalu is None
            and len(self._dmfns) == 0
            and len(self._tiafns) == 0
        ):
            # If a single binary file is provided with no dimensional info and no DM/TIA files, determine number of frames based
            # on the bit depth and the file size. Resulting data structure will be: [nframes , 1 , detY , detX]
            if self._row_end_skip != 0:
                _logger.info(
                    "Do you really want to set 'row_end_skip' to %d?"
                    % (self._row_end_skip)
                )
            if self._ds_start_skip != 0:
                _logger.info(
                    "Do you really want to set 'ds_start_skip' to %d?"
                    % (self._ds_start_skip)
                )
            # This will fail in the unlikely case of the the header size
            # changing mid-file, but should work in most cases.

            nframes = int(
                (
                    os.path.getsize(self._binfns[0])
                    / (self._Mbhp.header_bytesize + self._im_bytesize)
                )
                / self._colourN
            )
            # subtract _ds_start_skip / offset here, since nframes is total number of images in the file
            nframes = nframes - self._ds_start_skip

            self._scanXalu = (nframes, "xaxis", "pixels")
            self._scanYalu = (1, "yaxis", "pixels")
        elif (
            self._scanYalu is None
            or self._scanXalu is None
            or (
                self._scanXalu is not None
                and self._scanXalu[0] is None
                and self._scanXalu[1] is None
            )
        ):
            # must use DM or TIA file
            if len(self._dmfns) > 0:
                pauX, pauY, tagd, dm_im, dm_bin = _parse_dm_tia_file(self._dmfns[0])
            elif len(self._tiafns) > 0:
                pauX, pauY, tagd, tia_im, tia_bin = _parse_dm_tia_file(self._tiafns[0])
            else:
                _logger.exception(
                    "Either provide DM or TIA files or specify the scan parameters through `scanYalu` and `scanXalu`."
                )

        if self._scanYalu is not None:
            scanYaxis, scanYlabel, scanYunits = self._scanYalu
            if type(scanYaxis) is int:
                scanYaxis = list(range(scanYaxis))
            else:
                scanYaxis = np.array(scanYaxis)
            scanY = len(scanYaxis)
        else:
            scanY, scanYaxis, scanYunits = pauY
            scanYlabel = "scanY"

        if self._scanXalu is not None:
            scanXaxis, scanXlabel, scanXunits = self._scanXalu
            if scanXaxis is None:
                # generate 'scan' size and an index axis from header
                scanXaxis = int(self._Mhfp.hd["frames_in_acquisition"])
            if type(scanXaxis) is int:
                scanXaxis = list(range(scanXaxis))
            else:
                scanXaxis = np.array(scanXaxis)
            if scanXlabel is None:
                scanXlabel = "scanX"
            scanX = len(scanXaxis)
        else:
            scanX, scanXaxis, scanXunits = pauX
            scanXlabel = "scanX"

        self._scanY = scanY
        self._scanX = scanX

        # detector axes
        if self._detYalu is None:
            self._detYalu = (self._detY, "detY", "pixels")
        detYa, detYl, detYu = self._detYalu
        if type(detYa) is int:  # use as length
            detYa = list(range(detYa))
        else:  # use directly
            detYa = np.array(detYa)
        if len(detYa) != self._detY:
            _logger.exception(
                "y-axis detector (%d) and data (%d) pixels don't match"
                % (len(detYa), self._detY)
            )

        if self._detXalu is None:
            self._detXalu = (self._detX, "detX", "pixels")
        detXa, detXl, detXu = self._detXalu
        if type(detXa) is int:  # use as length
            detXa = list(range(detXa))
        else:  # use directly
            detXa = np.array(detXa)
        if len(detXa) != self._detX:
            _logger.exception(
                "x-axis detector (%d) and data (%d) pixels don't match"
                % (len(detXa), self._detX)
            )

        # FPD dataset axes details
        # TODO specify colour and detector axes and units
        # could be custom image tag in DM.
        caxis = [x + 1 for x in range(self._colourN)]
        self._dim_axes = [scanYaxis, scanXaxis, caxis, detYa, detXa]
        self._dim_names = [scanYlabel, scanXlabel, "colour", detYl, detXl]
        self._dim_units = [scanYunits, scanXunits, "colour_index", detYu, detXu]

        # make sure axes are floats so they may be updated at a later point
        self._dim_axes = [np.array(ax, dtype=float) for ax in self._dim_axes]

        if self._colourN == 1:
            # rm colour axis if singular
            _logger.info("Removing singular colour axis.")
            del self._dim_axes[2]
            del self._dim_names[2]
            del self._dim_units[2]

        self._dim_lens = [len(t) for t in self._dim_axes]
        _logger.info("FPD data shape: " + str(self._dim_lens))

    def write_hdf5(
        self,
        h5fn=None,
        chunks=None,
        scan_chunks=(16, 16),
        im_chunks=(16, 16),
        compression_opts=4,
        compression="gzip",
        repack=True,
        ow=False,
        embed_header_file=True,
        embed_source=False,
        count_extra=True,
        mask=None,
        allow_memmap=True,
        func=None,
        func_kwargs=None,
        MiB_max=512,
        progress_bar=True,
    ):
        """
        Convert the dataset into an hdf5 file composed of EMD [1] data sets.

        Parameters
        ----------
        h5fn : str
            hdf5 output filename. If None, the first binary filename is used
            with a '.hdf5' extension.
        chunks : tuple of length equalling dimensionality of data
            fpd dataset chunking for all axes. Note, this takes precedence
            over any values set for scan and image chunking. See Notes.
        scan_chunks : length 2 tuple
            y and x fpd data chunking for scan dimensions. See Notes.
        im_chunks : length 2 tuple
            y and x fpd data chunking for image dimensions. See Notes.
        compression_opts : int
            Dataset gzip compression level in interval [0,9].
        compression : str
            Main dataset compression method, one of ['gzip', 'LZ4'].
            LZ4 is much faster (~2x) than gzip at the cost of slightly larger files.
            Use of LZ4 requires the HDF5 plugins. In this package, plugins are
            obtained from the hdf5plugin package. This parameter only affects
            the main dataset. All other datasets are compressed with gzip so
            that they may always be read.
        repack : bool
            If True, fpd_data will be first written to an uncompressed temporary
            file, before being copied to the final file. See notes.
        ow : bool
            If True, pre-existing hdf5 output files are overwritten without asking.
        embed_header_file : bool
            If True, the Merlin header file is embedded in the hdf5 file.
        embed_source : bool
            If True, the source code used to generate the hdf5 file is embedded.
        count_extra : bool
            If True, extra images in input data are counted and reported.
        mask : None or 2-D bool array
            If not None, the detector images where mask is True are replaced
            with cubic interpolated values. The mask and interpolation parameters
            are stored as a dataset and attribute.
        allow_memmap : bool
            If True, files will be memmory mapped to increase conversion speed.
        func : None or function
            If not None, a function that applies out-of-place to each image. The
            function should be of the form f(image, scanYind, scanXind, **func_kwargs). Note that
            the result is coerced to be of the same dtype as the original data.
            If mask is not None, masking happens before the application of func.
            Additional arguments may be passed by the `func_kwargs` keyword.
            See notes.
        func_kwargs : None or dict
            A dictionary of keyword arguments passed to `func`.
        MiB_max : scalar
            Maximum number of mebibytes (1024**2 bytes) allowed for file access. This is
            a soft limit and controls how many chunks are read for calculating image sums.
            The minimum number of chunks is currently 1, so this limit might be breached
            for very large chunk sizes.
        progress_bar : bool
            If True, progress bars are printed.

        Notes
        -----
        `chunks` sets the chunking for all axes of the fpd dataset and takes
        precedence over the scan and image chunking parameters. If `chunks`
        is None, the fpd data chunking is set by `scan_chunks` and `im_chunks`,
        with the entire colour axis (when non-singular) as one chunk.

        If the file can be memory mapped, this is used to access the image data
        and write it to the hdf5 file one chunk at a time. If this is not possible,
        to efficiently chunk in the scan axes repacking is used. If `repack` is True,
        the fpd data is written to a temporary file, then copied to the compressed /
        chunked file. The copying is done by only loading one chunk at a time, so
        this is RAM efficient but disk space expensive. Increasing chunking in this
        way can significantly reduced the final file size and improve processing
        speed.

        If repacking is disabled, setting the scan axis chunks to anything other
        than (1, 1) will be slow. Because the binary Merlin data format is
        sequential, for chunks greater than 1 in the scan axes, h5py would have
        to deal with many successive writes to each chunk, probably uncompressing
        and recompressing the data each time.

        As an example of the `func` usage, the code below shows how to move each
        image in a dataset by a different amount. The amount is set by the syx array
        (not explicitly set here):

        >>> from fpd.synthetic_data import shift_im
        >>> def shift_func(image, scanYind, scanXind, shift_array):
        >>>     syx = shift_array[:, scanYind, scanXind]
        >>>     new_im = shift_im(image, syx)
        >>>     return new_im

        >>> mb.write_hdf5(h5fn='shifted.hdf5', func=shift_func, func_kwargs={'shift_array': syx})

        The shift array but could be from a center of mass analysis of the memory
        mapped file (e.g., fpd.fpd_processing.center_of_mass), or any other method.

        References
        ----------
        [1] http://emdatasets.lbl.gov/spec/

        """

        file_format_version = __version__
        fpd_version_attr_name = "fpd_version"

        # default emd parameters
        emd_ds_default = {
            "data": None,
            "compression": "gzip",
            "compression_opts": compression_opts,
            "shuffle": True,
            "fillvalue": np.nan,
            "track_times": True,
            "fletcher32": True,
        }

        # process compression options
        compression_modes = ["gzip", "LZ4"]

        if compression not in compression_modes:
            raise NotImplementedError(
                "'compression' (%s) must be one of: " % (compression)
                + str(compression_modes)
            )
        if compression == "gzip":
            fpd_ds_opts = emd_ds_default
        elif compression == "LZ4":
            fpd_ds_opts = emd_ds_default.copy()
            fpd_ds_opts.update(dict(hdf5plugin.LZ4().items()))

        # make an empty dictionary for function but no args
        if (func is not None) and (func_kwargs is None):
            func_kwargs = {}

        # set chunking
        if chunks is not None:
            # use chunks directly
            if not len(chunks) == len(self._dim_lens):
                message = "'chunks' length (%d) does not match data length (%d)." % (
                    len(chunks),
                    len(self._dim_lens),
                )
                raise Exception(message)
        else:
            # check for lower precedence chunking parameters
            chunks = scan_chunks + (1,) * (len(self._dim_lens) - 4) + im_chunks

        # coerce chunks to be <= shape
        chunks = tuple([min(chk, shp) for chk, shp in zip(chunks, self._dim_lens)])
        _logger.info("FPD data chunking: " + str(chunks))

        # condition filename
        if h5fn is None:
            h5fn = os.path.splitext(self._binfns[0])[0] + ".hdf5"
        if not ow and os.path.isfile(h5fn):
            reply = input(
                "File '%s' already exists. Overwrite? ([y]/n): "
                % (os.path.abspath(h5fn))
            )
            if reply.strip().lower() not in ["y", "yes", ""]:
                print("Doing nothing.")
                return
        _logger.info("Output: " + os.path.abspath(h5fn))

        # if memmap is possible, use it for reading chunks
        memmap_possible = allow_memmap and self._memmap_check(throw_error=False)
        if memmap_possible:
            m = "Memory mapping the file is possible, so repacking is disabled."
            _logger.info(m)
            repack = False
            mm = self.get_memmap()

        if repack:
            m = "This will use a lot of temporary disk space, set 'repack=False' if this is an issue."
            _logger.warning(m)
            # make temp file
            h5fn_tmp = h5fn + ".tmp"
            h5f_tmp = h5py.File(h5fn_tmp, "w", driver=None)
            h5f_tmp_fpd = h5f_tmp.create_dataset(
                "fpd_data", shape=self._dim_lens, dtype=self._det_dtype
            )
        elif not memmap_possible:
            # warn about chunking
            non_data_chunks = np.array(chunks[:-2])
            if (non_data_chunks > 1).any():
                m = "This could be very slow. Try setting 'repack=True'."
                _logger.warning(m)

        # create main hdf5
        with h5py.File(h5fn, "w", driver=None) as h5f:
            # http://docs.h5py.org/en/latest/faq.html?highlight=core
            # http://docs.h5py.org/en/latest/high/file.html?highlight=file%20drivers

            # EMD file version
            h5f.attrs["version_major"] = 0
            h5f.attrs["version_minor"] = 2

            # file format version
            vmajor, vminor, vpatch = [int(x) for x in file_format_version.split(".")]
            h5f.attrs[fpd_version_attr_name] = file_format_version
            h5f.attrs["fpd_version_major"] = vmajor
            h5f.attrs["fpd_version_minor"] = vminor
            h5f.attrs["fpd_version_patch"] = vpatch

            # fpd package version
            vmajor, vminor, vpatch = [int(x) for x in fpd_pkg_version.split(".")]
            h5f.attrs["fpd_pkg_version"] = fpd_pkg_version
            h5f.attrs["fpd_pkg_version_major"] = vmajor
            h5f.attrs["fpd_pkg_version_minor"] = vminor
            h5f.attrs["fpd_pkg_version_patch"] = vpatch

            ### Create groups / datasets and populate all except binary FPD data.
            # fpd experiment group (top level)
            fpd_expt_grp = h5f.create_group(name="fpd_expt")

            # source
            if embed_source:
                _embed_source_in_hdf5_group(fpd_expt_grp)

            # header file
            if self._hdrfn and embed_header_file:
                _embed_header_in_hdf5_group(fpd_expt_grp, self._Mhfp)

            # sum_im (sum of all detector signals at each scan pixel)
            det_pix_max_val = 2**self._Mbhp.bitdepth_data - 1
            fpd_sum_im_dataset = _create_sum_im_hdf5_dataset(
                fpd_expt_grp,
                emd_ds_default,
                det_pix_max_val,
                self._dim_lens,
                self._dim_names,
                self._dim_axes,
                self._dim_units,
            )

            # sum_dif (sum of each detector pixel across all scan pixels)
            fpd_sum_dif_dataset = _create_sum_dif_hdf5_dataset(
                fpd_expt_grp,
                emd_ds_default,
                det_pix_max_val,
                self._dim_lens,
                self._dim_names,
                self._dim_axes,
                self._dim_units,
            )

            # Write DM files
            _write_dm_files_to_hdf5(
                fpd_expt_grp,
                emd_ds_default,
                self._dmfns,
                self._dim_lens,
                self._dim_names,
                self._dim_axes,
                self._dim_units,
            )

            # Write TIA files
            _write_tia_files_to_hdf5(
                fpd_expt_grp,
                emd_ds_default,
                self._tiafns,
                self._dim_lens,
                self._dim_names,
                self._dim_axes,
                self._dim_units,
            )

            # FPD binary header string.
            fpd_binary_hdr_str_dataset = _create_bin_header_str_dataset(
                fpd_expt_grp,
                emd_ds_default,
                self._dim_lens,
                self._dim_names,
                self._dim_axes,
                self._dim_units,
            )
            # Create FPD data.
            fpd_dataset = _create_fpd_data_hdf5_dataset(
                fpd_expt_grp,
                fpd_ds_opts,
                self._det_dtype,
                self._dim_lens,
                self._dim_names,
                self._dim_axes,
                self._dim_units,
                chunks,
            )

            # Binary header params
            bhd_dss = _create_bin_header_param_hdf5_dataset(
                fpd_expt_grp,
                emd_ds_default,
                det_pix_max_val,
                self._dim_lens,
                self._dim_names,
                self._dim_axes,
                self._dim_units,
                self._Mbhp.params,
            )

            # mask prep
            if mask is not None:
                mask = np.asarray(mask, dtype=bool)
                mask = mask
                if mask.any() is False:
                    _logger.warning("Mask is all False, mask has been set to None")
                    mask = None
            if mask is not None:
                # create dataset
                fpd_mask_dataset = _create_mask_hdf5_dataset(
                    fpd_expt_grp,
                    emd_ds_default,
                    self._dim_lens,
                    self._dim_names,
                    self._dim_axes,
                    self._dim_units,
                )
                fpd_mask_dataset[...] = mask
                fpd_mask_dataset.parent.attrs["mask_method"] = (
                    "scipy.interpolate.CloughTocher2DInterpolator"
                )

                # set up for interpolation
                CI = CubicImageInterpolator(mask)

            # scanXbindata is dimension in bin file >= scanX due to row_end_skip
            scanXbindata = self._scanX + self._row_end_skip

            # (scanY, scanXbindata[, colourN])
            ims_inc_extra_shape = self._ims_shape[:]
            ims_inc_extra_shape[1] = scanXbindata
            ims_inc_extra_shape = tuple(ims_inc_extra_shape)
            total_no_ims_inc_extra = np.prod(ims_inc_extra_shape)

            ### Process binary file(s), parsing header each time
            i = 0  # index in scan, including row ends
            j = 0  # index of read data
            k = 0  # index of saved data
            stop = False
            pbar = tqdm(
                total=self._total_no_ims,
                mininterval=0,
                leave=True,
                unit="images",
                disable=(not progress_bar),
            )
            for binfn_index, binfn in enumerate(self._binfns):
                if stop == True:
                    # needed for multiple single images and breaking loop over iterator
                    break
                # loop over input files
                with open(binfn, "rb") as f:
                    fp = f  # file reference

                    # create partial for looping
                    if self._hdrfn:
                        read_binary_func = partial(
                            _read_header_and_data,
                            fp,
                            self._Mhfp.fmt_ver,
                            self._im_bytesize,
                            return_bin_data=(not memmap_possible),
                        )

                    else:
                        read_binary_func = partial(
                            _read_header_and_data,
                            fp,
                            1,
                            self._im_bytesize,
                            return_bin_data=(not memmap_possible),
                        )

                    bin_file_iter = iter(read_binary_func, "")
                    for hdr_str, hdr_params, data_buffer in bin_file_iter:
                        if j < self._ds_start_skip:
                            pass
                        else:
                            try:
                                sYi_sXi_opCi = np.unravel_index(
                                    i, ims_inc_extra_shape, "C"
                                )
                                scanYind, scanXind = sYi_sXi_opCi[:2]
                                s = np.s_[sYi_sXi_opCi]
                                if self._row_end_skip and scanXind > self._scanX - 1:
                                    i += 1
                                    # skip row_end_skip
                                    pass
                                else:
                                    if not memmap_possible:
                                        # we need to process images here
                                        im_array = _raw_merlin_to_array(
                                            data_buffer,
                                            self._det_dtype,
                                            self._Mbhp.bitdepth_data,
                                            self._Mbhp.bitdepth_bin,
                                            self._Mbhp.raw_mode,
                                            self._detY,
                                            self._detX,
                                        )
                                        # these are repeated below in a slightly different form
                                        # for memmappable files. They could be combined into one function.
                                        if mask is not None:
                                            CI.interpolate_image(
                                                image=im_array,
                                                clip_min=0,
                                                clip_max=det_pix_max_val,
                                                in_place=True,
                                            )
                                        if func is not None:
                                            im_array = func(
                                                im_array,
                                                scanYind,
                                                scanXind,
                                                **func_kwargs,
                                            )
                                            im_array = im_array.astype(
                                                self._det_dtype, copy=False
                                            )
                                        if repack:
                                            h5f_tmp_fpd[s + (np.s_[:],) * 2] = im_array
                                        else:
                                            fpd_dataset[s + (np.s_[:],) * 2] = im_array

                                    # binary header string
                                    fpd_binary_hdr_str_dataset[s] = hdr_str
                                    # bin hdr
                                    for pk in hdr_params.keys():
                                        ds = fpd_expt_grp[hdr_params[pk]["name"]][
                                            "data"
                                        ]
                                        ds[s] = hdr_params[pk]["data"]
                                    i += 1
                                    k += 1
                                    pbar.update(1)
                            except ValueError as e:
                                # detect unravel index error for more images than data
                                e1 = (
                                    str(e) == "invalid entry in index array"
                                )  # older numpy
                                e2 = "is out of bounds" in str(e)  # newer numpy
                                if e1 or e2:
                                    if count_extra:
                                        if len(self._binfns) == 1:
                                            # single file
                                            # Remaining data after processing last row,
                                            # including row_end_skip.
                                            no_extra_images = sum(
                                                1 for _ in bin_file_iter
                                            )
                                        else:
                                            # multiple files
                                            no_extra_images = len(self._binfns) - (
                                                binfn_index + 1
                                            )
                                        m = (
                                            "%d images at end of file(s) not processed!"
                                            % (no_extra_images)
                                        )
                                        _logger.warning(m)
                                    else:
                                        m = "Extra images at end of file(s) not processed!"
                                        _logger.warning(m)
                                        m = "Use 'count_extra=True' to count extra images."
                                        _logger.info(m)
                                    stop = True
                                    break
                                else:
                                    raise
                        j += 1
            if k + 1 < self._total_no_ims:
                m = "Only %d of expected %d images read!" % (k + 1, self._total_no_ims)
                _logger.warning(m)
            pbar.close()

            if repack:
                _logger.info("Repacking data.")

                data_chunks = fpd_dataset.chunks
                data_shape = h5f_tmp_fpd.shape

                # loop over axes
                sinds, n = slice_indices(data_shape, data_chunks)

                for si in tqdm(
                    sinds, total=n, unit="chunks", disable=(not progress_bar)
                ):
                    s = tuple([slice(x[0], x[1]) for x in si])
                    fpd_dataset[s] = h5f_tmp_fpd[s]

                # close and delete file
                h5f_tmp.close()
                os.remove(h5fn_tmp)

            if memmap_possible:
                _logger.info("Packing memmaped data.")

                data_chunks = fpd_dataset.chunks
                data_shape = mm.shape

                if (mask is None) and (func is None):
                    # loop over chunks
                    sinds, n = slice_indices(data_shape, data_chunks)

                    for si in tqdm(
                        sinds, total=n, unit="chunks", disable=(not progress_bar)
                    ):
                        s = tuple([slice(x[0], x[1]) for x in si])
                        fpd_dataset[s] = mm[s]
                else:
                    # loop over chunks only in non-image axes, correcting data
                    data_chunks = fpd_dataset.chunks[:-2]
                    sinds, n = slice_indices(data_shape, data_chunks)

                    for si in tqdm(
                        sinds,
                        total=n,
                        unit="non-image chunks",
                        disable=(not progress_bar),
                    ):
                        s = tuple([slice(x[0], x[1]) for x in si])
                        nis = np.ascontiguousarray(mm[s])  # non_image_slice

                        if func is not None:
                            yyi, xxi = np.mgrid[s]
                            yyi = yyi.flatten()
                            xxi = xxi.flatten()

                        nis_shape = nis.shape
                        nis.shape = (np.prod(nis_shape[:-2]),) + nis_shape[-2:]
                        for i, im_array in enumerate(nis):
                            if mask is not None:
                                CI.interpolate_image(
                                    image=im_array,
                                    clip_min=0,
                                    clip_max=det_pix_max_val,
                                    in_place=True,
                                )
                            if func is not None:
                                scanYind, scanXind = yyi[i], xxi[i]
                                im_array = func(
                                    im_array, scanYind, scanXind, **func_kwargs
                                )
                                nis[i] = im_array.astype(self._det_dtype, copy=False)
                        nis.shape = nis_shape

                        fpd_dataset[s] = nis

                    # need to load entire image, run through filter and write as chunks
                    # loop over slices in non-image dimensions
                    # _interp_image_partial(im_array)
                    # loop over slices in image dimensions
                    # write to file

                del mm

            # generate sum images from main dataset
            from .fpd_processing import sum_dif, sum_im

            read_chunk_y, read_chunk_x = determine_read_chunks(
                fpd_dataset, MiB_max=MiB_max, det_im_processing=True
            )
            # print(read_chunk_y, read_chunk_x)
            fpd_sum_im_dataset[:] = sum_im(
                fpd_dataset, read_chunk_y, read_chunk_x, progress_bar=progress_bar
            )

            read_chunk_y, read_chunk_x = determine_read_chunks(
                fpd_dataset, MiB_max=MiB_max, det_im_processing=False
            )
            # print(read_chunk_y, read_chunk_x)
            fpd_sum_dif_dataset[:] = sum_dif(
                fpd_dataset, read_chunk_y, read_chunk_x, progress_bar=progress_bar
            )

        return h5fn

    def _memmap_check(self, throw_error=True):
        try:
            if len(self._binfns) > 1:
                raise NotImplementedError(
                    "Only single multi-image binary files may be memory mapped."
                )

            if self._Mbhp.raw_mode:
                raise NotImplementedError("Raw data cannot be memory mapped.")
                # This could be done, but it would be up to the user to unravel the
                # images when in-memory.

            if self._Mbhp.bitdepth_data == 1:
                raise NotImplementedError("1 bit data cannot be memory mapped.")

            if self._Mbhp.header_bytesize % self._Mbhp.bitdepth_bin != 0:
                raise NotImplementedError(
                    "The header is not an integer multiple of the dtype"
                )
        except NotImplementedError:
            print(
                "INFO: The only way to access the data is to index the class's array-like"
            )
            print(
                "INFO: interface to read the data into memory, or convert the data to an hdf5"
            )
            print("INFO: file with the `write_hdf5` method.")
            if throw_error:
                raise
            else:
                return False
        return True

    def get_memmap(self):
        """
        Returns a memory mapped array for out-of-core data access.
        Delete the memmap instance to close the file.

        Returns
        -------
        mm : numpy.core.memmap.memmap
            Memory mapped on-disk array. To close the file, delete the object
            with `del mm`.

        Examples
        --------
        >>> from fpd.fpd_file import MerlinBinary, DataBrowser  +SKIP
        >>> import numpy as np  +SKIP

        >>> mb = MerlinBinary(binary_filename, header_filename, dm_filename)    +SKIP

        >>> mm = mb.get_memmap()    +SKIP

        This may be plotted with a navigation image generated from the data:
        >>> nav_im = mm.sum((-2,-1))    +SKIP
        or a blank image:
        >>> nav_im = np.zeros(mm.shape[:2]) +SKIP
        >>> b = DataBrowser(mm, nav_im=nav_im)  +SKIP

        Notes
        -----
        Based on https://gitlab.com/fpdpy/fpd/issues/16#note_72345827

        """

        self._memmap_check()

        header_pixels = self._Mbhp.header_bytesize // (self._Mbhp.bitdepth_bin // 8)
        image_pixels = self._detY * self._detX + header_pixels
        offset = self._ds_start_skip * (self._Mbhp.header_bytesize + self._im_bytesize)

        shape = list(self.shape)
        shape[1] = shape[1] + self._row_end_skip
        shape = shape[:-2] + [image_pixels]
        shape = tuple(shape)

        mm = np.memmap(
            self._binfns[0], dtype=self._det_dtype, mode="r", offset=offset, shape=shape
        )

        end_ind = -self._row_end_skip
        if end_ind == 0:
            end_ind = None
        mm = mm[:, :end_ind, ..., header_pixels:]
        # invert so origin is at top
        mm = mm.reshape(self.shape)[..., ::-1, :]
        return mm

    def __getitem__(self, key):
        # make into list (default is int or slice object or tuples thereof)
        if type(key) is not tuple:
            key = list([key])
        else:
            key = list(key)

        # only ints, slices, or ellipsis
        key_types = [type(k) for k in key]
        for kt in key_types:
            if kt not in [slice, int, type(Ellipsis)]:
                # only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
                raise IndexError(
                    "only integers, slices (`:`), and ellipsis (`...`) are valid indices"
                )

        # at most one ellipsis
        n_ellipsis = sum([kt == type(Ellipsis) for kt in key_types])
        if n_ellipsis > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")

        # check slice axes length
        if len(key) - n_ellipsis > self.ndim:
            raise IndexError("too many indices for array")

        # handle ellipsis
        if n_ellipsis:
            kti = key_types.index(type(Ellipsis))
            if len(key) == self.ndim + 1:
                # remove if not needed
                del key_types[kti]
                del key[kti]
            else:
                # replace ellipsis
                del key[kti]
                new_axis = slice(None, None, None)
                naxes_to_add = self.ndim - len(key)
                for i in range(naxes_to_add):
                    key.insert(kti, new_axis)

        # TODO
        # negative indices

        # homogenise keys to be all complete slices
        for i, k in enumerate(key):
            # convert ints to slices
            if type(k) is int:
                k = slice(k, k + 1, None)
                key[i] = k

            # replace None in slices with ints
            sss = k.start, k.stop, k.step
            start, stop, step = sss
            if start is None:
                start = 0
            if stop is None:
                stop = self.shape[i]
            if step is None:
                step = 1
            if start > stop:
                raise IndexError("start index must be greater than stop index")
            stop = min(stop, self.shape[i])
            sss = start, stop, step
            if min(sss) < 0:
                # only positive values
                raise IndexError("indices must be positive")
            k = slice(*sss)
            key[i] = k

        # add missing slices
        missing_ndims = self.ndim - len(key)
        if missing_ndims > 0:
            for i in range(len(key), len(key) + missing_ndims):
                key.append(slice(0, self.shape[i], 1))

        # determine indices and shape
        index_axes = [range(k.start, k.stop, k.step) for k in key]
        index_shape = tuple([len(i) for i in index_axes])

        # determine if detector images are indexed or full
        full_detector_images = all(
            [i1 == i2 for i1, i2 in zip(index_shape[-2:], self.shape[-2:])]
        )

        # make dataset array
        fpd_dataset = np.empty(shape=index_shape, dtype=self.dtype)

        # generate flat on disk indices
        from itertools import product, starmap
        from functools import partial

        rmip = partial(np.ravel_multi_index, dims=self.shape[:-2])

        def rmip_args(*args):
            r, c = args[:2]
            # ignoring extra pixels
            disk_dataset_flat_index = rmip(args)
            # including extra pixels
            disk_dataset_flat_index += self._ds_start_skip + r * self._row_end_skip
            return disk_dataset_flat_index

        flat_indices_iterator = starmap(rmip_args, product(*index_axes[:-2]))
        # this should be in on disk order

        # reshape output data to n, detY, detX
        fpd_dataset.shape = (np.prod(index_shape[:-2]),) + index_shape[-2:]

        # loop over data
        ### Process binary file(s), parsing header each time
        sel_ind = flat_indices_iterator.__next__()
        i = 0  # index in scan, including row ends
        k = 0  # index of saved data
        stop = False
        pbar = tqdm(
            total=np.prod(index_shape[:-2]),
            mininterval=0,
            leave=True,
            unit="images",
            disable=(not self.array_interface_progress_bar),
        )
        im_indices = np.ogrid[index_axes[-2:]]
        for binfn_index, binfn in enumerate(self._binfns):
            if stop == True:
                # needed for multiple single images and breaking loop over iterator
                break
            # loop over input files
            with open(binfn, "rb") as f:
                fp = f  # file reference

                # create partial for looping
                if self._hdrfn:
                    read_binary_func = partial(
                        _read_header_and_data, fp, self._Mhfp.fmt_ver, self._im_bytesize
                    )
                else:
                    read_binary_func = partial(
                        _read_header_and_data, fp, self._Mbhp.fmt_ver, self._im_bytesize
                    )
                bin_file_iter = iter(read_binary_func, "")
                for hdr_str, hdr_params, data_buffer in bin_file_iter:
                    # process bin_data
                    if i == sel_ind:
                        im_array = _raw_merlin_to_array(
                            data_buffer,
                            self._det_dtype,
                            self._Mbhp.bitdepth_data,
                            self._Mbhp.bitdepth_bin,
                            self._Mbhp.raw_mode,
                            self._detY,
                            self._detX,
                        )

                        # fill array
                        if full_detector_images:
                            fpd_dataset[k] = im_array
                        else:
                            fpd_dataset[k] = im_array[tuple(im_indices)]
                        k += 1
                        pbar.update(1)
                        try:
                            sel_ind = flat_indices_iterator.__next__()
                        except StopIteration:
                            stop = True
                            break
                    i += 1
        pbar.close()

        # reshape output data
        fpd_dataset.shape = index_shape
        return np.squeeze(fpd_dataset)

    def to_array(self, count_extra=True, read_max=None, progress_bar=True):
        """
        Convert the dataset into an in-memory numpy array.

        Parameters
        ----------
        count_extra : bool
            If True, extra images in input data are counted and reported.
        read_max : int or None
            If not None, only `read_max` images non-skip will be read.
            If not None, `count_extra` is disabled.
        progress_bar : bool
            If True, progress bars are printed.

        Returns
        -------
        fpd_dataset : ndarray
            Dataset image array.
        """
        if read_max is not None:
            count_extra = False

        fpd_dataset = np.empty(self._dim_lens, dtype=self._det_dtype)

        # scanXbindata is dimension in bin file >= scanX due to row_end_skip
        scanXbindata = self._scanX + self._row_end_skip

        # (scanY, scanXbindata[, colourN])
        ims_inc_extra_shape = self._ims_shape[:]
        ims_inc_extra_shape[1] = scanXbindata
        ims_inc_extra_shape = tuple(ims_inc_extra_shape)
        total_no_ims_inc_extra = np.prod(ims_inc_extra_shape)

        ### Process binary file(s), parsing header each time
        i = 0  # index in scan, including row ends
        j = 0  # index of read data
        k = 0  # index of saved data
        stop = False
        pbar = tqdm(
            total=self._total_no_ims,
            mininterval=0,
            leave=True,
            unit="images",
            disable=(not progress_bar),
        )
        for binfn_index, binfn in enumerate(self._binfns):
            if stop == True:
                # needed for multiple single images and breaking loop over iterator
                break
            # loop over input files
            with open(binfn, "rb") as f:
                fp = f  # file reference

                # create partial for looping
                read_binary_func = partial(
                    _read_header_and_data, fp, self._Mhfp.fmt_ver, self._im_bytesize
                )
                bin_file_iter = iter(read_binary_func, "")
                for hdr_str, hdr_params, data_buffer in bin_file_iter:
                    if j < self._ds_start_skip:
                        pass
                    else:
                        try:
                            sYi_sXi_opCi = np.unravel_index(i, ims_inc_extra_shape, "C")
                            scanYind, scanXind = sYi_sXi_opCi[:2]
                            s = np.s_[sYi_sXi_opCi]
                            if self._row_end_skip and scanXind > self._scanX - 1:
                                i += 1
                                # skip row_end_skip
                                pass
                            else:
                                # process bin_data
                                bitdepth_data = self._Mbhp.bitdepth_data
                                bitdepth_bin = self._Mbhp.bitdepth_bin
                                raw_mode = self._Mbhp.raw_mode

                                im_array = _raw_merlin_to_array(
                                    data_buffer,
                                    self._det_dtype,
                                    bitdepth_data,
                                    bitdepth_bin,
                                    raw_mode,
                                    self._detY,
                                    self._detX,
                                )
                                fpd_dataset[s + (np.s_[:],) * 2] = im_array
                                i += 1
                                k += 1
                                pbar.update(1)
                                if k == read_max:
                                    m = (
                                        "Read %d images. Set 'read_max=None' to read all images."
                                        % (read_max)
                                    )
                                    _logger.info(m)
                                    stop == True
                                    break
                        except ValueError as e:
                            # detect unravel index error for more images than data
                            e1 = str(e) == "invalid entry in index array"  # older numpy
                            e2 = "is out of bounds" in str(e)  # newer numpy
                            if e1 or e2:
                                if count_extra:
                                    if len(self._binfns) == 1:
                                        # single file
                                        # Remaining data after processing last row,
                                        # including row_end_skip.
                                        no_extra_images = sum(1 for _ in bin_file_iter)
                                    else:
                                        # multiple files
                                        no_extra_images = len(self._binfns) - (
                                            binfn_index + 1
                                        )
                                    m = "%d images at end of file(s) not processed!" % (
                                        no_extra_images
                                    )
                                    _logger.warning(m)
                                else:
                                    m = "Extra images at end of file(s) not processed!"
                                    _logger.warning(m)
                                    m = "Use 'count_extra=True' to count extra images."
                                    _logger.info(m)
                                stop = True
                                break
                            else:
                                raise
                    j += 1
        if k + 1 < self._total_no_ims and read_max is None:
            m = "Only %d of expected %d images read!" % (k + 1, self._total_no_ims)
            _logger.warning(m)
        pbar.close()

        return fpd_dataset


def slice_indices(data_shape, data_chunks, iterator=False):
    """
    Generates indices that may be passed to a slice to chunk an array.

    Parameters
    ----------
    data_shape : tuple
        Data shape.
    data_chunks : tuple
        Data chunks in each axis.
    iterator : bool
        If True, an iterator is returned. If False, a list is returned.

    Returns
    -------
    Tuple of sinds, n.
    sinds : iterator or list
        Start and end indices of chunks.
    n : integer
        total number of chunks.

    Examples
    --------
    >>> import numpy as np
    >>> from tqdm import tqdm

    >>> data_chunks =  (17, 32, 64)
    >>> data_shape = (128, 128, 256)

    >>> sinds, n = slice_indices(data_shape, data_chunks)

    >>> for si in tqdm(sinds, total=n, unit='chunks'):
    ...     s = tuple([slice(x[0],x[1]) for x in si])
    ...     # ndarray[s]


    """

    inds = [
        np.arange(0, np.ceil(float(shp) / cks).astype(int) + 1, dtype=int) * cks
        for cks, shp in zip(data_chunks, data_shape)
    ]
    # for int x 2 (start, stop) for each axis
    sss = [np.array([x[:-1], x[1:]]).T for x in inds]
    # clip to max size
    sss = [d.clip(max=lim) for d, lim in zip(sss, data_shape)]
    # convert to list
    sss = [x.tolist() for x in sss]
    # calculate total chuncks
    n = np.prod([len(i) for i in sss])
    # generate iterator
    sinds = product(*sss)

    if not iterator:
        # convert to list
        sinds = list(sinds)

    return sinds, n


def _get_hdf5_file_from_obj(fpg):
    """
    Gets hdf5 file from any one of filename, file, group or dataset.

    Parameters
    ----------
    fpg : str, file, group or dataset
        Filename, or hdf5 file, group or dataset.

    Returns
    -------
    closef : bool
        True if file should be closed to return to original state.

    h5f : hdf5 file

    """

    closef = False
    h5fnp_type = type(fpg)

    if h5fnp_type is str:
        h5f = h5py.File(fpg, "r")
        closef = True
    elif h5fnp_type == h5py._hl.files.File:
        h5f = fpg
    elif h5fnp_type == h5py._hl.group.Group:
        h5f = fpg.file
    elif h5fnp_type == h5py._hl.dataset.Dataset:
        h5f = fpg.file
    else:
        raise Exception("Unknown type:", h5fnp_type)

    return closef, h5f


def _check_fpd_file(
    fpg, min_version=None, max_version=None, raise_exception=True, silent=False
):
    """
    Check if hdf5 file is a recognised FPD format.

    Parameters
    ----------
    fpg : str, file, group or dataset
        Filename, or hdf5 file, group or dataset.
    min_version : str
        Minimum FPD version.
    max_version : str
        Maximum FPD version.
    raise_exception : bool
        If True, raise and exception. Else, a warning is issued.
    silent : bool
        if True, no messages are printed and no exceptions raised.

    Returns
    -------
    tuple of (isfpd, isknown, vs):

    isfpd : boolean
        If True, file is either an FPD file or a PED file.
    isknown : boolean
        If True, file version is known.
    vs : string
        File format version string.

    """

    try:
        closef, h5f = _get_hdf5_file_from_obj(fpg)
    except:
        # not an hdf5 file
        return False, False, ""

    fpd_file = False
    PED_file = False
    try:
        vs = h5f.attrs["fpd_version"]
        fpd_file = True
    except KeyError:
        try:
            vs = h5f.attrs["PED_version"]
            PED_file = True
        except KeyError:
            if not silent:
                print("'%s' is not a recognised FPD file." % (h5f.file.filename))
            vs = ""
            bs = (False, vs)
    isfpd = fpd_file or PED_file

    if fpd_file:
        if min_version is None:
            min_version = _min_version
        if max_version is None:
            max_version = __version__

        if Version(min_version) > Version(vs) > Version(max_version):
            # version understood
            if not silent:
                print(
                    "FPD version '%s' in file '%s' is not supported"
                    % (vs, h5f.file.filename)
                )
            bs = (False, vs)
        else:
            bs = (True, vs)
    elif PED_file:
        PED_version_min = "0.4.1"
        PED_version_max = "0.4.1"

        if min_version is None:
            min_version = PED_version_min
        if max_version is None:
            max_version = PED_version_max

        if Version(min_version) > Version(vs) > Version(max_version):
            # version understood
            if not silent:
                print(
                    "PED version '%s' in file '%s' is not supported"
                    % (vs, h5f.file.filename)
                )
            bs = (False, vs)
        else:
            bs = (True, vs)

    if closef:
        h5f.close()

    # exception / warnings
    if not bs[0] and not silent:
        m = "Not a valid FPD file. Version: %s." % (bs[1])
        if raise_exception:
            _logger.error(m)
            raise Exception(m)
            # + '\n' + 'Set `raise_exception=False` to ignore tags.')
        else:
            _logger.warning(m)
            # _logger.warning('Continuing anyway. Set `raise_exception=True` to force exception')

    isknown, vs = bs
    return (isfpd, isknown, vs)


def find_hdf5_objname_by_attribute(fpg, attr_name, attr_val=None, fpd_check=True):
    """
    Returns a list of object names in hdf5 object with specified
    attribute name and, optionally, attribute value.

    Parameters
    ----------
    fpg : str, file, group or dataset
        Filename, or hdf5 file, group or dataset.
    attr_name : str
        Attribute name.
    attr_val : str or None
        Atribute value to match. If not None, returned objects have
        attr_name with attr_val. If None, all attributes matching
        attr_name are returned.
    fpd_check : bool
        If True, the file format version is checked.

    """

    if fpd_check:
        _ = _check_fpd_file(fpg)

    closef = False
    h5fnp_type = type(fpg)
    if h5fnp_type is str:
        h5f = h5py.File(fpg, "r")
        closef = True
    elif h5fnp_type == h5py._hl.files.File:
        h5f = fpg
    elif h5fnp_type == h5py._hl.group.Group:
        h5f = fpg
    elif h5fnp_type == h5py._hl.dataset.Dataset:
        h5f = fpg
    else:
        raise Exception("Unknown type:", h5fnp_type)

    obj_names = []

    def _append_if_attr(name):
        if attr_name in h5f[name].attrs and (
            not attr_val or h5f[name].attrs[attr_name] == attr_val
        ):
            obj_names.append(name)

    h5f.visit(_append_if_attr)

    if closef:
        h5f.close()
    return obj_names


def _dm_tia_tags_to_dict(
    fpg,
    fpd_check=True,
    group_attr="dm_group_type",
    group_attr_val=1,
    group_fn="dm_filename",
):
    """
    read tags from DM / TIA files and return as dict

    used by:
        hdf5_dm_tags_to_dict
        hdf5_tia_tags_to_dict
    """

    if fpd_check:
        _ = _check_fpd_file(fpg)

    closef, h5f = _get_hdf5_file_from_obj(fpg)

    group_paths = find_hdf5_objname_by_attribute(h5f, group_attr, group_attr_val, False)
    n_files = len(group_paths)
    print("Found %d file(s)." % (n_files))

    tagds = []
    filenames = []
    for dm_grp in group_paths:
        fn_hdf5 = h5f[dm_grp].attrs[group_fn]

        tag_grp = h5f[dm_grp + "/tags"]
        tag_grp_flat = _flatten(tag_grp, "", "/")
        tag_grp_flat_d = dict([(k, v[...]) for k, v in tag_grp_flat.items()])
        tagd = _unflatten(tag_grp_flat_d, "/")

        tagds.append(tagd)
        filenames.append(fn_hdf5)

    if closef:
        h5f.close()
    return tagds, group_paths, filenames


def hdf5_dm_tags_to_dict(fpg, fpd_check=True):
    """
    Return hdf5 dm tag groups as list of dictionaries (one per file),
    along with lists of group paths and original dm filenames.

    Parameters
    ----------
    fpg : str, file, group or dataset
        Filename, or hdf5 file, group or dataset.
    fpd_check : bool
        If True, the file format version is checked.

    Returns
    -------
    tagds : list of dicts
        Unflattened tag dictionary.
    dm_group_paths : list of str
        Object names.
    dm_filenames : list of str
        Original filenames.

    """

    rtn = _dm_tia_tags_to_dict(
        fpg,
        fpd_check,
        group_attr="dm_group_type",
        group_attr_val=1,
        group_fn="dm_filename",
    )
    tagds, dm_group_paths, dm_filenames = rtn

    return tagds, dm_group_paths, dm_filenames


def hdf5_tia_tags_to_dict(fpg, fpd_check=True):
    """
    Return hdf5 tia tag groups as list of dictionaries (one per file),
    along with lists of group paths and original tia filenames.

    Parameters
    ----------
    fpg : str, file, group or dataset
        Filename, or hdf5 file, group or dataset.
    fpd_check : bool
        If True, the file format version is checked.

    Returns
    -------
    tagds : list of dicts
        Unflattened tag dictionary.
    tia_group_paths : list of str
        Object names.
    tia_filenames : list of str
        Original filenames.

    """

    rtn = _dm_tia_tags_to_dict(
        fpg,
        fpd_check,
        group_attr="tia_group_type",
        group_attr_val=1,
        group_fn="tia_filename",
    )
    tagds, tia_group_paths, tia_filenames = rtn

    return tagds, tia_group_paths, tia_filenames


def _dm_tia_to_bin(
    fpg,
    fns=None,
    fpd_check=True,
    ow=False,
    group_attr="dm_group_type",
    group_attr_val=1,
    group_fn="dm_filename",
):
    """
    Extract DM / TIA binary blobs and write to file.

    used by:
        hdf5_dm_to_bin
        hdf5_tia_to_bin
    """

    if fpd_check:
        _ = _check_fpd_file(fpg)

    closef, h5f = _get_hdf5_file_from_obj(fpg)

    group_paths = find_hdf5_objname_by_attribute(h5f, group_attr, group_attr_val, False)
    n_files = len(group_paths)
    print("Found %d file(s)." % (n_files))

    # prepare zipped list of tuples
    if fns is None:
        fn_gn = zip((None,) * n_files, group_paths)
    else:
        # type(fns) is str:
        fn_gn = zip(fns, group_paths)

    for fn, grp in fn_gn:
        bin = h5f[grp + "/bin"][...]
        fn_hdf5 = h5f[grp].attrs[group_fn]
        if fn is None:
            fn = fn_hdf5
        else:
            # parse filename
            _, fn_hdf5_ext = os.path.splitext(fn_hdf5)
            fn_fn, fn_ext = os.path.splitext(fn)
            if fn_ext != fn_hdf5_ext:
                fn = fn_fn + fn_hdf5_ext

        if not ow and os.path.isfile(fn):
            reply = input(
                "File '%s' already exists. Overwrite? ([y]/n): " % (os.path.abspath(fn))
            )
            if reply.strip().lower() in ["y", "yes", ""]:
                pass
            else:
                print("Doing nothing.")
                continue
        print("Extracting file in '%s', called '%s', as '%s'." % (grp, fn_hdf5, fn))
        with open(fn, "wb") as f:
            bin.tofile(f, sep="")
    if closef:
        h5f.close()


def hdf5_dm_to_bin(fpg, dmfns=None, fpd_check=True, ow=False):
    """
    Write DM binary from hdf5 file with output filename given by dmfns.

    Parameters
    ----------
    fpg : str, file, group or dataset
        Filename, or hdf5 file, group or dataset.
    dmfns : None, list of str and None
        DM output filename(s). If None, the original filenames at time
        of hdf5 creation are used. If a list, non-None entries will be
        used as new filenames. New filenames are parsed to ensure the
        correct extension is used.
    fpd_check : bool
        If True, the file format version is checked.

    """

    _dm_tia_to_bin(
        fpg,
        dmfns,
        fpd_check,
        ow,
        group_attr="dm_group_type",
        group_attr_val=1,
        group_fn="dm_filename",
    )


def hdf5_tia_to_bin(fpg, tiafns=None, fpd_check=True, ow=False):
    """
    Write TIA binary from hdf5 file with output filename given by tiafns.

    Parameters
    ----------
    fpg : str, file, group or dataset
        Filename, or hdf5 file, group or dataset.
    tiafns : None, list of str and None
        TIA output filename(s). If None, the original filenames at time
        of hdf5 creation are used. If a list, non-None entries will be
        used as new filenames. New filenames are parsed to ensure the
        correct extension is used.
    fpd_check : bool
        If True, the file format version is checked.

    """

    _dm_tia_to_bin(
        fpg,
        tiafns,
        fpd_check,
        ow,
        group_attr="tia_group_type",
        group_attr_val=1,
        group_fn="tia_filename",
    )


def hdf5_fpd_to_bin(fpg, fpd_fn=None, fpd_check=True, ow=False):
    """
    Write FPD binary from hdf5 file with output filename given by fpdfn.
    Header information is omitted.

    Parameters
    ----------
    fpg : str, file, group or dataset
        Filename, or hdf5 file, group or dataset.
    fpd_fn : str
        FPD output filename. If None, the input filename is used with
        a '.bin' extension.
    fpd_check : bool
        If True, the file format version is checked.

    Notes
    -----
    Alternatively hdf5 tools can be used:
        h5dump -d /fpd_expt/fpd_data/data -b BE -o h5dump.bin ./fpd_test_data/bin_257.hdf5

    """

    if fpd_check:
        _ = _check_fpd_file(fpg)

    closef, h5f = _get_hdf5_file_from_obj(fpg)

    data = h5f["fpd_expt/fpd_data/data"][...]

    if fpd_fn is None:
        fpd_fn = os.path.splitext(h5f.filename)[0] + ".bin"
    if not ow and os.path.isfile(fpd_fn):
        reply = input(
            "File '%s' already exists. Overwrite? ([y]/n): " % (os.path.abspath(fpd_fn))
        )
        if reply.strip().lower() in ["y", "yes", ""]:
            pass
        else:
            print("Doing nothing.")
            return
    print("Writing FPD data from '%s' to binary file '%s'." % (h5f.filename, fpd_fn))
    with open(fpd_fn, "wb") as f:
        data.tofile(f, sep="")

    if closef:
        h5f.close()


def hdf5_src_to_file(fpg, src_fn=None, fpd_check=True, ow=False):
    """
    Extract code used to generate FPD hdf5 file and write to file.

    Parameters
    ----------
    fpg : str, file, group or dataset
        Filename, or hdf5 file, group or dataset.
    src_fn : str or None
        Source output filename. If None, the original filename is used.
    fpd_check : bool
        If True, the file format version is checked.

    Notes
    -----
    Alternative:
        h5dump -a /fpd_expt/src -b LE -o out2.py ./fpd_test_data/bin_257.hdf5

    """

    if fpd_check:
        _ = _check_fpd_file(fpg)

    closef, h5f = _get_hdf5_file_from_obj(fpg)

    src = str(h5f["fpd_expt/src"][...])
    src_path = h5f["fpd_expt/src"].attrs["src_path"]
    if src_fn is None:
        src_fn = os.path.split(src_path)[-1]
    if not ow and os.path.isfile(src_fn):
        reply = input(
            "File '%s' already exists. Overwrite? ([y]/n): " % (os.path.abspath(src_fn))
        )
        if reply.strip().lower() in ["y", "yes", ""]:
            pass
        else:
            print("Doing nothing.")
            return
    print(
        "Writing FPD source originally from '%s' to file '%s'."
        % (src_path, os.path.abspath(src_fn))
    )
    with open(src_fn, "w") as f:
        f.write(src)

    if closef:
        h5f.close()


def fpd_to_hyperspy(fpg, fpd_check=True, assume_yes=False, group_names=None):
    """
    Open an fpd dataset in hyperspy.

    Parameters
    ----------
    fpg : str, file, group or dataset
        Filename, or hdf5 file, group or dataset.
    assume_yes : bool
        If True, open files without asking, even if they consume large
        amounts of memory.
    group_names : None, string, or list.
        Group names to filter the return by. If None, the default set is
        returned (see notes). If not None, only groups matching group_names
        are returned.

    Returns
    -------
    A named tuple of hyperspy objects. See notes.

    Notes
    -----
    In hyperspy versions with lazy signal support, the returned object is a
    namedtuple of all filtered emd groups. The field_names are generated
    from the emd group names in the fpd file. This enables tab completion.
    If group_names is None, all groups are returned.

    In non-lazy versions of hyperspy, the named tuple only includes fpd_data
    and dif and scan images by default (group_names=None).

    Colours are not handled yet.

    Examples
    --------
    Load and select the main 4-D dataset:
    >>> from merlintools.vendor.fpd import fpd_file as fpdf
    >>> fpd_signals = fpdf.fpd_to_hyperspy(filename.hdf5)
    >>> im = fpd_signals.fpd_data

    Load only selected groups:
    >>> fpd_signals = fpdf.fpd_to_hyperspy(filename,
    ...     group_names=['fpd_data', 'fpd_sum_im', 'fpd_sum_dif'])

    Plotting in hyperspy without waiting for it to calculate a navigation image:
    >>> fpd_signals.fpd_data.plot(navigator=fpd_signals.fpd_sum_im.transpose())

    The above is very slow. It is much faster and easier to navigate the same
    data with:
    >>> db = fpdf.DataBrowser('filename.hdf5')

    If more flexibility is needed than is provided by the DataBrowser, then
    hyperspy may be faster with different file chunking. For example, setting the
    diffraction axes chunk size to the full image size, and setting the scan
    axes chunk size to something very small (maybe 1 or 2) at the point of fpd
    file creation would reduce the overhead in reading a single image. The downside
    of this is that is greatly reduces many of the data access efficiencies that the
    default fpd chunking provides, so unless you know there are fixed desired file
    access paterns, then a compromise may be worth exploring.

    """
    from hyperspy import __version__ as hsv

    hs_lv = Version(hsv)
    if (hs_lv > Version("1.5.2")) & (hs_lv < Version("1.7.0")):
        m = """HyperSpy versions (1.5.2, 1.7.0) don't return the metadata needed by this function.
Your current version is "%s". Try downgrading HyperSpy to version 1.5.2.
See https://gitlab.com/fpdpy/fpd/-/issues/41 for details.""" % (hsv)
        for mi in m.split("\n"):
            _logger.warning(mi)

    def _check_titles(titles, group_names, hs_non_lazy):
        # helper for parsing group_names against titles
        print("Detected emd groups:", titles)
        # select groups
        if group_names is None:
            if hs_non_lazy:
                group_names = ["fpd_data", "fpd_sum_im", "fpd_sum_dif"]
            else:
                group_names = titles
        elif type(group_names) is str:
            group_names = [group_names]
        non_extant_groups = [t for t in group_names if t not in titles]
        if len(non_extant_groups):
            print("The following requested groups do not exists:", non_extant_groups)
        # get indices
        indices = [titles.index(t) for t in group_names if t in titles]
        return indices

    hs_non_lazy = False
    if hs_lv < Version("1.0.0"):
        hs_non_lazy = True

    if fpd_check:
        _ = _check_fpd_file(fpg)

    if hs_non_lazy:
        # older hyperspy
        from hyperspy.signals import Image

        if not assume_yes:
            # check user wants to read data into memory
            reply = input(
                "This may consume a large amount of memory. Continue? ([y]/n): "
            )
            if reply.strip().lower() in ["y", "yes", ""]:
                pass
            else:
                print("Aborting.")
                return None

        closef, h5f = _get_hdf5_file_from_obj(fpg)
        fpd_group = h5f["fpd_expt"]
        titles = find_hdf5_objname_by_attribute(
            fpd_group, attr_name="emd_group_type", attr_val=1
        )
        indices = _check_titles(titles, group_names, hs_non_lazy)
        titles_selection = [titles[i] for i in indices]

        ss = []
        for tsi in titles_selection:
            emd_dg = fpd_group[tsi]
            ds = emd_dg["data"]
            data = ds[...]
            s = Image(data[:])
            num_axes = len(ds.shape)
            for i in range(num_axes):
                h5_ax = emd_dg["dim%d" % (i + 1)]
                # horrible hack to accommodate hyperspy's non-c ordering
                if i == 0:
                    i = 1
                elif i == 1:
                    i = 0
                if num_axes >= 4:
                    if i == num_axes - 2:
                        i = num_axes - 1
                    elif i == num_axes - 1:
                        i = num_axes - 2
                hs_ax = s.axes_manager[i]

                hs_ax.axis = h5_ax[...]
                hs_ax.scale = np.diff(hs_ax.axis)[0]
                hs_ax.name = h5_ax.attrs["name"]
                hs_ax.units = h5_ax.attrs["units"]
            ss.append(s)
        fpd_signals = namedtuple("fpd_signals", titles_selection)
        s = fpd_signals(*ss)

        if closef:
            h5f.close()
        return s
    else:
        # lazy loading in newer hyperspy
        closef, h5f = _get_hdf5_file_from_obj(fpg)
        h5f_fn = h5f.filename
        if closef:
            h5f.close()

        from hyperspy.io_plugins import emd
        from hyperspy.io import dict2signal

        ds = emd.file_reader(h5f_fn, lazy=True)
        if "General" in ds[0]["original_metadata"].keys():
            titles = [
                d["original_metadata"]["General"]["title"].split("/")[-1] for d in ds
            ]
        elif "General" in ds[0]["metadata"].keys():
            titles = [d["metadata"]["General"]["title"].split("/")[-1] for d in ds]
        else:
            raise KeyError("Unknown metadata format")
        indices = _check_titles(titles, group_names, hs_non_lazy)
        titles_selection = [titles[i] for i in indices]
        ds_selecion = [ds[i] for i in indices]

        ss = []
        for d_select in ds_selecion:
            if "attributes" in d_select.keys():
                d_select["attributes"]["_lazy"] = True
            t = [axis for axis in d_select["axes"] if "scan" in axis["name"]]
            for axis in t:
                axis["navigate"] = True
            s = dict2signal(d_select)
            s = s.as_lazy()
            ss.append(s)
        fpd_signals = namedtuple("fpd_signals", titles_selection)
        s = fpd_signals(*ss)
        return s


def fpd_to_tuple(fpg, group_names=None, nd_max=3, gigabytes_max=1, fpd_check=True):
    """
    Open an fpd dataset as a hierarchical named tuple, with all the tab
    completion goodness.

    Parameters
    ----------
    fpg : str, file, group or dataset
        Filename, or hdf5 file, group or dataset.
    group_names : None, string, or list.
        Group names to filter the return by. If None, the default set is
        returned (see notes). If not None, only groups matching group_names
        are returned.
    nd_max : int
        Maximum number of dimensions for datasets below or equal to which
        values are loaded into memory. `gigabytes_max` must also be satisfied.
    gigabytes_max : scalar
        Maximum Gb for datasets below or equal to which values are loaded
        into memory. `nd_max` must also be satisfied.
    fpd_check : bool
        if True, the hdf5 file is checked for being valid.

    Returns
    -------
    fpd_nt : namedtuple
        A named tuple of in-memory and hdf5 objects. See notes.

    Notes
    -----
    The returned object is a named tuple of optionally filtered emd groups.
    The field_names are generated from the emd group names in the fpd file.
    This enables tab completion.

    If `group_names` is None, all groups are returned. The named tuple can
    also be indexed. The order of the names in `group_names` is respected.
    If not specified, the fields are in alphabetical order (the field names
    may be accessed as usual for a namedtuple with fpd_nt._fields).

    The attributes of the dataset are also extracted ('name' and 'units'),
    as are each dimension (eg 'dim1'). The structure of the tuple is the
    following, where 'dim1' is another named tuple.

    fpd_nt.dataset_name.data
    fpd_nt.dataset_name.name
    fpd_nt.dataset_name.unit
    fpd_nt.dataset_name.dim1.data
    fpd_nt.dataset_name.dim1.name
    fpd_nt.dataset_name.dim1.units

    The file is also included as fpd_nt.file.
    ...

    Examples
    --------
    To read everything:

    >>> from merlintools.vendor.fpd import fpd_file as fpdf
    >>> fpd_nt = fpdf.fpd_to_tuple(filename)
    >>> im = fpd_nt.fpd_data.data[0, 0]

    To return only specific datasets:

    >>> fpd_nt = fpdf.fpd_to_tuple(filename,
    ...     group_names=['fpd_data', 'fpd_sum_im', 'fpd_sum_dif'])

    Here, 'fpd_data' is first, so it may be accessed by indexing by:

    >>> fpd_nt[0].data

    Unless `group_names` is specified, it may not be first. To access by name:

    >>> fpd_nt.fpd_data.data

    References
    ----------
    https://docs.python.org/3/library/collections.html#collections.namedtuple

    See Also
    --------
    fpd_nt_cm

    """

    def _check_titles(titles, group_names):
        # helper for parsing group_names against titles
        print("Detected emd groups:", titles)
        # select groups
        if group_names is None:
            group_names = titles
        elif type(group_names) is str:
            group_names = [group_names]
        non_extant_groups = [t for t in group_names if t not in titles]
        if len(non_extant_groups):
            print("The following requested groups do not exists:", non_extant_groups)
        # get indices
        indices = [titles.index(t) for t in group_names if t in titles]
        return indices

    if fpd_check:
        _ = _check_fpd_file(fpg)

    closef, h5f = _get_hdf5_file_from_obj(fpg)
    fpd_group = h5f["fpd_expt"]
    titles = find_hdf5_objname_by_attribute(
        fpd_group, attr_name="emd_group_type", attr_val=1, fpd_check=False
    )
    indices = _check_titles(titles, group_names)
    titles_selection = [titles[i] for i in indices]

    ntis = []
    for tsi in titles_selection:
        # loop over data groups
        emd_dg = fpd_group[tsi]
        dct = dict(emd_dg.items())
        data_name = "data"

        # add attrs
        data_attrs_dct = dict(emd_dg[data_name].attrs.items())
        _ = data_attrs_dct.pop("CLASS", None)
        _ = data_attrs_dct.pop("IMAGE_VERSION", None)
        dct.update(data_attrs_dct)

        # add dataset as array if small
        nd = dct[data_name].ndim
        nbytes = dct[data_name].size
        if nd <= nd_max and nbytes / 1e9 <= gigabytes_max:
            dct.update({data_name: emd_dg[data_name][...]})

        # add axes
        for i in range(1, nd + 1):
            ax_name = "dim%i" % (i)
            ax_gp = emd_dg[ax_name]
            ax_dct = dict(ax_gp.attrs.items())
            ax_dct.update({"data": ax_gp[...]})
            ax_nt = namedtuple(ax_name, ax_dct.keys())(**ax_dct)
            dct.update({ax_name: ax_nt})

        nti = namedtuple(tsi, dct.keys())(**dct)
        ntis.append(nti)
    titles_selection.append("file")
    ntis.append(h5f.file)
    fpd_nt = namedtuple("fpd_nt", titles_selection)
    fpd_nt = fpd_nt(*ntis)

    return fpd_nt


@contextlib.contextmanager
def fpd_nt_cm(*args, **kwargs):
    """
    Simple contextmanager for fpd.fpd_file.fpd_to_tuple,
    enabling use of the function in a `with` statement, and the
    automatic closing of the underlying file.

    Parameters
    ----------
    All parameters are passed unchanged to `fpd_to_tuple`. See that
    function for documentation.

    Returns
    -------
    fpd_nt : namedtuple
        A named tuple of in-memory and hdf5 objects. See `fpd_to_tuple`
        for details.

    Example
    -------
    >>> from merlintools.vendor.fpd import fpd_file as fpdf

    >>> with fpdf.fpd_nt_cm('file.hdf5') as fpd_nt:
    ...     # file is open here
    ...     print('inside with:', fpd_nt.file)
    >>> # file is closed here
    >>> print('outside with:', fpd_nt.file)

    See Also
    --------
    fpd_to_tuple

    """

    fpd_nt = fpd_to_tuple(*args, **kwargs)
    try:
        yield fpd_nt
    finally:
        fpd_nt.file.close()


def make_updated_fpd_file(
    src,
    dst,
    func,
    func_kwargs=None,
    update_sum_im=True,
    update_sum_dif=True,
    MiB_max=512,
    ow=False,
    fpd_check=False,
    progress_bar=True,
    reopen_file=True,
):
    """
    Make a new file with the dataset(s) modified by `func`.

    Parameters
    ----------
    src : str, file, group or dataset
        Filename, or hdf5 file, group or dataset of the source file.
        If not a filename, the file should be open. See Notes.
    dst : str
        Filename of the modified file.
    func : function
        A function that applies out-of-place to each image. The
        function should be of the form:
            f(image, scanYind, scanXind, **func_kwargs).
        Note that the result is coerced to be of the same dtype as the original
        data. Additional arguments may be passed by the `func_kwargs` keyword.
        See notes.
    func_kwargs : None or dict
        A dictionary of keyword arguments passed to `func`.
    update_sum_im : bool
        If True, the `sum_im` image is recalculated.
    update_sum_dif : bool
        If True, the `sum_dif` image is recalculated.
    MiB_max : scalar
        Maximum number of mebibytes (1024**2 bytes) allowed for file access. This is
        a soft limit and controls how many chunks are read for calculating image sums.
        The minimum number of chunks is currently 1, so this limit might be breached
        for very large chunk sizes.
    ow : bool
        If True, `dst` is overwritten if it exists.
    fpd_check : bool
        If True, the hdf5 file is checked for being valid.
    progress_bar : bool
        If True, progress bars are printed.
    reopen_file : bool
        If True, the source file is closed and reopened before copying only if the OS
        is Windows. See Notes.

    Returns
    -------
    new_fn : string
        The name of the output file.

    Notes
    -----
    As an example of the `func` usage, the code below shows how to move each
    image in a dataset by a different amount. The amount is set by the syx array
    (not explicitly set here, but could be from fpd.fpd_processing.center_of_mass
    or similar):

    >>> from fpd.synthetic_data import shift_im
    >>> def shift_func(image, scanYind, scanXind, shift_array):
    >>>     syx = shift_array[:, scanYind, scanXind]
    >>>     new_im = shift_im(image, -syx, fill_value=0).astype(int, copy=False)
    >>>     return new_im


    Some Windows users have found kernel crashes when operating with an already open
    source file. There are no reports of this in Linux. Details are logged at
    https://gitlab.com/fpdpy/fpd/-/issues/38. Closing the file before running this
    function prevents this from occurring. With `reopen_file=True`, the file will
    be closed and reopened automatically within the function. However, as this cannot
    be done if `src` is a filename, it is best to provide a file object as `src`.

    """

    # could operate in place if dst==Nonw

    ### condition inputs
    if func_kwargs is None:
        func_kwargs = {}

    ext = ".hdf5"
    if not dst.lower().endswith(ext):
        dst += ext
    new_fn = os.path.abspath(dst)

    ### check if dst exists
    if not ow and os.path.isfile(dst):
        reply = input(
            "File '%s' already exists. Overwrite? ([y]/n): " % (os.path.abspath(dst))
        )
        if reply.strip().lower() in ["y", "yes", ""]:
            pass
        else:
            print("Aborting.")
            return None

    ### get file
    closef, h5f = _get_hdf5_file_from_obj(src)

    # if on windows, check if file is open
    if platform.system() == "Windows":
        h5f_absfn = os.path.abspath(h5f.filename)
        p = psutil.Process()
        ofs = p.open_files()
        fopened = [ofsi for ofsi in ofs if h5f_absfn in ofsi.path]
        if len(fopened):
            # if open, close file to avoid kernel crash on windows
            # https://gitlab.com/fpdpy/fpd/-/issues/38
            if reopen_file and closef:
                # src was a string and so we cannot (easily) ensure the file was not already open
                m = """Cannot ensure file is closed. This may be an issue on Windows.
If there are problems, provide a file reference as `src`.
See https://gitlab.com/fpdpy/fpd/-/issues/38 for details."""
                for mi in m.split("\n"):
                    _logger.warning(mi)
            if reopen_file and not closef:
                # file was open
                src_fn = h5f.filename
                h5f.close()
                closef, h5f = _get_hdf5_file_from_obj(src_fn)

    if fpd_check:
        _ = _check_fpd_file(src)

    ### make file and copy contents
    # helper for copying attributes
    def cp_attrs(src, trg):
        for k, v in src.attrs.items():
            trg.attrs[k] = v

    # function for visiting objects
    def cp(obj_name):
        obj = h5f[obj_name]

        # make group
        if isinstance(obj, h5py.Group):
            g = new_file.create_group(obj.name)
            cp_attrs(obj, g)

        # copy or make datasets
        if isinstance(obj, h5py.Dataset):
            try:
                m = re.match("^fpd_expt/DM[0-9]+/bin$", obj_name)
                dm_bin = m is not None
                n = re.match("^fpd_expt/TIA[0-9]+/bin$", obj_name)
                tia_bin = n is not None
                if dm_bin:
                    # binary blobs handled here to avoid:
                    # Unable to create dataset (object header message is too large)
                    ds = new_file.create_dataset(name=obj_name, data=np.void(obj[()]))
                elif tia_bin:
                    # binary blobs handled here to avoid:
                    # Unable to create dataset (object header message is too large)
                    ds = new_file.create_dataset(name=obj_name, data=np.void(obj[()]))
                else:
                    # seems chunks are automatically set to 1024 for empty data
                    # with automatic chunking. So update here to same automatic
                    # chunking when making new one.
                    kw = dict()
                    if obj.size == 0:
                        kw = dict(chunks=True)
                    # make dataset and copy unless it is the main data
                    ds = new_file.create_dataset_like(obj_name, obj, **kw)
                    if "fpd_data/data" not in obj_name:
                        ds[()] = obj[()]

                cp_attrs(obj, ds)
            except Exception as e:
                print("ERROR: ", e, obj_name)

    # actually do the copying
    with h5py.File(dst, "w") as new_file:
        cp_attrs(h5f, new_file)
        h5f.visit(cp)

        ### update data
        ds = h5f["fpd_expt/fpd_data/data"]
        ds_new = new_file["fpd_expt/fpd_data/data"]

        # loop over chunks only in non-image axes, correcting data
        data_chunks = ds.chunks[:-2]
        data_shape = ds.shape
        sinds, n = slice_indices(data_shape, data_chunks)

        for si in tqdm(
            sinds, total=n, unit="non-image chunks", disable=(not progress_bar)
        ):
            s = tuple([slice(x[0], x[1]) for x in si])
            nis = np.ascontiguousarray(ds[s])  # non_image_slice

            yyi, xxi = np.mgrid[s]
            yyi = yyi.flatten()
            xxi = xxi.flatten()

            nis_shape = nis.shape
            nis.shape = (np.prod(nis_shape[:-2]),) + nis_shape[-2:]
            for i, im_array in enumerate(nis):
                scanYind, scanXind = yyi[i], xxi[i]
                im_array = func(im_array, scanYind, scanXind, **func_kwargs)
                nis[i] = im_array.astype(ds_new.dtype, copy=False)
            nis.shape = nis_shape

            ds_new[s] = nis

        ### update sum images from main dataset
        from .fpd_processing import sum_dif, sum_im

        if update_sum_im:
            read_chunk_y, read_chunk_x = determine_read_chunks(
                ds_new, MiB_max=MiB_max, det_im_processing=True
            )
            # print(read_chunk_y, read_chunk_x)
            new_file["fpd_expt/fpd_sum_im/data"][:] = sum_im(
                ds_new, read_chunk_y, read_chunk_x, progress_bar=progress_bar
            )

        if update_sum_dif:
            read_chunk_y, read_chunk_x = determine_read_chunks(
                ds_new, MiB_max=MiB_max, det_im_processing=False
            )
            # print(read_chunk_y, read_chunk_x)
            new_file["fpd_expt/fpd_sum_dif/data"][:] = sum_dif(
                ds_new, read_chunk_y, read_chunk_x, progress_bar=progress_bar
            )

    ### close src file
    if closef:
        h5f.close()

    return new_fn


def update_calibration(
    filename, scan=[None, None], detector=[None, None], colour=[None, None]
):
    """
    Update calibration of an existing fpd file.

    DM / TIA files are not updated.

    Parameters
    ----------
    filename : str
        Filename of file to be updated.
    scan, detector, colour : iterable
        Length-2 iterable of [axis_cal, axis_units] with 'axis_cal' a scalar and
        'axis_units' a string. If a parameter is None then that parameter is not
        updated.
    """

    # check axes being updated
    sb = (np.array(scan) == None).all() == False
    db = (np.array(detector) == None).all() == False
    cb = (np.array(colour) == None).all() == False
    if sb == False and db == False and cb == False:
        _logger.warning("All parameters are None, so nothing will be done!")
        return None

    with h5py.File(filename, "r+") as h5f:
        grp = h5f["fpd_expt"]
        data_shape = grp["fpd_data/data"].shape

        c_exists = len(data_shape) == 5
        if cb and c_exists == False:
            m = "No colour axis exists! Set 'colour=[None, None]'."
            _logger.warning(m)
            raise Exception(m)

        ## data axes are not tagged, so need to run over all known datasets
        # first 2 axes are scanY, scanX. If colour, it is third.
        scan_c_grps = [
            "DAC",
            "Exposure",
            "Threshold",
            "Unixtime",
            "binary_hdr",
            "fpd_sum_im",
            "fpd_data",
        ]
        # last 2 axes are detY, detX. If colour, it is third last except for 'fpd_mask'.
        det_grps = ["fpd_sum_dif", "fpd_mask", "fpd_data"]
        # simple approach here means 'fpd_data' colour will be processed twice

        def update_dim(dim, cal_unit):
            cal, unit = cal_unit
            # we can always update the units, so do this first
            if unit is not None:
                dim.attrs["units"] = unit
            # axes dtype can't be changed, so need to replace with float if int
            if cal is not None:
                new_axis = np.arange(dim.shape[0]) * cal
                if issubclass(dim.dtype.type, numbers.Integral):
                    attrs = dict(dim.attrs.items())
                    name = dim.name
                    h5f.__delitem__(name)
                    new_dim = h5f.create_dataset(name=name, data=new_axis, dtype=float)
                    for ki, vi in attrs.items():
                        new_dim.attrs[ki] = vi
                else:
                    dim[:] = new_axis

        for k, v in grp.items():
            if k in det_grps:
                n = len(v["data"].shape)
                for dim in [v["dim%d" % (n - 2 + 1)], v["dim%d" % (n - 1 + 1)]]:
                    update_dim(dim, detector)
                if c_exists and k is not "fpd_mask":
                    dim = v["dim%d" % (n - 3 + 1)]
                    update_dim(dim, colour)
            if k in scan_c_grps:
                for dim in [v["dim%d" % (1)], v["dim%d" % (2)]]:
                    update_dim(dim, scan)
                if c_exists:
                    dim = v["dim%d" % (3)]
                    update_dim(dim, colour)

        # check for extra / missing
        all_grps = set(scan_c_grps + det_grps)

        emd_objs = find_hdf5_objname_by_attribute(h5f, attr_name="emd_group_type")
        emd_objs_grps = set([t.rsplit("/", 1)[1] for t in emd_objs])

        dm_objs = find_hdf5_objname_by_attribute(h5f, attr_name="dm_group_type")
        dm_objs_grps = set([t.rsplit("/", 1)[1] for t in dm_objs])

        tia_objs = find_hdf5_objname_by_attribute(h5f, attr_name="tia_group_type")
        tia_objs_grps = set([t.rsplit("/", 1)[1] for t in tia_objs])

        # rm dm and tia objects
        objs_grps = emd_objs_grps - dm_objs_grps - tia_objs_grps

        not_present_in_file = set(all_grps) - set(objs_grps)
        unknown_in_file = set(objs_grps) - set(all_grps)
        # optional
        if "fpd_mask" in not_present_in_file:
            not_present_in_file.remove("fpd_mask")

        if len(not_present_in_file):
            _logger.warning(
                "Expected emd objects not present: %s" % (str(not_present_in_file))
            )
        if len(unknown_in_file):
            _logger.warning(
                "Extra emd objects not processed: %s" % (str(unknown_in_file))
            )


class DataBrowser:
    def __init__(
        self,
        fpgn,
        nav_im=None,
        cmap=None,
        norm="lin",
        colour_index=None,
        nav_im_dict=None,
        fpd_check=True,
        progress_bar=True,
    ):
        """
        Navigate fpd data set.

        Parameters
        ----------
        fpgn : hdf5 str, file, group, dataset, ndarray, or dask array.
             hdf5 filename, file, group or dataset, or numpy array,
             `MerlinBinary` object, or dask array.
        nav_im : 2-D array or None
            Navigation image. If None, this is taken as the sum image.
            For numpy arrays, it is calculated directly.
        cmap : str or None
            Matplotlib colourmap name used for diffraction image.
            If None, `viridis` is used if available, else `gray`.
        norm : str:
            One of ['lin', 'log'] used to set the diffraction image norm.
        colour_index : int or None
            Colour index used for plotting. If None, the first index is
            used.
        nav_im_dict : None or dictionary
            Keyword arguments passed to the navigation imshow call.
        fpd_check : bool
            If True, the file format version is checked.
        progress_bar : bool
            If True, progress bars are printed.

        TODO
        ----
        log / linear norms
        nav_im list input and switch between images?
        display with axis units rather than pixels?

        """

        import fpd

        isnumpy = isinstance(fpgn, np.ndarray)
        ismerlin = isinstance(fpgn, fpd.fpd_file.MerlinBinary)
        isdask = "dask.array.core.Array" in str(type(fpgn))
        ish5data = isinstance(fpgn, h5py._hl.dataset.Dataset)
        isfpd = _check_fpd_file(fpgn, silent=True)[0]

        treat_as_array = isnumpy or ismerlin or isdask or (ish5data and not isfpd)

        if isfpd:
            if fpd_check:
                # this will raise error if file is both not understood
                isfpd, isknown, vs = _check_fpd_file(fpgn)
            try:
                # try to use FPD hdf5 format
                self.closef, self.h5f = _get_hdf5_file_from_obj(fpgn)
                self.h5f_ds = self.h5f["fpd_expt/fpd_data/data"]
            except:
                self.closef = False
                self.h5f_ds = fpgn
        elif treat_as_array:
            self.closef = False
            self.h5f_ds = fpgn

        self.nav_im_dict = nav_im_dict

        # get data shape info
        self.scanY, self.scanX = self.h5f_ds.shape[:2]
        self.detY, self.detX = self.h5f_ds.shape[-2:]

        # determine colour info
        ds_shape_len = len(self.h5f_ds.shape)
        if ds_shape_len == 4:
            # no colour data
            self.ncolours = 0
        elif ds_shape_len == 5:
            self.ncolours = self.h5f_ds.shape[2]

        self.colour_index = None
        if colour_index is None and self.ncolours:
            self.colour_index = 0

        # navigation image
        if nav_im is None:
            if not treat_as_array:
                # get data from fpd file
                if self.colour_index is not None:
                    nav_im = self.h5f["fpd_expt/fpd_sum_im/data"][
                        ..., self.colour_index
                    ]
                else:
                    nav_im = self.h5f["fpd_expt/fpd_sum_im/data"][...]
            else:
                # access array
                if self.ncolours == 0:
                    # nav_im = self.h5f_ds.sum((-2, -1))
                    nav_im = fpd.fpd_processing.sum_im(
                        self.h5f_ds, 16, 16, progress_bar=progress_bar
                    )
                else:
                    # nav_im = self.h5f_ds[:, :, self.colour_index].sum((-2, -1))
                    nav_im = fpd.fpd_processing.sum_im(
                        self.h5f_ds, 16, 16, progress_bar=progress_bar
                    )[..., self.colour_index]
        self.nav_im = nav_im

        self.scanYind = 0
        self.scanXind = 0
        self.scanYind_old = self.scanYind
        self.scanXind_old = self.scanXind
        if self.colour_index is not None:
            self.plot_data = self.h5f_ds[
                self.scanYind, self.scanXind, self.colour_index, :, :
            ]
        else:
            self.plot_data = self.h5f_ds[self.scanYind, self.scanXind, :, :]
        self.plot_data = np.ascontiguousarray(self.plot_data)

        self.rwh = max(self.scanY, self.scanX) // 64
        if self.rwh == 0:
            self.rwh = 2
        self.rect = None
        self.press = None
        self.background = None
        self.plot_nav_im()

        # cmap
        if cmap is None:
            try:
                self.cmap = mpl.cm.get_cmap("viridis")
            except ValueError:
                self.cmap = mpl.cm.get_cmap("gray")
        else:
            self.cmap = mpl.cm.get_cmap(cmap)
        self.cmap = copy.copy(self.cmap)
        self.cmap.set_bad(self.cmap(0))

        # handle norms
        norms = ["lin", "log"]
        if norm not in norms:
            raise NotImplementedError(
                "'norm' (%s) must be one of: " % (norm) + str(norms)
            )
        if norm == "lin":
            self.norm = None
        elif norm == "log":
            self.norm = mpl.colors.LogNorm()

        self.plot_dif()
        self.connect()

    def plot_nav_im(self):
        kwd = dict(adjustable="box-forced", aspect="equal")
        if _mpl_non_adjust:
            _ = kwd.pop("adjustable")

        self.f_nav, ax = plt.subplots(subplot_kw=kwd)
        self.f_nav.canvas.set_window_title("nav")

        d = {"cmap": "gray"}
        if self.nav_im_dict is not None:
            d.update(self.nav_im_dict)
        im = ax.imshow(self.nav_im, interpolation="nearest", **d)
        if self.nav_im.ndim != 3:
            plt.colorbar(mappable=im)

        rect_c = "r"
        if self.nav_im.ndim == 3 or (self.nav_im.ndim == 2 and im.cmap.name != "gray"):
            # rgb
            rect_c = "w"  #'k'
        self.rect = mpl.patches.Rectangle(
            (self.scanYind - self.rwh / 2, self.scanXind - self.rwh / 2),
            self.rwh,
            self.rwh,
            ec=rect_c,
            fc="none",
            lw=2,
            picker=4,
        )
        ax.add_patch(self.rect)
        plt.tight_layout()
        plt.draw()

    def plot_dif(self):
        kwd = dict(adjustable="box-forced", aspect="equal")
        if _mpl_non_adjust:
            _ = kwd.pop("adjustable")

        self.f_dif, ax = plt.subplots(subplot_kw=kwd)
        self.f_dif.canvas.set_window_title("dif")

        self.im = ax.matshow(
            self.plot_data, interpolation="nearest", cmap=self.cmap, norm=self.norm
        )
        plt.sca(ax)
        self.cbar = plt.colorbar(self.im)
        ax.format_coord = self.format_coord
        self.update_dif_plot()

        plt.tight_layout()
        plt.draw()

    def connect(self):
        "connect to all the events we need"
        self.cidpress = self.rect.figure.canvas.mpl_connect(
            "button_press_event", self.on_press
        )
        self.cidrelease = self.rect.figure.canvas.mpl_connect(
            "button_release_event", self.on_release
        )
        self.cidmotion = self.rect.figure.canvas.mpl_connect(
            "motion_notify_event", self.on_motion
        )

        self.cid_f_nav = self.f_nav.canvas.mpl_connect("close_event", self.handle_close)
        self.cid_f_dif = self.f_dif.canvas.mpl_connect("close_event", self.handle_close)

    def handle_close(self, e):
        if self.closef == True:
            self.h5f.file.close()
        self.disconnect()
        # close other fig
        if e.canvas.get_window_title() == "nav":
            plt.close(self.f_dif)
        else:
            plt.close(self.f_nav)

    def on_press(self, event):
        if event.inaxes != self.rect.axes:
            return

        contains, attrd = self.rect.contains(event)
        if contains:
            # print('event contains', self.rect.xy)
            x0, y0 = self.rect.xy  # xy is lower left

            # draw everything but the selected rectangle and store the pixel buffer
            canvas = self.rect.figure.canvas
            axes = self.rect.axes
            self.rect.set_animated(True)
            canvas.draw()
            self.background = canvas.copy_from_bbox(self.rect.axes.bbox)

            axes.draw_artist(self.rect)  # now redraw just the rectangle
            canvas.blit(axes.bbox)  # and blit just the redrawn area
        else:
            # in axis but not rectangle
            # print event.xdata, event.ydata
            x0, y0 = (None,) * 2

        self.press = x0, y0, event.xdata, event.ydata
        self.yind_temp = self.scanYind
        self.xind_temp = self.scanXind

    def on_motion(self, event):
        if self.press is None:
            return
        if event.inaxes != self.rect.axes:
            return
        if self.background is None:
            return

        x0, y0, xpress, ypress = self.press
        dx = int(event.xdata - xpress)
        dy = int(event.ydata - ypress)
        if abs(dy) > 0 or abs(dx) > 0:
            # print('x0=%f, xpress=%f, event.xdata=%f, dx=%f, x0+dx=%f'%(x0, xpress, event.xdata, dx, x0+dx))
            self.rect.set_x(x0 + dx)
            self.rect.set_y(y0 + dy)
            self.scanYind = self.yind_temp + dy
            self.scanXind = self.xind_temp + dx
            # print(dy, dx)

            canvas = self.rect.figure.canvas
            axes = self.rect.axes
            canvas.restore_region(self.background)  # restore the background region
            axes.draw_artist(self.rect)  # redraw just the current rectangle
            canvas.blit(axes.bbox)  # blit just the redrawn area

            # self.rect.figure.canvas.draw()
            self.update_dif_plot()

    def on_release(self, event):
        if event.inaxes != self.rect.axes:
            return

        x, y = self.press[2:]
        if np.round(event.xdata - x) == 0 and np.round(event.ydata - y) == 0:
            # mouse didn't move
            x, y = np.round(x), np.round(y)
            self.rect.set_x(x - self.rwh / 2)
            self.rect.set_y(y - self.rwh / 2)
            self.scanYind = int(y)
            self.scanXind = int(x)

            # self.rect.figure.canvas.draw()
            self.update_dif_plot()
        elif self.background is not None:
            canvas = self.rect.figure.canvas
            axes = self.rect.axes
            canvas.restore_region(self.background)  # restore the background region
            axes.draw_artist(self.rect)  # redraw just the current rectangle
            canvas.blit(axes.bbox)  # blit just the redrawn area

        "on release we reset the press data"
        self.press = None
        self.yind_temp = None
        self.yind_temp = None

        # turn off the rect animation property and reset the background
        self.rect.set_animated(False)
        self.background = None

        # redraw the full figure
        self.rect.figure.canvas.draw()

    def format_coord(self, x, y):
        col = np.ceil(x - 0.5).astype(int)
        row = np.ceil(y - 0.5).astype(int)
        if col >= 0 and col < self.detX and row >= 0 and row < self.detY:
            z = self.plot_data[row, col]
            return "x=%d, y=%d, z=%d" % (x, y, z)
        else:
            return "x=%d, y=%d" % (x, y)

    def update_dif_plot(self):
        if self.colour_index is not None:
            self.plot_data = self.h5f_ds[
                self.scanYind, self.scanXind, self.colour_index, :, :
            ]
        else:
            self.plot_data = self.h5f_ds[self.scanYind, self.scanXind, :, :]
        self.plot_data = np.ascontiguousarray(self.plot_data)
        self.im.set_data(self.plot_data)
        self.im.autoscale()
        self.im.changed()
        self.im.axes.set_xlabel("scanX %s" % self.scanXind)
        self.im.axes.set_ylabel("scanY %s" % self.scanYind)
        self.im.axes.figure.canvas.draw()

    def disconnect(self):
        "disconnect all the stored connection ids"
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)

        self.f_nav.canvas.mpl_disconnect(self.cid_f_nav)
        self.f_dif.canvas.mpl_disconnect(self.cid_f_dif)
