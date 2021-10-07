# -*- coding: utf-8 -*-
#
# This file is part of MerlinTools

"""
io module for MerlinTools package.

@author: Andrew Herzing
"""

import re
import numpy as np
import os
import logging
import h5py
import glob
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def sort_mibs(filename_list):
    """
    Sort a list of MIB file names by frame number.

    Args
    ----------
    filename_list : List
        A list of MIB files to sort

    Returns
    ----------
    sorted_list : List
        A list of MIB files sorted by frame number

    """
    key_list = []

    for filename in filename_list:
        root_name = filename.split(".")[-2]
        key_list.append(int(re.search(r'\d+', root_name[::-1]).group()[::-1]))

    sorted_list = [filename_list for (key_list, filename_list)
                   in sorted(zip(key_list, filename_list))]
    return sorted_list


def get_exposure_times(mibfiles, n=None):
    """
    Extract the exposure time for each frame from MIB file header(s).

    Args
    ----------
    mibfiles : List
        A list of MIB files to inspect

    Returns
    ----------
    expsoures : NumPy array
        The exposure time, in seconds, for each frame in the dataset

    """
    if type(mibfiles) is list and len(mibfiles) > 1:
        # If n is not provided, get number of frames from the filesize
        # and read all exposures
        if n is None:
            n = len(mibfiles)
        logger.info("Reading %s exposure times" % n)
        exposures = np.zeros(n)
        for i in range(0, n):
            with open(mibfiles[i], 'r') as h:
                line = h.readline(200)
            exposures[i] = np.float32(line.split(',')[10]) * 1000
    else:
        if type(mibfiles) is list:
            mibfiles = mibfiles[0]
        with open(mibfiles, 'rb') as h:
            hdr_temp = np.fromfile(h, 'int8', 384)
        hdr_temp = ''.join([chr(item) for item in hdr_temp]).split(',')
        header_length = int(hdr_temp[2])
        data_type = hdr_temp[6]
        if data_type == 'U32':
            dtype = 'uint32'
            data_length = 4
        elif data_type == 'U16':
            dtype = 'uint16'
            data_length = 2
        else:
            dtype = 'uint8'
            data_length = 1
        # If n is not provided, get number of frames from the filesize
        # and read all exposures
        if n is None:
            logger.info("Reading all exposure times")
            n = int(os.path.getsize(mibfiles) / (data_length * (256**2) + header_length))
        else:
            logger.info("Reading %s exposure times" % n)
        exposures = np.zeros(n)
        with open(mibfiles, 'rb') as h:
            for i in range(0, n):
                hdr_temp = np.fromfile(h, 'int8', header_length)
                hdr_temp = ''.join([chr(item) for item in hdr_temp]).split(',')
                exposures[i] = hdr_temp[10]
                _ = np.fromfile(h, dtype, 256**2)
    return exposures


def parse_hdr(hdrfile):
    """
    Extract all information from a HDR file.

    Args
    ----------
    hdrfile : str
        HDR file to parse

    Returns
    ----------
    header : dict
        Extracted header information

    """
    header = {}
    with open(hdrfile, 'r') as h:
        _ = h.readlines(1)[0].rstrip()
        header['TimeStamp'] = h.readlines(1)[0].rstrip().split(':\t')[1]
        header['ChipID'] = h.readlines(1)[0].rstrip().split(':\t')[1]
        header['ChipType'] = h.readlines(1)[0].rstrip().split(':\t')[1]
        header['ChipConfig'] = h.readlines(1)[0].rstrip().split(':\t')[1][3:]
        header['ChipMode'] = h.readlines(1)[0].rstrip().split(':\t')[1]
        header['CounterDepth'] = h.readlines(1)[0].rstrip().split(':\t')[1]
        header['Gain'] = h.readlines(1)[0].rstrip().split(':\t')[1]
        header['ActiveCounters'] = h.readlines(1)[0].rstrip().split(':\t')[1]
        header['Thresholds'] = h.readlines(1)[0].rstrip().split(':\t')[1]
        header['DACS'] = h.readlines(1)[0].rstrip().split(':\t')[1]
        header['BPC_File'] = h.readlines(1)[0].rstrip().split(':\t')[1]
        header['DAC_File'] = h.readlines(1)[0].rstrip().split(':\t')[1]
        header['GapFill'] = h.readlines(1)[0].rstrip().split(':\t')[1]
        header['FlatField_File'] = h.readlines(1)[0].rstrip().split(':\t')[1]
        header['DeadTime_File'] = h.readlines(1)[0].rstrip().split(':\t')[1]
        header['AcqType'] = h.readlines(1)[0].rstrip().split(':\t')[1]
        header['TotalFrames'] = h.readlines(1)[0].rstrip().split(':\t')[1]
        header['FramesPerTrigger'] = h.readlines(1)[0].rstrip().split(':\t')[1]
        header['TriggerStart'] = h.readlines(1)[0].rstrip().split(':\t')[1]
        header['TriggerStop'] = h.readlines(1)[0].rstrip().split(':\t')[1]
        header['Bias'] = h.readlines(1)[0].rstrip().split(':\t')[1]
        header['Polarity'] = h.readlines(1)[0].rstrip().split(':\t')[1]
        header['Temperature'] = h.readlines(1)[0].rstrip().split(':\t')[1]
        header['Humidity'] = h.readlines(1)[0].rstrip().split(':\t')[1]
        header['Clock'] = h.readlines(1)[0].rstrip().split(':\t')[1]
        header['Readout'] = h.readlines(1)[0].rstrip().split(':\t')[1]
        header['SoftwareVersion'] = h.readlines(1)[0].rstrip().split(':\t')[1]
        return header


def parse_mib_header(mibfile):
    """
    Extract header information from a MIB file.

    Args
    ----------
    mibfile : str
        MIB file to parse

    Returns
    ----------
    mib_hdr : dict
        Extracted header information

    """
    mib_hdr = {}
    with open(mibfile, 'r') as h:
        hdr_temp = np.fromfile(h, 'int8', 384)
        hdr_temp = ''.join([chr(item) for item in hdr_temp]).split(',')

    mib_hdr['HeaderID'] = hdr_temp[0]
    mib_hdr['AcquisitionSequenceNumber'] = hdr_temp[1]
    mib_hdr['DataOffset'] = np.int32(hdr_temp[2])
    mib_hdr['NumChips'] = np.int32(hdr_temp[3])
    mib_hdr['PixDimX'] = np.int32(hdr_temp[4])
    mib_hdr['PixDimY'] = np.int32(hdr_temp[5])
    mib_hdr['PixDepth'] = hdr_temp[6]
    mib_hdr['SensorLayout'] = hdr_temp[7][3:]
    mib_hdr['ChipSelect'] = np.int32(hdr_temp[8])
    mib_hdr['TimeStamp'] = hdr_temp[9]
    mib_hdr['ShutterTime'] = np.float32(hdr_temp[10])
    mib_hdr['Counter'] = np.int32(hdr_temp[11])
    mib_hdr['ColourMode'] = np.int32(hdr_temp[12])
    mib_hdr['GainMode'] = np.int32(hdr_temp[13])
    mib_hdr['Thresholds'] = np.zeros(8, np.float32)
    for i in range(0, 8):
        mib_hdr['Thresholds'][i] = np.float32(hdr_temp[14 + i])
    mib_hdr['DACs'] = {}
    mib_hdr['DACs']['Format'] = hdr_temp[22]
    mib_hdr['DACs']['Thresh0'] = np.uint16(hdr_temp[23])
    mib_hdr['DACs']['Thresh1'] = np.uint16(hdr_temp[24])
    mib_hdr['DACs']['Thresh2'] = np.uint16(hdr_temp[25])
    mib_hdr['DACs']['Thresh3'] = np.uint16(hdr_temp[26])
    mib_hdr['DACs']['Thresh4'] = np.uint16(hdr_temp[27])
    mib_hdr['DACs']['Thresh5'] = np.uint16(hdr_temp[28])
    mib_hdr['DACs']['Thresh6'] = np.uint16(hdr_temp[29])
    mib_hdr['DACs']['Thresh7'] = np.uint16(hdr_temp[30])
    mib_hdr['DACs']['Preamp'] = np.uint8(hdr_temp[31])
    mib_hdr['DACs']['Ikrum'] = np.uint8(hdr_temp[32])
    mib_hdr['DACs']['Shaper'] = np.uint8(hdr_temp[33])
    mib_hdr['DACs']['Disc'] = np.uint8(hdr_temp[34])
    mib_hdr['DACs']['DiscLS'] = np.uint8(hdr_temp[35])
    mib_hdr['DACs']['ShaperTest'] = np.uint8(hdr_temp[36])
    mib_hdr['DACs']['DACDiscL'] = np.uint8(hdr_temp[37])
    mib_hdr['DACs']['DACTest'] = np.uint8(hdr_temp[38])
    mib_hdr['DACs']['DACDISCH'] = np.uint8(hdr_temp[39])
    mib_hdr['DACs']['Delay'] = np.uint8(hdr_temp[40])
    mib_hdr['DACs']['TPBuffIn'] = np.uint8(hdr_temp[41])
    mib_hdr['DACs']['TPBuffOut'] = np.uint8(hdr_temp[42])
    mib_hdr['DACs']['RPZ'] = np.uint8(hdr_temp[43])
    mib_hdr['DACs']['GND'] = np.uint8(hdr_temp[44])
    mib_hdr['DACs']['TPRef'] = np.uint8(hdr_temp[45])
    mib_hdr['DACs']['FBK'] = np.uint8(hdr_temp[46])
    mib_hdr['DACs']['Cas'] = np.uint8(hdr_temp[47])
    mib_hdr['DACs']['TPRefA'] = np.uint16(hdr_temp[48])
    mib_hdr['DACs']['TPRefB'] = np.uint16(hdr_temp[49])
    mib_hdr['ExtID'] = hdr_temp[50]
    mib_hdr['ExtTimeStamp'] = hdr_temp[51]
    mib_hdr['ExtID'] = np.float64(hdr_temp[52][:-2])
    mib_hdr['ExtCounterDepth'] = np.uint8(hdr_temp[53])
    return mib_hdr


def get_scan_shape(mibfiles):
    """
    Determine scan shape from file size and exposure times.

    rgs
    ----------
    mibfiles : list
        List of MIB files

    Returns
    ----------
    scanXalu, scanYalu : list
        List providing the scan axes parameters. Format: [axis, name, units]
    skip_frames : int
        Number of extra frames at the beginning of the scan
    total_frames : int
        Total number of frames in scan
    """
    mib_hdr = parse_mib_header(mibfiles[0])
    n_detector_pix = mib_hdr['PixDimX'] * mib_hdr['PixDimY']
    header_length = mib_hdr['DataOffset']
    if mib_hdr['PixDepth'] == 'U08':
        data_length = 1
    elif mib_hdr['PixDepth'] == 'U16':
        data_length = 2
    elif mib_hdr['PixDepth'] == 'U32':
        data_length = 4
    if len(mibfiles) == 1:
        total_frames = int(os.path.getsize(mibfiles[0]) / (data_length * n_detector_pix + header_length))
    else:
        total_frames = len(mibfiles)
    logger.info("Total frames: %s" % total_frames)

    exp = 1000 * get_exposure_times(mibfiles)
    exp_round = np.round(exp, 0)
    vals, counts = np.unique(exp_round, return_counts=True)
    exposure_time, flyback_time = vals[counts.argsort()[::-1]][0:2]
    flyback_pixels = np.where(exp_round == flyback_time)[0]
    scan_width = np.diff(flyback_pixels)[0]
    scan_height = len(flyback_pixels) + 1
    skip_frames = flyback_pixels[0] - scan_width
    extra_frames = total_frames - skip_frames - scan_width * scan_height

    logger.info("Exposure time (ms): %s" % exposure_time)
    logger.info("Flyback time (ms): %s" % flyback_time)
    logger.info("Extra frames at beginning: %s" % skip_frames)
    logger.info("Scan width based on flyback: %s pixels" % scan_width)
    logger.info("Scan height based on flyback: %s pixels" % scan_height)

    if extra_frames >= 0:
        logger.info("Extra frames at end: %s" % extra_frames)
    else:
        logger.warning("Missing %s frames at end of dataset!"
                       % (-extra_frames))

    scanXalu = [np.arange(0, scan_width), 'x', 'pixels']
    scanYalu = [np.arange(0, scan_height), 'y', 'pixels']
    return scanXalu, scanYalu, skip_frames, total_frames


def get_merlin_parameters(data):
    """
    Get Merlin parameters for 4D-STEM data from FPD HDF5 file.

    Args
    ----------
    data : h5py file or str
        Either the FPD HDF5 filename or FPD file object

    Returns
    ----------
    params : dict
        Dictionary containing the Merlin frame time, scan shape,
        and detector shape

    """
    if isinstance(data, h5py.File):
        scanY, scanX, detY, detX = data["fpd_expt/fpd_data/data"].shape
        exposures = np.round(data["fpd_expt/Exposure/data"][...] / 1e6, 1)
        frame_times, counts = np.unique(exposures, return_counts=True)
        frame_time = frame_times[counts.argmax()]
        thresholds = data["fpd_expt/Threshold/data"][...][:, :, 0]
        threshold = np.unique(thresholds[~np.isnan(thresholds)])[0]

    elif isinstance(data, str):
        with h5py.File(data, 'r') as h5:
            scanY, scanX, detY, detX = h5["fpd_expt/fpd_data/data"].shape
            exposures = np.round(h5["fpd_expt/Exposure/data"][...] / 1e6, 1)
            frame_times, counts = np.unique(exposures, return_counts=True)
            frame_time = frame_times[counts.argmax()]
            thresholds = h5["fpd_expt/Threshold/data"][...][:, :, 0]
            threshold = np.unique(thresholds[~np.isnan(thresholds)])[0]

    params = {'Frame time': frame_time,
              'Scan shape': [scanY, scanX],
              'Detector shape': [detY, detX],
              'Threshold': threshold}
    return params


def get_microscope_parameters(data, display=False):
    """
    Get microscope parameters for 4D-STEM data.

    Parses the simultaneously acquired Digital Micrograph (.dm3/.dm4) file
    to determine camera length, high tension, and magnification.

    Args
    ----------
    data : h5py file, str, or FPD named tuple
        Variable containing the acquisition metadata

    Returns
    ----------
    params : dict
        Dictionary containing the microscope voltage, camera length,
        and magnification

    """
    if isinstance(data, h5py.File):
        if "DM0" in data["/fpd_expt/"].keys():
            cl = float(data["fpd_expt/DM0/tags/ImageList/TagGroup0/ImageTags/"
                            "Microscope Info/STEM Camera Length"][...])
            ht = float(data["fpd_expt/DM0/tags/ImageList/TagGroup0/ImageTags/"
                            "Microscope Info/Voltage"][...]) / 1000
            mag = float(data["fpd_expt/DM0/tags/ImageList/TagGroup0/ImageTags/"
                             "Microscope Info/Indicated Magnification"][...])
        else:
            cl = "Unknown"
            ht = "Unknown"
            mag = "Unknown"
    elif isinstance(data, str):
        with h5py.File(data, 'r') as h5:
            if "DM0" in h5["/fpd_expt/"].keys():
                cl = float(h5["fpd_expt/DM0/tags/ImageList/TagGroup0/ImageTags/"
                              "Microscope Info/STEM Camera Length"][...])
                ht = float(h5["fpd_expt/DM0/tags/ImageList/TagGroup0/ImageTags/"
                              "Microscope Info/Voltage"][...]) / 1000
                mag = float(h5["fpd_expt/DM0/tags/ImageList/TagGroup0/ImageTags/"
                               "Microscope Info/Indicated Magnification"][...])
            else:
                cl = "Unknown"
                ht = "Unknown"
                mag = "Unknown"
    else:
        if "DM0" in data._fields():
            cl = float(data.DM0.tags["ImageList/TagGroup0/ImageTags/"
                                     "Microscope Info/STEM Camera Length"][...])
            ht = float(data.DM0.tags["ImageList/TagGroup0/ImageTags/"
                                     "Microscope Info/Voltage"][...]) / 1000
            mag = float(data.DM0.tags["fpd_expt/DM0/tags/ImageList/TagGroup0/ImageTags/"
                                      "Microscope Info/Indicated Magnification"][...])
        else:
            cl = "Unknown"
            ht = "Unknown"
            mag = "Unknown"
    if display:
        print("Microscope voltage: %.1f kV" % ht)
        print("STEM Camera Length: %.1f mm" % cl)
        print("Magnification: %.1f X" % mag)

    params = {'CL': cl, 'HT': ht, 'Magnification': mag}
    return params


def get_spatial_axes_dict(nt):
    """
    Create a Hyperspy axes manager dictionary from an FPD data object.

    Args
    ----------
    nt : tuple
        Tuple containing FPD data created using fpd.fpd_file.fpd_to_tuple

    Returns
    ----------
    axes_dict : dict
        Hyperspy ready dictionary containing axes info

    """
    unitsX = nt.fpd_data.dim2.units
    originX = nt.fpd_data.dim2.data[0]
    scaleX = nt.fpd_data.dim2.data[1] - nt.fpd_data.dim2.data[0]
    sizeX = nt.fpd_data.dim2.data.shape[0]

    if unitsX in ['um', 'µm']:
        unitsX = 'nm'
        scaleX = scaleX * 1000
        originX = originX * 1000

    unitsY = nt.fpd_data.dim1.units
    originY = nt.fpd_data.dim1.data[0]
    scaleY = nt.fpd_data.dim1.data[1] - nt.fpd_data.dim1.data[0]
    sizeY = nt.fpd_data.dim1.data.shape[0]

    if unitsY in ['um', 'µm']:
        unitsY = 'nm'
        scaleY = scaleY * 1000
        originY = originY * 1000

    axes_dict = {'axis-0': {'name': 'y', 'offset': originY, 'units': unitsY,
                            'scale': scaleY, 'size': sizeY},
                 'axis-1': {'name': 'x', 'offset': originX, 'units': unitsX,
                            'scale': scaleX, 'size': sizeX}}
    return axes_dict


def get_experimental_parameters(rootpath="./"):
    """
    Create a Pandas DataFrame containing experimental parameters from HDF5 files.

    Args
    ----------
    rootpath : str
        Path to data.  All sub-folders will also be inspected

    Returns
    ----------
    df : DataFrame
        Pandas DataFrame containing the extracted experimental parameters

    """
    h5files = glob.glob(rootpath + "**/*.hdf5", recursive=True)
    h5files = [x for x in h5files if not x.endswith('Aligned.hdf5') or x.endswith('aligned.hdf5')]

    datapaths = [None] * len(h5files)
    h5filenames = [None] * len(h5files)
    hts = [None] * len(h5files)
    cls = [None] * len(h5files)
    mags = [None] * len(h5files)
    scan_shapes = [None] * len(h5files)
    det_shapes = [None] * len(h5files)
    thresholds = [None] * len(h5files)
    frame_times = [None] * len(h5files)

    for i in range(0, len(h5files)):
        datapaths[i], h5filenames[i] = os.path.split(h5files[i])
        if fpd_check(h5files[i]):
            microscope_params = get_microscope_parameters(h5files[i])
            hts[i] = microscope_params['HT']
            cls[i] = microscope_params['CL']
            mags[i] = microscope_params['Magnification']

            merlin_params = get_merlin_parameters(h5files[i])
            scan_shapes[i] = ("%s x %s" % (merlin_params['Scan shape'][0], merlin_params['Scan shape'][1]))
            det_shapes[i] = ("%s x %s" % (merlin_params['Detector shape'][0], merlin_params['Detector shape'][1]))
            frame_times[i] = merlin_params['Frame time']
            thresholds[i] = merlin_params['Threshold']

    df = pd.DataFrame()
    df['Data Path'] = datapaths
    df['H5 File'] = h5filenames
    df['Beam Energy'] = hts
    df['Camera Length'] = cls
    df['Mag'] = mags
    df['Scan Shape'] = scan_shapes
    df['Detector Shape'] = det_shapes
    df['Frame time'] = frame_times
    df['Threshold'] = thresholds

    return df


def fpd_check(data):
    """
    Check if a file is an FPD-created HDF5.

    Args
    ----------
    data : str
        Filename to check

    Returns
    ----------
    bool
        If True, file is in FPD format.  If False, the format is unknown.

    """
    if isinstance(data, str):
        with h5py.File(data, 'r') as h5:
            if "/fpd_expt/" in h5.keys():
                return True
            else:
                return False
    elif isinstance(data, h5py.File):
        if "/fpd_expt/" in data.keys():
            return True
        else:
            return False
