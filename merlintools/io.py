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
from merlintools.utils import get_calibration
from merlintools.processing import radial_profile, shift_align
import fpd.fpd_file as fpdf
import fpd.fpd_processing as fpdp
from hyperspy.io import load
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def convert_to_zspy(datapath, ow=False):
    """
    Convert Merlin 4DSTEM data to HyperSpy Zarr format.

    Data is assumed to consist of a single MIB/HDR pair.  If present, an EMI/SER file pair will be used to determine the scan shape.

    Args
    ----------
    datapath : str
        Location of data to conver
    ow : bool
        If True, overwrite data already converted in the given path


    """
    mibfile = glob.glob(datapath + "/*.mib")[0]
    fileparts = mibfile.split("/")
    data4d_zspy_file = fileparts[-2] + "/" + fileparts[-1][:-4] + ".zspy"

    emifile = glob.glob(datapath + "/*.emi")
    if len(emifile) == 0:
        dp = load(mibfile, lazy=False)
        dp.set_diffraction_calibration(1.0)
        dp.axes_manager[1].name = "y"
        dp.axes_manager[1].name = "x"
        dp.axes_manager[1].units = "pixels"
        dp.axes_manager[1].units = "pixels"
        dp.save(data4d_zspy_file, overwrite=ow)

    else:
        sum_dp_file = fileparts[-2] + "/" + fileparts[-1][:-4] + "_Sum_DP.hspy"
        sum_img_file = fileparts[-2] + "/" + fileparts[-1][:-4] + "_Sum_Image.hspy"
        emi = load(emifile[0])
        cl = emi.metadata.Acquisition_instrument.TEM.camera_length / 10
        dwell = emi.metadata.Acquisition_instrument.TEM.Detector.Camera.exposure * 1000
        scan_rotation = (
            emi.original_metadata.ObjectInfo.ExperimentalDescription.Stem_rotation_deg
        )
        pixsize = emi.axes_manager[-1].scale

        dp = load(mibfile, lazy=True)
        dp.metadata.Acquisition_instrument.TEM = emi.metadata.Acquisition_instrument.TEM
        dp.set_experimental_parameters(
            camera_length=cl,
            exposure_time=dwell,
            scan_rotation=scan_rotation,
        )
        dp.set_scan_calibration(pixsize)
        dp.set_diffraction_calibration(1.0)

        sum_dp = dp.sum((0, 1))
        sum_dp.compute()

        sum_img = dp.sum((2, 3))
        sum_img.compute()
        sum_img = sum_img.as_signal2D((0, 1))

        sum_dp.save(sum_dp_file, overwrite=ow, file_format="HSPY")
        sum_img.save(sum_img_file, overwrite=ow, file_format="HSPY")
        dp.save(data4d_zspy_file, overwrite=ow)

        del sum_dp, sum_img
    del dp


def get_data_summary(datapath, text_offset=0.15):
    """
    Summarize the Merlin 4DSTEM experimental parameters in a given directory.

    Data is assumed to be in HyperSpy Zarr format.

    Args
    ----------
    datapath : str
        Location of data to summarize


    """
    zspy_file = glob.glob(datapath + "/*.zspy")[0]
    s = load(zspy_file, lazy=True)
    data_shape = s.data.shape

    dwell = s.metadata.Acquisition_instrument.dwell_time * 1000
    pixel_size = s.axes_manager[0].scale
    pixel_units = s.axes_manager[0].units
    timestamp = datapath[:-1]
    if s.metadata.has_item(
        "Acquisition_instrument.TEM.Detector.Diffraction.camera_length"
    ):
        cl = (
            s.metadata.Acquisition_instrument.TEM.Detector.Diffraction.camera_length
            * 10
        )
        cl = str(round(cl, 1))
    else:
        cl = "Unknown"
    if s.metadata.has_item("Acquisition_instrument.TEM.beam_energy"):
        beam_energy = s.metadata.Acquisition_instrument.TEM.beam_energy
        beam_energy = str(round(beam_energy, 1))
    else:
        beam_energy = "Unknown"
    if s.metadata.has_item("Acquisition_instrument.TEM.scan_rotation"):
        scan_rotation = s.metadata.Acquisition_instrument.TEM.scan_rotation
        scan_rotation = str(round(scan_rotation, 1))
    else:
        scan_rotation = "Unknown"
    filename = zspy_file.split("/")[1]

    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    fig.suptitle("Dataset: %s" % timestamp)
    ax[0].text(text_offset, 0.80, "Filename: %s" % filename)

    sum_img_file = glob.glob(datapath + "/*Sum_Image.hspy")
    if len(sum_img_file) == 0:
        ax[0].text(text_offset, 0.70, "Scan Shape: (%s , )" % data_shape[0])
        ax[1].imshow(np.zeros([256, 256]), cmap="grey")
        ax[1].text(50, 128, "No Scan Data", fontsize=15, color="red")
    else:
        sum_img = load(sum_img_file[0])
        ax[0].text(
            text_offset, 0.70, "Scan Shape: (%s , %s)" % (data_shape[0], data_shape[1])
        )
        ax[1].imshow(sum_img.data, cmap="inferno")

    ax[0].text(text_offset, 0.60, "Pixel Size: %.2f %s" % (pixel_size, pixel_units))
    ax[0].text(text_offset, 0.50, "Frame Time: %.1f msec" % dwell)
    ax[0].text(text_offset, 0.40, "Beam Energy: %s keV" % beam_energy)
    ax[0].text(text_offset, 0.30, "Camera Length: %s mm" % cl)
    ax[0].text(text_offset, 0.20, "Scan Rotation: %s degrees" % scan_rotation)
    ax[0].axis("off")


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
        key_list.append(int(re.search(r"\d+", root_name[::-1]).group()[::-1]))

    sorted_list = [
        filename_list
        for (key_list, filename_list) in sorted(zip(key_list, filename_list))
    ]
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
            with open(mibfiles[i], "r") as h:
                line = h.readline(200)
            exposures[i] = np.float32(line.split(",")[10]) * 1000
    else:
        if type(mibfiles) is list:
            mibfiles = mibfiles[0]
        with open(mibfiles, "rb") as h:
            hdr_temp = np.fromfile(h, "int8", 384)
        hdr_temp = "".join([chr(item) for item in hdr_temp]).split(",")
        header_length = int(hdr_temp[2])
        data_type = hdr_temp[6]
        if data_type == "U32":
            dtype = "uint32"
            data_length = 4
        elif data_type == "U16":
            dtype = "uint16"
            data_length = 2
        else:
            dtype = "uint8"
            data_length = 1
        # If n is not provided, get number of frames from the filesize
        # and read all exposures
        if n is None:
            logger.info("Reading all exposure times")
            n = int(
                os.path.getsize(mibfiles) / (data_length * (256**2) + header_length)
            )
        else:
            logger.info("Reading %s exposure times" % n)
        exposures = np.zeros(n)
        with open(mibfiles, "rb") as h:
            for i in range(0, n):
                hdr_temp = np.fromfile(h, "int8", header_length)
                hdr_temp = "".join([chr(item) for item in hdr_temp]).split(",")
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
    with open(hdrfile, "r") as h:
        _ = h.readlines(1)[0].rstrip()
        header["TimeStamp"] = h.readlines(1)[0].rstrip().split(":\t")[1]
        header["ChipID"] = h.readlines(1)[0].rstrip().split(":\t")[1]
        header["ChipType"] = h.readlines(1)[0].rstrip().split(":\t")[1]
        header["ChipConfig"] = h.readlines(1)[0].rstrip().split(":\t")[1][3:]
        header["ChipMode"] = h.readlines(1)[0].rstrip().split(":\t")[1]
        header["CounterDepth"] = h.readlines(1)[0].rstrip().split(":\t")[1]
        header["Gain"] = h.readlines(1)[0].rstrip().split(":\t")[1]
        header["ActiveCounters"] = h.readlines(1)[0].rstrip().split(":\t")[1]
        header["Thresholds"] = h.readlines(1)[0].rstrip().split(":\t")[1]
        header["DACS"] = h.readlines(1)[0].rstrip().split(":\t")[1]
        header["BPC_File"] = h.readlines(1)[0].rstrip().split(":\t")[1]
        header["DAC_File"] = h.readlines(1)[0].rstrip().split(":\t")[1]
        header["GapFill"] = h.readlines(1)[0].rstrip().split(":\t")[1]
        header["FlatField_File"] = h.readlines(1)[0].rstrip().split(":\t")[1]
        header["DeadTime_File"] = h.readlines(1)[0].rstrip().split(":\t")[1]
        header["AcqType"] = h.readlines(1)[0].rstrip().split(":\t")[1]
        header["TotalFrames"] = h.readlines(1)[0].rstrip().split(":\t")[1]
        header["FramesPerTrigger"] = h.readlines(1)[0].rstrip().split(":\t")[1]
        header["TriggerStart"] = h.readlines(1)[0].rstrip().split(":\t")[1]
        header["TriggerStop"] = h.readlines(1)[0].rstrip().split(":\t")[1]
        header["Bias"] = h.readlines(1)[0].rstrip().split(":\t")[1]
        header["Polarity"] = h.readlines(1)[0].rstrip().split(":\t")[1]
        header["Temperature"] = h.readlines(1)[0].rstrip().split(":\t")[1]
        header["Humidity"] = h.readlines(1)[0].rstrip().split(":\t")[1]
        header["Clock"] = h.readlines(1)[0].rstrip().split(":\t")[1]
        header["Readout"] = h.readlines(1)[0].rstrip().split(":\t")[1]
        header["SoftwareVersion"] = h.readlines(1)[0].rstrip().split(":\t")[1]
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
    with open(mibfile, "r") as h:
        hdr_temp = np.fromfile(h, "int8", 384)
        hdr_temp = "".join([chr(item) for item in hdr_temp]).split(",")

    mib_hdr["HeaderID"] = hdr_temp[0]
    mib_hdr["AcquisitionSequenceNumber"] = hdr_temp[1]
    mib_hdr["DataOffset"] = np.int32(hdr_temp[2])
    mib_hdr["NumChips"] = np.int32(hdr_temp[3])
    mib_hdr["PixDimX"] = np.int32(hdr_temp[4])
    mib_hdr["PixDimY"] = np.int32(hdr_temp[5])
    mib_hdr["PixDepth"] = hdr_temp[6]
    mib_hdr["SensorLayout"] = hdr_temp[7][3:]
    mib_hdr["ChipSelect"] = np.int32(hdr_temp[8])
    mib_hdr["TimeStamp"] = hdr_temp[9]
    mib_hdr["ShutterTime"] = np.float32(hdr_temp[10])
    mib_hdr["Counter"] = np.int32(hdr_temp[11])
    mib_hdr["ColourMode"] = np.int32(hdr_temp[12])
    mib_hdr["GainMode"] = np.int32(hdr_temp[13])
    mib_hdr["Thresholds"] = np.zeros(8, np.float32)
    for i in range(0, 8):
        mib_hdr["Thresholds"][i] = np.float32(hdr_temp[14 + i])
    mib_hdr["DACs"] = {}
    mib_hdr["DACs"]["Format"] = hdr_temp[22]
    mib_hdr["DACs"]["Thresh0"] = np.uint16(hdr_temp[23])
    mib_hdr["DACs"]["Thresh1"] = np.uint16(hdr_temp[24])
    mib_hdr["DACs"]["Thresh2"] = np.uint16(hdr_temp[25])
    mib_hdr["DACs"]["Thresh3"] = np.uint16(hdr_temp[26])
    mib_hdr["DACs"]["Thresh4"] = np.uint16(hdr_temp[27])
    mib_hdr["DACs"]["Thresh5"] = np.uint16(hdr_temp[28])
    mib_hdr["DACs"]["Thresh6"] = np.uint16(hdr_temp[29])
    mib_hdr["DACs"]["Thresh7"] = np.uint16(hdr_temp[30])
    mib_hdr["DACs"]["Preamp"] = np.uint8(hdr_temp[31])
    mib_hdr["DACs"]["Ikrum"] = np.uint8(hdr_temp[32])
    mib_hdr["DACs"]["Shaper"] = np.uint8(hdr_temp[33])
    mib_hdr["DACs"]["Disc"] = np.uint8(hdr_temp[34])
    mib_hdr["DACs"]["DiscLS"] = np.uint8(hdr_temp[35])
    mib_hdr["DACs"]["ShaperTest"] = np.uint8(hdr_temp[36])
    mib_hdr["DACs"]["DACDiscL"] = np.uint8(hdr_temp[37])
    mib_hdr["DACs"]["DACTest"] = np.uint8(hdr_temp[38])
    mib_hdr["DACs"]["DACDISCH"] = np.uint8(hdr_temp[39])
    mib_hdr["DACs"]["Delay"] = np.uint8(hdr_temp[40])
    mib_hdr["DACs"]["TPBuffIn"] = np.uint8(hdr_temp[41])
    mib_hdr["DACs"]["TPBuffOut"] = np.uint8(hdr_temp[42])
    mib_hdr["DACs"]["RPZ"] = np.uint8(hdr_temp[43])
    mib_hdr["DACs"]["GND"] = np.uint8(hdr_temp[44])
    mib_hdr["DACs"]["TPRef"] = np.uint8(hdr_temp[45])
    mib_hdr["DACs"]["FBK"] = np.uint8(hdr_temp[46])
    mib_hdr["DACs"]["Cas"] = np.uint8(hdr_temp[47])
    mib_hdr["DACs"]["TPRefA"] = np.uint16(hdr_temp[48])
    mib_hdr["DACs"]["TPRefB"] = np.uint16(hdr_temp[49])
    mib_hdr["ExtID"] = hdr_temp[50]
    mib_hdr["ExtTimeStamp"] = hdr_temp[51]
    mib_hdr["ExtID"] = np.float64(hdr_temp[52][:-2])
    mib_hdr["ExtCounterDepth"] = np.uint8(hdr_temp[53])
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
    n_detector_pix = mib_hdr["PixDimX"] * mib_hdr["PixDimY"]
    header_length = mib_hdr["DataOffset"]
    if mib_hdr["PixDepth"] == "U08":
        data_length = 1
    elif mib_hdr["PixDepth"] == "U16":
        data_length = 2
    elif mib_hdr["PixDepth"] == "U32":
        data_length = 4
    if len(mibfiles) == 1:
        total_frames = int(
            os.path.getsize(mibfiles[0])
            / (data_length * n_detector_pix + header_length)
        )
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
        logger.warning("Missing %s frames at end of dataset!" % (-extra_frames))

    scanXalu = [np.arange(0, scan_width), "x", "pixels"]
    scanYalu = [np.arange(0, scan_height), "y", "pixels"]
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
        with h5py.File(data, "r") as h5:
            scanY, scanX, detY, detX = h5["fpd_expt/fpd_data/data"].shape
            exposures = np.round(h5["fpd_expt/Exposure/data"][...] / 1e6, 1)
            frame_times, counts = np.unique(exposures, return_counts=True)
            positive_idx = np.where(frame_times >= 0)
            frame_times = frame_times[positive_idx]
            counts = counts[positive_idx]
            frame_time = frame_times[counts.argmax()]
            thresholds = h5["fpd_expt/Threshold/data"][...][:, :, 0]
            threshold = np.unique(thresholds[~np.isnan(thresholds)])[0]

    params = {
        "Frame time": frame_time,
        "Scan shape": [scanY, scanX],
        "Detector shape": [detY, detX],
        "Threshold": threshold,
    }
    return params


def get_microscope_parameters(data, display=False):
    """
    Get microscope parameters for 4D-STEM data.

    Reads camera length, high tension, and magnifiction from metadata
    in HDF5 file or FPD named tuple.

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
            logger.info("Found microscope parameters in DM metadata in FPD file")
            cl = float(
                data[
                    "fpd_expt/DM0/tags/ImageList/TagGroup0/ImageTags/"
                    "Microscope Info/STEM Camera Length"
                ][...]
            )
            ht = (
                float(
                    data[
                        "fpd_expt/DM0/tags/ImageList/TagGroup0/ImageTags/"
                        "Microscope Info/Voltage"
                    ][...]
                )
                / 1000
            )
            mag = float(
                data[
                    "fpd_expt/DM0/tags/ImageList/TagGroup0/ImageTags/"
                    "Microscope Info/Indicated Magnification"
                ][...]
            )
        elif "TIA0" in data["/fpd_expt/"].keys():
            if "ExperimentalDescription" in data["/fpd_expt/TIA0/tags/ObjectInfo"]:
                logger.info("Found microscope parameters in TIA metadata in FPD file")
                cl = float(
                    1000
                    * data["fpd_expt/TIA0/tags/ObjectInfo/ExperimentalDescription"][
                        "Camera length_m"
                    ][...]
                )
                mag = float(
                    data["fpd_expt/TIA0/tags/ObjectInfo/ExperimentalDescription"][
                        "Magnification_x"
                    ][...]
                )
                ht = float(
                    data["fpd_expt/TIA0/tags/ObjectInfo/ExperimentalDescription"][
                        "High tension_kV"
                    ][...]
                )
        else:
            logger.info("Unable to find microscope parameters in FPD file")
            cl = "Unknown"
            ht = "Unknown"
            mag = "Unknown"
    elif isinstance(data, str):
        h5_has_dm = False
        h5_has_tia = False
        if os.path.splitext(data)[-1].lower() == ".hdf5":
            with h5py.File(data, "r") as h5:
                h5keys = h5["/fpd_expt/"].keys()
                h5_has_dm = "DM0" in h5keys
                if "TIA0" in h5keys:
                    if (
                        "ExperimentalDescription"
                        in h5["/fpd_expt/TIA0/tags/ObjectInfo"].keys()
                    ):
                        h5_has_tia = True
            if h5_has_dm:
                with h5py.File(data, "r") as h5:
                    cl = float(
                        h5[
                            "fpd_expt/DM0/tags/ImageList/TagGroup0/ImageTags/"
                            "Microscope Info/STEM Camera Length"
                        ][...]
                    )
                    ht = (
                        float(
                            h5[
                                "fpd_expt/DM0/tags/ImageList/TagGroup0/ImageTags/"
                                "Microscope Info/Voltage"
                            ][...]
                        )
                        / 1000
                    )
                    mag = float(
                        h5[
                            "fpd_expt/DM0/tags/ImageList/TagGroup0/ImageTags/"
                            "Microscope Info/Indicated Magnification"
                        ][...]
                    )
                    logger.info("Found DM metadata in FPD file")
            elif h5_has_tia:
                with h5py.File(data, "r") as h5:
                    cl = float(
                        1000
                        * h5["fpd_expt/TIA0/tags/ObjectInfo/ExperimentalDescription"][
                            "Camera length_m"
                        ][...]
                    )
                    ht = float(
                        h5["fpd_expt/TIA0/tags/ObjectInfo/ExperimentalDescription"][
                            "High tension_kV"
                        ][...]
                    )
                    mag = float(
                        h5["fpd_expt/TIA0/tags/ObjectInfo/ExperimentalDescription"][
                            "Magnification_x"
                        ][...]
                    )
                    logger.info("Found TIA metadata in FPD file")
            elif len(glob.glob(os.path.split(data)[0] + "/*.emi")) > 0:
                emifile = glob.glob(os.path.split(data)[0] + "/*.emi")[0]
                im = load(emifile)
                if type(im) is list:
                    im = im[0]
                cl = im.metadata.Acquisition_instrument.TEM.camera_length
                ht = im.metadata.Acquisition_instrument.TEM.beam_energy
                mag = im.metadata.Acquisition_instrument.TEM.magnification
                logger.info("Found microscope params in EMI file")
            else:
                logger.info("Unable to find microscope parameters in FPD file")
                cl = "Unknown"
                ht = "Unknown"
                mag = "Unknown"
    elif isinstance(data, tuple) and hasattr(data, "_fields"):
        if "DM0" in data._fields:
            cl = float(
                data.DM0.tags[
                    "ImageList/TagGroup0/ImageTags/Microscope Info/STEM Camera Length"
                ][...]
            )
            ht = (
                float(
                    data.DM0.tags[
                        "ImageList/TagGroup0/ImageTags/Microscope Info/Voltage"
                    ][...]
                )
                / 1000
            )
            mag = float(
                data.DM0.tags[
                    "ImageList/TagGroup0/ImageTags/"
                    "Microscope Info/Indicated Magnification"
                ][...]
            )
            logger.info("Found microscope parameters in DM metadata of FPD file")
        elif "TIA0" in data._fields:
            if "ExperimentalDescription" in data.TIA0.tags["ObjectInfo"].keys():
                cl = float(
                    1000
                    * data.TIA0.tags["ObjectInfo/ExperimentalDescription"][
                        "Camera length_m"
                    ][...]
                )
                ht = float(
                    data.TIA0.tags["ObjectInfo/ExperimentalDescription"][
                        "High tension_kV"
                    ][...]
                )
                mag = float(
                    data.TIA0.tags["ObjectInfo/ExperimentalDescription"][
                        "Magnification_x"
                    ][...]
                )
                logger.info("Found microscope parameters in TIA metadata of FPD file")
        else:
            logger.info("Unable to find microscope parameters in FPD file")
            cl = "Unknown"
            ht = "Unknown"
            mag = "Unknown"
    else:
        logger.info("Unable to find microscope parameters in FPD file")
        cl = "Unknown"
        ht = "Unknown"
        mag = "Unknown"
    if display:
        print("Microscope voltage: %.1f kV" % ht)
        print("STEM Camera Length: %.1f mm" % cl)
        print("Magnification: %.1f X" % mag)

    params = {"CL": cl, "HT": ht, "Magnification": mag}
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

    if unitsX in ["um", "µm"]:
        unitsX = "nm"
        scaleX = scaleX * 1000
        originX = originX * 1000

    unitsY = nt.fpd_data.dim1.units
    originY = nt.fpd_data.dim1.data[0]
    scaleY = nt.fpd_data.dim1.data[1] - nt.fpd_data.dim1.data[0]
    sizeY = nt.fpd_data.dim1.data.shape[0]

    if unitsY in ["um", "µm"]:
        unitsY = "nm"
        scaleY = scaleY * 1000
        originY = originY * 1000

    axes_dict = {
        "axis-0": {
            "name": "y",
            "offset": originY,
            "units": unitsY,
            "scale": scaleY,
            "size": sizeY,
        },
        "axis-1": {
            "name": "x",
            "offset": originX,
            "units": unitsX,
            "scale": scaleX,
            "size": sizeX,
        },
    }
    return axes_dict


def get_experimental_parameters(h5files):
    """
    Create a Pandas DataFrame containing experimental parameters from HDF5 files.

    Args
    ----------
    h5files : str or list
        Files to parse

    Returns
    ----------
    df : DataFrame
        Pandas DataFrame containing the extracted experimental parameters

    """
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
        with h5py.File(h5files[i], "r") as h5:
            h5keys = h5.keys()
            if "filename" in h5keys:
                if type(h5["filename"][()]) is str:
                    parent_filename = h5["filename"][()]
                else:
                    parent_filename = h5["filename"][()].decode()
        if fpd_check(h5files[i]):
            microscope_params = get_microscope_parameters(h5files[i])
            hts[i] = microscope_params["HT"]
            cls[i] = microscope_params["CL"]
            mags[i] = microscope_params["Magnification"]

            merlin_params = get_merlin_parameters(h5files[i])
            scan_shapes[i] = "%s x %s" % (
                merlin_params["Scan shape"][0],
                merlin_params["Scan shape"][1],
            )
            det_shapes[i] = "%s x %s" % (
                merlin_params["Detector shape"][0],
                merlin_params["Detector shape"][1],
            )
            frame_times[i] = merlin_params["Frame time"]
            thresholds[i] = merlin_params["Threshold"]
        elif parent_filename:
            microscope_params = get_microscope_parameters(parent_filename)
            hts[i] = microscope_params["HT"]
            cls[i] = microscope_params["CL"]
            mags[i] = microscope_params["Magnification"]

            merlin_params = get_merlin_parameters(parent_filename)
            scan_shapes[i] = "%s x %s" % (
                merlin_params["Scan shape"][0],
                merlin_params["Scan shape"][1],
            )
            det_shapes[i] = "%s x %s" % (
                merlin_params["Detector shape"][0],
                merlin_params["Detector shape"][1],
            )
            frame_times[i] = merlin_params["Frame time"]
            thresholds[i] = merlin_params["Threshold"]

    df = pd.DataFrame()
    df["Data Path"] = datapaths
    df["H5 File"] = h5filenames
    df["Beam Energy"] = hts
    df["Camera Length"] = cls
    df["Mag"] = mags
    df["Scan Shape"] = scan_shapes
    df["Detector Shape"] = det_shapes
    df["Frame time"] = frame_times
    df["Threshold"] = thresholds

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
        with h5py.File(data, "r") as h5:
            if "/fpd_expt/" in h5.keys():
                return True
            else:
                return False
    elif isinstance(data, h5py.File):
        if "/fpd_expt/" in data.keys():
            return True
        else:
            return False


def create_dataset(h5file, full_align=False, check_file=True):
    """
    Read an FPD dataset and perform useful operations.

    Args
    ----------
    h5file : str
        Filename of FPD generated HDF5 file

    Returns
    ----------
    dataset : dict
        Dictionary containing all FPD generated data as well as microscope parameters,
        reciprocal space calibration, center of mass, alignment shifts, and aligned
        radial profile.

    """
    params = get_microscope_parameters(h5file)
    if params["HT"] == "Unknown":
        qcal = 1
        qcal_units = "pixels"
    else:
        qcal = get_calibration(params["HT"], params["CL"], "q")
        qcal_units = "A^-1"
    nt = fpdf.fpd_to_tuple(h5file, fpd_check=check_file)

    # Get scan dimension calibrations
    ycal = nt.fpd_sum_im.dim1.data[1] - nt.fpd_sum_im.dim1.data[0]
    ycal_units = nt.fpd_sum_im.dim1.units
    xcal = nt.fpd_sum_im.dim2.data[1] - nt.fpd_sum_im.dim2.data[0]
    xcal_units = nt.fpd_sum_im.dim2.units

    com_yx = fpdp.center_of_mass(nt.fpd_data.data, 32, 32, print_stats=False)
    # Remove NaN's from com_yx and replace with nearest non-NaN value
    mask = np.isnan(com_yx)
    com_yx[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), com_yx[~mask])
    min_center = np.percentile(com_yx, 50, (-2, -1))
    radial_shifts = -(com_yx - min_center[..., None, None])
    ali_shifts = com_yx - min_center[..., None, None]
    if full_align:
        ali = shift_align(nt.fpd_data.data, ali_shifts, 32, 32, True, 3)
        sum_im = ali.sum((2, 3))
        sum_dp = ali.sum((0, 1))
    else:
        ali = None
        sum_im = nt.fpd_sum_im.data
        sum_dp = nt.fpd_sum_dif.data
    profile = radial_profile(nt.fpd_data.data, com_yx, qcal)
    dataset = {
        "filename": h5file,
        "nt": nt,
        "params": params,
        "qcal": qcal,
        "qcal_units": qcal_units,
        "xcal": xcal,
        "xcal_units": xcal_units,
        "ycal": ycal,
        "ycal_units": ycal_units,
        "com_yx": com_yx,
        "min_center": min_center,
        "shifts": radial_shifts,
        "radial_profile": profile,
        "apertures": None,
        "radii": None,
        "aligned": ali,
        "sum_image": sum_im,
        "sum_dp": sum_dp,
        "images": {},
    }
    return dataset


def save_results(h5filename, dataset):
    """
    Save an FPD dataset along with other parameters to HDF5 file.

    Args
    ----------
    h5filename : str
        Filename for saving

    dataset : dict
        Dictionary as specified by merlintools.io.create_dataset.

    """
    with h5py.File(h5filename, "w") as h5:
        for k in dataset.keys():
            if dataset[k] is None:
                h5.create_dataset(k, data=h5py.Empty(None))
            elif k == "nt":
                pass
            elif k == "params":
                for item, value in dataset["params"].items():
                    h5.create_dataset(k + "/" + item, data=value)
            elif k == "radial_profile":
                grp = h5.create_group(k)
                grp.create_dataset("x", data=dataset[k][0])
                grp.create_dataset("y", data=dataset[k][1])
            elif k == "images":
                grp = h5.create_group("images")
                for im in dataset["images"].keys():
                    grp.create_dataset(im, data=dataset["images"][im])
            else:
                h5.create_dataset(k, data=dataset[k])
    return


def read_h5_results(h5file):
    """
    Read HDF5 file generated using merlintools.io.save_results.

    Args
    ----------
    h5file : str
        Filename of HDF5 file to read

    Returns
    ----------
    dataset : dict
        Dictionary containing all FPD generated data as well as microscope parameters,
        reciprocal space calibration, center of mass, alignment shifts, and aligned
        radial profile, etc.

    """
    data_keys = [
        "qcal",
        "xcal",
        "min_center",
        "ycal",
        "shifts",
        "com_yx",
        "radii",
        "apertures",
    ]
    str_keys = ["qcal_units", "xcal_units", "ycal_units"]
    dataset = {}
    with h5py.File(h5file, "r") as h5:
        dataset["filename"] = h5["filename"][()]
        if type(dataset["filename"]) is bytes:
            dataset["filename"] = dataset["filename"].decode()
        dataset["nt"] = fpdf.fpd_to_tuple(dataset["filename"])
        dataset["params"] = {
            "CL": "Unknown",
            "HT": "Unknown",
            "Magnification": "Unknown",
        }
        for param in ["CL", "HT", "Magnification"]:
            if type(h5["params/%s" % param][()]) is bytes:
                pass
            elif h5["params/%s" % param][()] == "Unknown":
                pass
            else:
                dataset["params"][param] = np.float32(h5["params/%s" % param][...])
        for k in data_keys:
            if k in h5.keys():
                dataset[k] = h5[k][...]
        for k in str_keys:
            if k in h5.keys():
                if type(h5[k][()]) is str:
                    dataset[k] = h5[k][()]
                else:
                    dataset[k] = h5[k].asstr()[()]
        for k in ["sum_image", "sum_dp", "aligned"]:
            dataset[k] = (h5[k][...],)
            dataset[k] = dataset[k][0]
        if "radial_profile" in h5.keys():
            dataset["radial_profile"] = [
                h5["radial_profile/x"][...],
                h5["radial_profile/y"][...],
            ]
        if "images" in h5.keys():
            dataset["images"] = {}
            for im in h5["images"].keys():
                dataset["images"][im] = h5["images"][im][...]
    return dataset


def read_single_mib(filename, det_shape=[256, 256]):
    """
    Quickly read an individual MIB file.

    Args
    ----------
    filename : str
        MIB file to be read
    det_shape : tuple
        Size of detectore. Default is 256x256

    Returns
    ----------
    dp : NumPy array

    """
    header = parse_mib_header(filename)
    header_length = header["DataOffset"]
    data_type = header["PixDepth"]
    if data_type == "U16":
        data_type = ">H"
    elif data_type == "U32":
        data_type = ">L"

    with open(filename, "rb") as h:
        _ = np.fromfile(h, np.int8, header_length)
        dp = np.fromfile(h, dtype=data_type)
    dp = dp.reshape(det_shape)
    return dp
