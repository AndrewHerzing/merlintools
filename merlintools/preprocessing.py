# -*- coding: utf-8 -*-
#
# This file is part of MerlinTools

"""
preprocessing module for MerlinTools package.

@author: Andrew Herzing
"""

import os
import glob
import logging
import fpd.fpd_file as fpdf
from merlintools.io import get_scan_shape, create_dataset, save_results
from hyperspy.io import load

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_merlin_binary(mib, hdr, emi, ser, dm):
    '''Basic check to see if TIA SER files are present. If none are present, then the scanX and scanY variables
    are set to None and FPD will not attempt to reshape the data. If SER files are present alongside an EMI
    file, the data is loaded via the EMI file as this will pull in the microscope parameters as well.
    If SER files are present but no EMI file is present, load the data using the SER files.
    I'm sure Nexus has more robust ways of handling TIA EMI/SER file pairs.'''
    save_dm = False
    save_tia = False
    if len(ser) == 0 and len(dm) == 0:
        scanX = None
        scanY = None
    elif len(ser) > 0:
        save_tia = True
        if len(emi) == 0:
            logger.info("'No EMI file found.  Microscope parameters cannot be determined")
            im = load(ser[0])
            tiafiles = ser
            '''Define scan axes for use with fpd.fpd_file.MerlinBinary using metadata from SER file'''
            scanX, scanY = [(ax.axis, ax.name, ax.units) for ax in im.axes_manager.signal_axes]
        else:
            im = load(emi)
            '''If multiple SER files are present, loading the EMI will result in a list of Signal2D objects.
            Check if this is the case.  If so, discard all but the first Signal2D.'''
            if type(im) is list:
                im = im[0]
            tiafiles = emi
            '''Define scan axes for use with fpd.fpd_file.MerlinBinary using metadata from SER file'''
            scanX, scanY = [(ax.axis, ax.name, ax.units) for ax in im.axes_manager.signal_axes]
    elif len(dm) > 0:
        scanX, scanY, skip_frames, total_frames = get_scan_shape(mib)
        save_dm = True

    if save_dm:
        mb = fpdf.MerlinBinary(binfns=mib,
                               hdrfn=hdr[0],
                               ds_start_skip=skip_frames,
                               row_end_skip=0,
                               dmfns=dm[0],
                               sort_binary_file_list=False,
                               strict=False)
    elif save_tia:
        mb = fpdf.MerlinBinary(binfns=mib,
                               hdrfn=hdr[0],
                               ds_start_skip=0,
                               row_end_skip=0,
                               tiafns=tiafiles[0],
                               sort_binary_file_list=False,
                               strict=False)
    else:
        mb = fpdf.MerlinBinary(binfns=mib,
                               hdrfn=hdr[0],
                               ds_start_skip=0,
                               row_end_skip=0,
                               scanXalu=scanX,
                               scanYalu=scanY,
                               sort_binary_file_list=False,
                               strict=False)
    return mb


def merlin_to_fpd(datadir, savedir='./', reshape_dm=False):
    """
    Convert Merlin files to FPD HDF5 archive and transfer to a storage location.

    Args
    ----------
    datadir : str
        Rootpath containing Merlin datasets.
    reshape_dm : bool
        If True, reshape data generated using Digital Micrograph by attempting to determine scan
        shape based on exposure times.  This usually works but should be disabled if stability is important.
        If False, DM generated data will be saved without reshaping. Default is False.

    """

    '''Pull filenames from datadir and define output HDF5 file name'''
    mib = glob.glob(datadir + '/*.mib')
    hdr = glob.glob(datadir + '/*.hdr')
    emi = glob.glob(datadir + '/*.emi')
    ser = glob.glob(datadir + '/*.ser')
    dm = glob.glob(datadir + '/*.dm*')
    h5file = os.path.splitext(mib[0])[0] + '.hdf5'
    h5file = os.path.join(savedir, os.path.split(h5file)[1])

    '''Basic check to see if Merlin MIB and HDR files are present'''
    if len(mib) == 0 or len(hdr) == 0:
        raise ValueError('Merlin data not found!')

    mb = get_merlin_binary(mib, hdr, emi, ser, dm)
    '''Write MerlinBinary to HDF5 file'''
    mb.write_hdf5(h5file, allow_memmap=False)
    return


def preprocess(datadir, processed_data_path=".", full_align=False, check_fpd=True):
    """Perform preprocessing steps on FPD dataset
    Convert Merlin files to FPD HDF5 archive and transfer to a storage location.

    Args
    ----------
    datadir : str
        Rootpath containing FPD datasets to preprocess.
    processed_data_path : str
        Path where preprocessed data will be saved.
    full_align : bool
        If True, data set is fully aligned and saved in results file.  Otherwise,
        preprocessing is done without shifting the data.

    """
    h5files = glob.glob(datadir + "/**/*.hdf5", recursive=True)
    data = [None] * len(h5files)
    for i in range(0, len(h5files)):
        data[i] = create_dataset(h5files[i], full_align, check_fpd)
        outpath = os.path.join(processed_data_path, h5files[i].split('/')[-3], h5files[i].split('/')[-2])
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        outfile = outpath + '/' + h5files[i].split('/')[-1][:-5] + '_Processed.hdf5'
        save_results(outfile, data[i])
    return
