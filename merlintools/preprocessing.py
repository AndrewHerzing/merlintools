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
from merlintools.io import get_scan_shape
import tkinter as tk
from tkinter import filedialog
import time
import shutil
from pathlib import Path
from hyperspy.io import load
import h5py

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def merlin_to_fpd_legacy(rootpath=None, savepath=None, keep_unshaped=False,
                         shutdown=False, discard_first_column=False,
                         discard_data=False):
    """
    Convert Merlin files to FPD HDF5 archive and transfer to a storage location.

    Args
    ----------
    rootpath : str
        Directory where data to be processed is stored. All sub-directories
        will be searched for Merlin data and processed in turn. If None, a
        dialog will prompt for the save directory.
    savepath : str
        Path where data will be saved.  If None, a dialog will prompt for the
        save directory.
    keep_unshaped : bool
        If True, a second FPD archive will be saved where no reshaping is
        performed and no frames are discarded.
    shutdown : bool
        If True, shutdown the local PC after processing.  Default is False.
    discard_first_column : bool
        If True, the first column of data will be discarded after reshaping
        in an attempt to eliminate the flyback pixels. Default is False.
    discard_data : bool
        If True, the local data will be deleted after processing. Default
        is False.

    Returns
    ----------
    h5filenames : list
        List of the resulting HDF5 filenames.

    """
    if not rootpath:
        root = tk.Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', True)
        rootpath = filedialog.askdirectory(initialdir="c:/users/aherzing/data",
                                           title="Select data directory...")
        rootpath = rootpath + "/"
    if not savepath:
        startpath = os.path.abspath(os.path.join(rootpath, "..")).\
            replace('\\', '/')
        root = tk.Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', True)
        savepath = filedialog.askdirectory(initialdir=startpath,
                                           title="Select save directory...")
        savepath = savepath + "/"

    root_parent = str(Path(rootpath).parent.absolute())

    if os.access("/", os.W_OK):
        temp_dir = root_parent +\
            time.strftime("/%Y_%m_%d_%H%M%S_FPD_EXPORT/")
    else:
        temp_dir = os.getcwd() +\
            time.strftime("/%Y_%m_%d_%H%M%S_FPD_EXPORT/")
    if savepath[-1:] != "/":
        savepath = savepath + "/"

    dirs = [x[0] for x in os.walk(rootpath)]
    empty_dirs = []
    for i in dirs:
        if len(glob.glob(i + "/*.mib")) == 0:
            empty_dirs.append(i)
    [dirs.remove(i) for i in empty_dirs]
    temppaths = [None] * len(dirs)
    outpaths = [None] * len(dirs)
    tempfilenames = [None] * len(dirs)
    h5filenames = [None] * len(dirs)

    for i in range(0, len(dirs)):
        mibfiles = glob.glob(dirs[i] + "/*.mib")
        mibfiles = [filepath.replace('\\', '/') for filepath in mibfiles]
        temppaths[i] = temp_dir + mibfiles[0].split('/')[-3] +\
            '/' + mibfiles[0].split('/')[-2] + '/'
        outpaths[i] = savepath + mibfiles[0].split('/')[-3] +\
            '/' + mibfiles[0].split('/')[-2] + '/'

        if not os.path.isdir(temppaths[i]):
            os.makedirs(temppaths[i])
        tempfilenames[i] = temppaths[i] + \
            os.path.splitext(os.path.split(mibfiles[0])[1])[0] + '.hdf5'
        h5filenames[i] = outpaths[i] + \
            os.path.splitext(os.path.split(mibfiles[0])[1])[0] + '.hdf5'

    for i in range(0, len(dirs)):
        mibfiles = glob.glob(dirs[i] + "/*.mib")
        hdrfile = glob.glob(dirs[i] + "/*.hdr")[0]
        dmfile = glob.glob(dirs[i] + "/*.dm*")
        tiafile = glob.glob(dirs[i] + "/*.emi")

        logger.info("Merlin Data File: %s" % mibfiles[0])
        logger.info("Merlin Header File: %s" % hdrfile)
        logger.info("Saving to path: %s" % outpaths[i])

        if len(dmfile) > 0:
            logger.info("Found DM file: %s" % dmfile[0])
            save_dm = True
            scanX, scanY, skip_frames, total_frames = get_scan_shape(mibfiles)
        elif len(tiafile) > 0:
            logger.info("Found TIA file: %s" % tiafile[0])
            save_dm = False
            im = load(tiafile)
            scanX, scanY = [(ax.axis, ax.name, ax.units) for ax in im.axes_manager.signal_axes]
            skip_frames = 0
        else:
            save_dm = False
            scanX, scanY, skip_frames, total_frames = get_scan_shape(mibfiles)

        if keep_unshaped:
            unshapedfilename = tempfilenames[i][:-5] + "_Unshaped.hdf5"
            unshaped = fpdf.MerlinBinary(binfns=mibfiles,
                                         hdrfn=hdrfile,
                                         ds_start_skip=0,
                                         row_end_skip=0,
                                         sort_binary_file_list=False,
                                         strict=False,
                                         repack=True)

            logger.info("Saving unshaped data to file: %s" % unshapedfilename)
            unshaped.write_hdf5(unshapedfilename, ow=True, allow_memmap=False)
            del unshaped

        if save_dm:
            s = fpdf.MerlinBinary(binfns=mibfiles,
                                  hdrfn=hdrfile,
                                  ds_start_skip=skip_frames,
                                  row_end_skip=0,
                                  dmfns=dmfile[0],
                                  sort_binary_file_list=False,
                                  strict=False,
                                  repack=True)

            logger.info("Saving to file w/ DM data: %s" % h5filenames[i])
        else:
            if discard_first_column:
                skip_frames += 1
                scanX[0] = scanX[0][:-1]
                end_skip = 1
            else:
                end_skip = 0
            s = fpdf.MerlinBinary(binfns=mibfiles,
                                  hdrfn=hdrfile,
                                  ds_start_skip=skip_frames,
                                  row_end_skip=end_skip,
                                  scanXalu=scanX,
                                  scanYalu=scanY,
                                  sort_binary_file_list=False,
                                  strict=False,
                                  repack=True)

            logger.info("Saving to file: %s" % h5filenames[i])
        s.write_hdf5(tempfilenames[i], ow=True, allow_memmap=False)
        del s

    shutil.copytree(temp_dir, savepath)
    if discard_data:
        shutil.rmtree(rootpath)
    if shutdown:
        logger.info("Processing complete. Shutting down computer.")
        os.system("shutdown /s /t 1")
    else:
        logger.info("Processing complete.")
    return h5filenames

def merlin_to_fpd(datadir, reshape_dm=False):
    '''Pull filenames from datadir and define output HDF5 file name'''
    mib = glob.glob(datadir + '*.mib')
    hdr = glob.glob(datadir + '*.hdr')
    emi = glob.glob(datadir + '*.emi')
    ser = glob.glob(datadir + '*.ser')
    dm = glob.glob(datadir + '*.dm*')
    h5file = os.path.splitext(mib[0])[0] + '.hdf5'

    '''Basic check to see if Merlin MIB and HDR files are present'''
    if len(mib) == 0 or len(hdr) == 0:
        raise ValueError('Merlin data not found!')

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
            '''Define scan axes for use with fpd.fpd_file.MerlinBinary using metadata from SER file'''
            scanX, scanY = [(ax.axis, ax.name, ax.units) for ax in im.axes_manager.signal_axes]
        else:
            im = load(emi)
            '''If multiple SER files are present, loading the EMI will result in a list of Signal2D objects.
            Check if this is the case.  If so, discard all but the first Signal2D.'''
            if type(im) is list:
                im = im[0]
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
    else:
        mb = fpdf.MerlinBinary(binfns=mib,
                               hdrfn=hdr[0],
                               ds_start_skip=0,
                               row_end_skip=0,
                               scanXalu=scanX,
                               scanYalu=scanY,
                               sort_binary_file_list=False,
                               strict=False)


    '''Write MerlinBinary to HDF5 file'''
    #TODO: Check that data chunking parameters are best configured for Nexus.  This coce uses the default options where
    #      the data is chunked by 16 in all dimensions.
    mb.write_hdf5(h5file, allow_memmap=False)

    if save_tia:
        with h5py.File(h5file, 'r+') as h5:
            h5.create_dataset('TIA', data=im)
    return
        
        