import os
import glob
import logging
import fpd
from merlintools.io import get_scan_shape
import tkinter as tk
from tkinter import filedialog
import time
import shutil
from pathlib import Path

try:
    optional_package = None
    import pyxem as optional_package
except ImportError:
    pass

if optional_package:
    import pyxem as px


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def preprocess(fpd_filename=None, center=True, com_threshold=3,
               shift_interpolation=0, r_bf=15, r_adf_inner=60,
               r_adf_outer=100, save_results=True, return_all=False,
               overwrite=False, outpath=None):
    """
    Convert FPD files to PyXEM/Hyperspy signals.

    Args
    ----------
    fpd_filename : str
        FPD archive from which to extract signals. If None, a file dialog will
        prompt for the file location.
    center : bool
        If True, perform center-of-mass-based centering of the diffraction
        patterns.
    com_threshold : int or float
        Threshold to apply prior to center of mass calculation.  All values
        below com_threshold*mean will be set to 0.
    shift_interpolation:
        Interpolation mode for pattern centering.
    r_bf : int or float
        Outer radius (in pixels) to describe the mask for producing a
        bright-field pseudo-image
    r_adf_inner : int or float
        Inner radius (in pixels) to describe the mask for producing an
        annular dark-field pseudo-image
    r_adf_outer : int or float
        Outer radius (in pixels) to describe the mask for producing an
        annular dark-field pseudo-image
    save_results : bool
        If True, save 4D STEM data as a Hyperspy HDF5 file and the sum image,
        sum pattern, bright-field image, and annular dark field image as
        PNG files. Default is True.
    return_all : bool
        If True, return all signals in a dictionary. Default is False.
    overwrite : bool
        If True, overwrite any files with the same name in the save path.
        Default is True.
    outpath : str
        Path to which to save the data.  If None, data will be saved to
        the same path as the FPD archive file.


    Returns
    ----------
    out_dict : dict
        Dictionary containing all signals if return_all is True.

    """

    if not fpd_filename:
        root = tk.Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', True)
        startdir = "c:/users/aherzing/data"
        fpd_filename = filedialog.askopenfilename(initialdir=startdir,
                                                  title="Select FPD file...")

    signals = fpd.fpd_file.fpd_to_hyperspy(fpd_filename)

    s = px.ElectronDiffraction2D(signals.fpd_data)
    sum_pattern = signals.fpd_sum_dif
    sum_image = signals.fpd_sum_im

    try:
        signals.DM0
    except AttributeError:
        logger.info('No DM file stored with data')
    else:
        dm = signals.DM0
        logger.info("DM data found in FPD file")
        if dm.axes_manager[0].units.lower() != 'nm':
            pixsize = 1000*dm.axes_manager[0].scale
            logger.info("Changed scale from microns to nanometers")
            pixsize = dm.axes_manager[0].scale

            s.axes_manager[0].scale = pixsize
            s.axes_manager[1].scale = pixsize
            s.axes_manager[0].units = 'nm'
            s.axes_manager[1].units = 'nm'

            sum_image.axes_manager[0].scale = pixsize
            sum_image.axes_manager[1].scale = pixsize
            sum_image.axes_manager[0].units = 'nm'
            sum_image.axes_manager[1].units = 'nm'

    if center:
        logger.info("Centering diffraction patterns")
        s_com = s.center_of_mass(threshold=com_threshold)
        s_com -= 128
        s = s.shift_diffraction(s_com.inav[0].data, s_com.inav[1].data,
                                interpolation_order=shift_interpolation)

    logger.info("Computing virtual bright-field image")
    bf = s.virtual_bright_field(cx=128, cy=128, r=r_bf)
    logger.info("Computing virtual dark-field image")
    adf = s.virtual_annular_dark_field(128, 128, r_inner=r_adf_inner,
                                       r=r_adf_outer)

    if save_results:
        if not outpath:
            outpath = os.path.split(fpd_filename)[0] + '/'
        logger.info("Saving data")
        rootname = os.path.splitext(os.path.split(fpd_filename)[1])[0]
        full_filename_out = outpath + rootname + ".hspy"
        s.save(full_filename_out, overwrite=overwrite)

        sum_pattern_out = sum_pattern.deepcopy()
        sum_pattern_out.data = (255*sum_pattern_out.data /
                                sum_pattern_out.data.max())
        sum_pattern_out.change_dtype('uint8')
        sum_filename_out = outpath + rootname + "_SumPattern.png"
        sum_pattern_out.save(sum_filename_out, overwrite=overwrite)

        sum_image_out = sum_image.deepcopy()
        sum_image_out.data = (255*sum_image_out.data /
                              sum_image_out.data.max())
        sum_image_out.change_dtype('uint8')
        sum_filename_out = outpath + rootname + "_SumImage.png"
        sum_image_out.T.save(sum_filename_out, overwrite=overwrite)

        bf_out = bf.deepcopy()
        bf_out.data = 255*bf_out.data/bf_out.data.max()
        bf_out.change_dtype('uint8')
        bf_filename_out = outpath + rootname + "_BF.png"
        bf_out.save(bf_filename_out, overwrite=overwrite)

        adf_out = adf.deepcopy()
        adf_out.data = 255*adf_out.data/adf_out.data.max()
        adf_out.change_dtype('uint8')
        adf_filename_out = outpath + rootname + "_ADF.png"
        adf_out.save(adf_filename_out, overwrite=overwrite)
    logger.info("Processing complete")

    if return_all:
        out_dict = {'s': s,
                    'sum_pattern': sum_pattern,
                    'sum_image': sum_image,
                    'bf': bf,
                    'adf': adf}
        return out_dict
    else:
        return


def merlin_to_fpd(rootpath=None, savepath=None, keep_unshaped=False,
                  shutdown=False, discard_first_column=False,
                  discard_data=False):
    """
    Convert Merlin .MIB/.HDR files to FPD files and transfer to a storage
    location.

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
        if len(glob.glob(i+"/*.mib")) == 0:
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

        logger.info("Merlin Data File: %s" % mibfiles[0])
        logger.info("Merlin Header File: %s" % hdrfile)
        logger.info("Saving to path: %s" % outpaths[i])

        if len(dmfile) > 0:
            logger.info("Found DM file: %s" % dmfile[0])
            save_dm = True
        else:
            save_dm = False

        scanX, scanY, skip_frames, total_frames = get_scan_shape(mibfiles)

        if keep_unshaped:
            unshapedfilename = tempfilenames[i][:-5] + "_Unshaped.hdf5"
            unshaped = fpd.fpd_file.MerlinBinary(binfns=mibfiles,
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
            s = fpd.fpd_file.MerlinBinary(binfns=mibfiles,
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
            s = fpd.fpd_file.MerlinBinary(binfns=mibfiles,
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
