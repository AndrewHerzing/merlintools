import os
import glob
import logging
import fpd
from merlintools.io import get_scan_shape
import tkinter as tk
from tkinter import filedialog
import time
import shutil

try:
    optional_package = None
    import pyxem as optional_package
except ImportError:
    pass

if optional_package:
    import pyxem as px


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def preprocess(datapath="./", mibfile=None, dmfile=None, com_threshold=3,
               shift_interpolation=0, r_bf=15, r_adf_inner=20,
               r_adf_outer=65, save_results=True, return_all=False,
               overwrite=False, outpath=None):
    if not mibfile:
        mibfile = glob.glob(datapath + "*.mib")[0]
    if not dmfile:
        dmfile = glob.glob(datapath + "*.dm3")[0]
    s = px.load_mib(mibfile)
    logger.info(".mib file loaded")
    dm = px.load(dmfile)
    logger.info(".dm3 file loaded")
    if dm.axes_manager[0].units.lower() != 'nm':
        pixsize = 1000*dm.axes_manager[0].scale
        logger.info("Changed scale from microns to nanometers")
    else:
        pixsize = dm.axes_manager[0].scale
    s.axes_manager[0].scale = pixsize
    s.axes_manager[1].scale = pixsize
    s.axes_manager[0].units = 'nm'
    s.axes_manager[1].units = 'nm'

    logger.info("Centering diffraction patterns")
    s_com = s.center_of_mass(threshold=com_threshold)
    s_com -= 128
    s = s.shift_diffraction(s_com.inav[0].data, s_com.inav[1].data,
                            interpolation_order=shift_interpolation)

    logger.info("Computing sum pattern")
    sum_pattern = s.sum((0, 1))
    sum_pattern.compute()

    logger.info("Computing virtual bright-field image")
    bf = s.virtual_bright_field(cx=128, cy=128, r=r_bf)
    logger.info("Computing virtual dark-field image")
    adf = s.virtual_annular_dark_field(128, 128, r_inner=r_adf_inner,
                                       r=r_adf_outer)

    if save_results:
        if not outpath:
            outpath = datapath
        logger.info("Saving data")
        rootname = os.path.splitext(os.path.split(mibfile)[1])[0]
        full_filename_out = outpath + rootname + ".hspy"
        s.save(full_filename_out, overwrite=overwrite)

        sum_pattern_out = sum_pattern.deepcopy()
        sum_pattern_out.data = (255*sum_pattern_out.data /
                                sum_pattern_out.data.max())
        sum_pattern_out.change_dtype('uint8')
        sum_filename_out = outpath + rootname + "_SumPattern.hspy"
        sum_pattern.save(sum_filename_out,
                         overwrite=overwrite)
        sum_filename_out = outpath + rootname + "_SumPattern.png"
        sum_pattern_out.save(sum_filename_out, overwrite=overwrite)

        bf_out = bf.deepcopy()
        bf_out.data = 255*bf_out.data/bf_out.data.max()
        bf_out.change_dtype('uint8')
        bf_filename_out = outpath + rootname + "_BF.hspy"
        bf.save(bf_filename_out, overwrite=overwrite)
        bf_filename_out = outpath + rootname + "_BF.png"
        bf_out.save(bf_filename_out, overwrite=overwrite)

        adf_out = adf.deepcopy()
        adf_out.data = 255*adf_out.data/adf_out.data.max()
        adf_out.change_dtype('uint8')
        adf_filename_out = outpath + rootname + "_ADF.hspy"
        adf_out.save(adf_filename_out, overwrite=overwrite)
        adf_filename_out = outpath + rootname + "_ADF.png"
        adf_out.save(adf_filename_out, overwrite=overwrite)
    logger.info("Processing complete")
    if return_all:
        return s, sum_pattern, bf, adf
    else:
        return


def preprocess_merlin_data(datapath, savepath=None):
    mibfiles = glob.glob(datapath + "*.mib")
    hdrfile = glob.glob(datapath + "*.hdr")[0]
    dmfile = glob.glob(datapath + "*.dm3")[0]
    logger.info("Merlin Data File: %s" % mibfiles[0])
    logger.info("Merlin Header File: %s" % hdrfile)
    logger.info("DM File: %s" % dmfile)

    if savepath is None:
        outpath = ".\\" + os.path.split(mibfiles[0])[1][:-4] + "\\"
    else:
        outpath = savepath

    if not os.path.isdir(outpath):
        os.mkdir(outpath)
    h5filename = outpath + \
        os.path.splitext(os.path.split(mibfiles[0])[1])[0] + '.hdf5'

    s = fpd.fpd_file.MerlinBinary(binfns=mibfiles,
                                  hdrfn=hdrfile,
                                  dmfns=[dmfile, ],
                                  ds_start_skip=0,
                                  row_end_skip=0,
                                  sort_binary_file_list=False,
                                  strict=False,
                                  repack=True)

    s.write_hdf5(h5filename, allow_memmap=False, ow=True)
    del s

    signals = fpd.fpd_file.fpd_to_hyperspy(h5filename,
                                           group_names=['Exposure',
                                                        'fpd_data',
                                                        'fpd_sum_im',
                                                        'fpd_sum_diff'])

    exposure_image = signals.Exposure
    exposure_image = exposure_image.as_signal2D((0, 1))

    sum_image = signals.fpd_sum_im
    sum_image = sum_image.as_signal2D((0, 1))

    sum_diff = signals.fpd_sum_diff
    sum_diff = sum_diff.as_signal2D((0, 1))

    rootname = os.path.splitext(h5filename)[0]
    exposure_image.save(rootname + '_Exposures.hspy', overwrite=True)
    exposure_image.save(rootname + '_Exposures.tiff', overwrite=True)

    sum_image.save(rootname + '_SumImage.hspy', overwrite=True)
    sum_image.save(rootname + '_SumImage.tiff', overwrite=True)

    sum_diff.save(rootname + '_SumDiff.hspy', overwrite=True)
    sum_diff.save(rootname + '_SumDiff.tiff', overwrite=True)
    return


def merlin_to_fpd(rootpath=None, savepath=None, keep_raw=False, shutdown=False,
                  discard_first_column=False, discard_data=False):
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

    temp_dir = os.path.dirname(os.path.dirname(rootpath)) +\
        time.strftime("/%Y%m%d_%H%M%S_FPD_EXPORT/")
    if savepath[-1:] != "/":
        savepath = savepath + "/"

    dirs = [x[0] for x in os.walk(rootpath)]
    empty_dirs = []
    for i in dirs:
        if len(glob.glob(i+"/*.mib")) == 0:
            empty_dirs.append(i)
    [dirs.remove(i) for i in empty_dirs]
    outpaths = [None] * len(dirs)
    h5filenames = [None] * len(dirs)

    for i in range(0, len(dirs)):
        mibfiles = glob.glob(dirs[i] + "/*.mib")
        mibfiles = [filepath.replace('\\', '/') for filepath in mibfiles]
        outpaths[i] = temp_dir + '/' + mibfiles[0].split('/')[-3] +\
            '/' + mibfiles[0].split('/')[-2] + '/'

        if not os.path.isdir(outpaths[i]):
            os.makedirs(outpaths[i])
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

        if keep_raw:
            rawfilename = h5filenames[i][:-5] + "_Raw.hdf5"
            raw = fpd.fpd_file.MerlinBinary(binfns=mibfiles,
                                            hdrfn=hdrfile,
                                            ds_start_skip=0,
                                            row_end_skip=0,
                                            sort_binary_file_list=False,
                                            strict=False,
                                            repack=True)

            logger.info("Saving unshaped data to file: %s" % rawfilename)
            raw.write_hdf5(rawfilename, ow=True, allow_memmap=False)
            del raw

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
        s.write_hdf5(h5filenames[i], ow=True, allow_memmap=False)
        del s

    savepath = savepath + time.strftime("%Y%m%d_%H%M%S/")
    shutil.copytree(temp_dir, savepath)
    if discard_data:
        shutil.rmtree(rootpath)
    if shutdown:
        logger.info("Processing complete. Shutting down computer.")
        os.system("shutdown /s /t 1")
    else:
        logger.info("Processing complete.")
    return h5filenames
