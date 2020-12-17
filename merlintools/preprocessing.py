import glob
import pyxem as px
import os
import logging
import numpy as np
import fpd
from merlintools.io import parse_mib_header, get_exposure_times

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


def process_fpd_data(full, exposure_image):
    skip_frames = np.argmax(exposure_image.isig[:, 0].data).compute() + 1
    logger.info("Extra frames at beginning of dataset: %s" % skip_frames)
    orig_shape = full.data.shape
    new_shape = [orig_shape[0]-1, orig_shape[1], orig_shape[2], orig_shape[3]]

    full_crop = full.deepcopy()
    data = full_crop.data.copy()
    data = data.reshape([data.shape[0]*data.shape[1], data.shape[2],
                         data.shape[3]])
    data = data[skip_frames:-(orig_shape[1]-skip_frames), :, :]
    data = data.reshape(new_shape)
    full_crop = full_crop.inav[1:, :-1]
    full_crop.data = data[:, :-1, :, :]

    full_crop.axes_manager[0].offset = 0.0
    full_crop.axes_manager[1].offset = 0.0

    logger.info("New shape: [%s, %s, %s, %s]" %
                tuple([i for i in full_crop.data.shape]))

    orig_shape = exposure_image.data.shape
    new_shape = [orig_shape[0]-1, orig_shape[1]]

    exposure_image_crop = exposure_image.deepcopy()
    data = exposure_image.data.copy()
    data = data.flatten()

    data = data[skip_frames:-(orig_shape[1]-skip_frames)]
    data = data.reshape(new_shape)
    exposure_image_crop = exposure_image_crop.isig[1:, :-1]
    exposure_image_crop.data = data[:, :-1]
    exposure_image_crop.axes_manager[0].offset = 0.0
    exposure_image_crop.axes_manager[1].offset = 0.0

    axes_info = full_crop.axes_manager

    sum_im = full_crop.sum((0, 1)).as_signal2D((0, 1))
    sum_diff = full_crop.sum((2, 3)).as_signal2D((0, 1))

    full_crop = px.LazyElectronDiffraction2D(full_crop.data)
    full_crop.axes_manager = axes_info

    sum_im.compute()
    sum_diff.compute()
    exposure_image_crop.compute()
    return full_crop, exposure_image_crop, sum_im, sum_diff


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
                                  strict=False)

    s.write_hdf5(h5filename, allow_memmap=False, ow=True)
    del s

    signals = fpd.fpd_file.fpd_to_hyperspy(h5filename,
                                           group_names=['Exposure',
                                                        'fpd_data'])

    exp_im = signals[0]
    exp_im = exp_im.as_signal2D((0, 1))

    full = signals[1]
    full_crop, exposure_image_crop, sum_im, sum_diff =\
        process_fpd_data(full, exp_im)

    rootname = os.path.splitext(h5filename)[0]
    full_crop.save(rootname + '_Processed.hspy', overwrite=True)
    exposure_image_crop.save(rootname + '_Exposures.hspy', overwrite=True)
    exposure_image_crop.save(rootname + '_Exposures.tiff', overwrite=True)

    sum_im.save(rootname + '_SumImage.hspy', overwrite=True)
    sum_im.save(rootname + '_SumImage.tiff', overwrite=True)

    sum_diff.save(rootname + '_SumDiff.hspy', overwrite=True)
    sum_diff.save(rootname + '_SumDiff.tiff', overwrite=True)
    return


def get_scan_shape(mibfiles):
    mib_hdr = parse_mib_header(mibfiles[0])
    n_detector_pix = mib_hdr['PixDimX'] * mib_hdr['PixDimY']
    header_length = mib_hdr['DataOffset']
    if mib_hdr['PixDepth'] == 'U8':
        data_length = 1
    elif mib_hdr['PixDepth'] == 'U16':
        data_length = 2
    elif mib_hdr['PixDepth'] == 'U32':
        data_length = 4
    total_frames = int(os.path.getsize(mibfiles[0]) /
                       (data_length*n_detector_pix + header_length))
    logger.info("Total frames: %s" % total_frames)

    exposures = get_exposure_times(mibfiles[0], int(0.1*total_frames))
    skip_frames = np.argmax(exposures[0:10])
    logger.info("Extra frames at beginning: %s" % skip_frames)

    flybacks = np.where(exposures[skip_frames:] >= 1.5 *
                        exposures[skip_frames + 1])[0]
    scan_width = flybacks[1]
    # scan_height = int(flybacks[-2]/scan_width)
    scan_height = int(np.round(total_frames/scan_width))
    if flybacks[1] == 0.5*flybacks[2]:
        logger.info("Scan width based on flyback: %s pixels" % scan_width)
        logger.info("Scan height based on flyback: %s pixels" % scan_height)
        scanXalu = [np.arange(0, scan_width), 'x', 'pixels']
        scanYalu = [np.arange(0, scan_height), 'y', 'pixels']
        logger.info("Extra frames at end: %s" %
                    (total_frames-skip_frames-scan_width*scan_height))
    else:
        logger.error("Unable to determine scan shape")
        scanXalu = [np.arange(0, 1), 'x', 'pixels']
        scanYalu = [np.arange(0, total_frames - skip_frames), 'y', 'pixels']

    return scanXalu, scanYalu, skip_frames, total_frames


def merlin_to_fpd(rootpath, savepath=".\\", keep_raw=False, shutdown=False,
                  discard_first_column=True):
    if savepath[-1:] != "\\":
        savepath = savepath + "\\"
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    dirs = [x[0] for x in os.walk(rootpath)]
    empty_dirs = []
    for i in dirs:
        if len(glob.glob(i+"\\*.mib")) == 0:
            empty_dirs.append(i)
    [dirs.remove(i) for i in empty_dirs]
    outpaths = [None] * len(dirs)
    h5filenames = [None] * len(dirs)

    for i in range(0, len(dirs)):
        mibfiles = glob.glob(dirs[i] + "\\*.mib")
        outpaths[i] = savepath + os.path.split(mibfiles[0])[0][2:] + "\\"

        if not os.path.isdir(outpaths[i]):
            os.makedirs(outpaths[i])
        h5filenames[i] = outpaths[i] + \
            os.path.splitext(os.path.split(mibfiles[0])[1])[0] + '.hdf5'

    for i in range(0, len(dirs)):
        mibfiles = glob.glob(dirs[i] + "\\*.mib")
        hdrfile = glob.glob(dirs[i] + "\\*.hdr")[0]

        logger.info("Merlin Data File: %s" % mibfiles[0])
        logger.info("Merlin Header File: %s" % hdrfile)
        logger.info("Saving to path: %s" % outpaths[i])
        scanX, scanY, skip_frames, total_frames = get_scan_shape(mibfiles)
        if discard_first_column:
            skip_frames += 1
            scanX[0] = scanX[0][:-1]
            end_skip = 1
        else:
            end_skip = 0

        if keep_raw:
            rawfilename = h5filenames[i][:-5] + "_Raw.hdf5"
            raw = fpd.fpd_file.MerlinBinary(binfns=mibfiles,
                                            hdrfn=hdrfile,
                                            ds_start_skip=0,
                                            row_end_skip=0,
                                            sort_binary_file_list=False,
                                            strict=False)

            logger.info("Saving unshaped data to file: %s" % rawfilename)
            raw.write_hdf5(rawfilename, ow=True)
            del raw

        s = fpd.fpd_file.MerlinBinary(binfns=mibfiles,
                                      hdrfn=hdrfile,
                                      ds_start_skip=skip_frames,
                                      row_end_skip=end_skip,
                                      scanXalu=scanX,
                                      scanYalu=scanY,
                                      sort_binary_file_list=False,
                                      strict=False)

        logger.info("Saving to file: %s" % h5filenames[i])
        s.write_hdf5(h5filenames[i], ow=True)
        del s

    if shutdown:
        logger.info("Processing complete. Shutting down computer.")
        os.system("shutdown /s /t 1")
    else:
        logger.info("Processing complete.")
    return
