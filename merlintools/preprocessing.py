import glob
import pyxem as px
import os
import logging

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
