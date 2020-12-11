import glob
import pyxem as px
import os


def preprocess(datapath="./", mibfile=None, dmfile=None, com_threshold=3,
               r_bf=15, r_adf_inner=20, r_adf_outer=65, save_results=True,
               return_all=False):
    if not mibfile:
        mibfile = glob.glob(datapath + "*.mib")[0]
    if not dmfile:
        dmfile = glob.glob(datapath + "*.dm3")[0]

    s = px.load_mib(mibfile)
    dm = px.load(dmfile)
    if dm.axes_manager[0].units.lower() != 'nm':
        pixsize = 1000*dm.axes_manager[0].scale
    s.axes_manager[0].scale = pixsize
    s.axes_manager[1].scale = pixsize
    s.axes_manager[0].units = 'nm'
    s.axes_manager[1].units = 'nm'

    s_com = s.center_of_mass(threshold=com_threshold)
    s_com -= 128
    s = s.shift_diffraction(s_com.inav[0].data, s_com.inav[1].data)

    sum_pattern = s.sum((0, 1))
    sum_pattern.compute()

    bf = s.virtual_bright_field(cx=128, cy=128, r=r_bf)
    adf = s.virtual_annular_dark_field(128, 128,
                                       r_inner=r_adf_inner, r=r_adf_outer)

    if save_results:
        s.save(os.path.splitext(mibfile)[0] + ".hspy")

        sum_pattern.save(os.path.splitext(mibfile)[0] + "_SumPattern.hspy")
        sum_pattern.save(os.path.splitext(mibfile)[0] + "_SumPattern.tiff")

        bf.save(os.path.splitext(mibfile)[0] + "_BF.hspy")
        bf.save(os.path.splitext(mibfile)[0] + "_BF.tiff")

        adf.save(os.path.splitext(mibfile)[0] + "_ADF.hspy")
        adf.save(os.path.splitext(mibfile)[0] + "_ADF.tiff")

    if return_all:
        return s, sum_pattern, bf, adf
    else:
        return
