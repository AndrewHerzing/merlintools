import re
import numpy as np
import hyperspy.api as hs
from fpd.fpd_file import MerlinBinary
import glob
import pyxem as pxm

def sort_mibs(filename_list):
    key_list = []
    
    for filename in filename_list:
        root_name = filename.split(".")[-2]
        key_list.append(int(re.search(r'\d+',root_name[::-1]).group()[::-1]))

    sorted_list = [filename_list for (key_list, filename_list) in sorted(zip(key_list,filename_list))]
    return sorted_list

def get_acquisition_time(filename):
    with open(filename, 'r') as h:
        line = h.readline(200)
    return np.float32(line.split(',')[10])*1000

def parse_hdr(hdrfile):
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
    
def get_merlin_data(merlin_datapath, dmfilename=None, scanX=None, scanY=None, verbose=False):
    if dmfilename:
        dm = hs.load(dmfilename)
        nframes = dm.data.ravel().shape[0]
        scanX, scanY = dm.data.shape
        scan_calibration = dm.axes_manager[0].scale
        if dm.axes_manager[0].units.lower() != 'nm':
            scan_calibration = scan_calibration * 1000
    else:
        if scanX is None or scanY is None:
            raise ValueError('If no DM file is provided, then scanX and scanY must be defined!')
        nframes = scanX*scanY
    
    hdrfile = glob.glob(merlin_datapath + '*.hdr')[0]
    mibfiles = glob.glob(merlin_datapath + '*.mib')
    mibfiles = sort_mibs(mibfiles)
    hdr = parse_hdr(hdrfile)
    if hdr['CounterDepth'] == '6' or hdr['CounterDepth']=='1':
        data_type = np.uint8
    elif hdr['CounterDepth'] == '12':
        data_type = np.uint16
    elif hdr['CounterDepth'] == '24':
        data_type = np.uint32
    exposures = np.zeros(10)
    for i in range(0, 10):
        exposures[i] = get_acquisition_time(mibfiles[i])
    skip_frames = np.argmax(exposures) + 1
    extra_frames = len(exposures[skip_frames:]) - nframes
    data = np.zeros([nframes,256**2], data_type)
    for i in range(0, nframes):
        h = open(mibfiles[i+skip_frames], 'rb')
        data[i,:] = np.fromfile(h, dtype=data_type, offset=384)
        h.close()
    data = data.reshape([scanX, scanY, 256, 256])
    data = pxm.ElectronDiffraction2D(data)
    
    if dmfilename:
        data.set_scan_calibration(scan_calibration)

    if verbose:
        print('\nMerlin path: %s' % merlin_datapath)
        print('DM File: %s' % dmfilename)
        print('Number of .MIB files: %s' % len(mibfiles))
        print('Header file: %s' % hdrfile)
        print('Extra frames at beginning of scan: %s' % skip_frames)
        print('Extra frames at end of scan: %s' % extra_frames)
        print('Resulting data shape: %s' % str(data.data.shape))
    return data