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

# def get_with_dmfile(datapath, verbose=False):
#     merlin_path, dmfile, hdrfile, mibfiles = get_files(datapath)
#     mibfiles = sort_mibs(mibfiles)

#     dm = hs.load(dmfile)
#     dm_rows, dm_cols = dm.data.shape
    
#     nfiles = dm_cols * 3

#     frame_times = np.zeros(nfiles)
#     for i in range(0,nfiles):
#         frame_times[i] = get_acquisition_time(mibfiles[i])
        
#     skip_frames = np.argmax(frame_times[0:np.int32(dm_cols/2)]) + 1
#     extra_frames = len(mibfiles) - dm_cols*dm_rows - skip_frames
    
#     s = MerlinBinary(binfns=mibfiles,
#                      hdrfn=hdrfile,
#                      dmfns = [dmfile,],
#                      ds_start_skip=skip_frames,
#                      row_end_skip=0)
#     if verbose:
#         print('\nRoot path: %s' % datapath)
#         print('DM File: %s' % dmfile)
#         print('Merlin path: %s' % merlin_path)
#         print('Number of .MIB files: %s' % len(mibfiles))
#         print('Header file: %s' % hdrfile)
#         print('Extra frames at beginning of scan: %s' % skip_frames)
#         print('Extra frames at end of scan: %s' % extra_frames)
#         print('Resulting data shape: %s' % str(s.shape))
    
#     return s

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
    
def get_merlin_data(merlin_datapath, dmfilename):
    dm = hs.load(dmfilename)
    nframes = dm.data.ravel().shape[0]
    scanX, scanY = dm.data.shape
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
    # extra_frames = len(exposures[skip_frames:]) - nframes
    data = np.zeros([nframes,256**2], data_type)
    for i in range(0, nframes):
        h = open(mibfiles[i+skip_frames], 'rb')
        data[i,:] = np.fromfile(h, dtype=data_type, offset=384)
        h.close()
    data = data.reshape([scanX, scanY, 256, 256])
    data = pxm.ElectronDiffraction2D(data)
    return data