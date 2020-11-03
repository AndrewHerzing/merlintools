import re
import numpy as np
import hyperspy.api as hs
from fpd.fpd_file import MerlinBinary
import glob
import pyxem as pxm
import tqdm
import os

def sort_mibs(filename_list):
    key_list = []
    
    for filename in filename_list:
        root_name = filename.split(".")[-2]
        key_list.append(int(re.search(r'\d+',root_name[::-1]).group()[::-1]))

    sorted_list = [filename_list for (key_list, filename_list) in sorted(zip(key_list,filename_list))]
    return sorted_list

def get_exposure_times(mibfiles, n=10):
    # If n is not provided, get number of frames from the filesize and read all exposures
    if n is None:
        n = int(os.path.getsize(mibfiles) / (2*(256**2) + 384))

    if type(mibfiles) is list and len(mibfiles) > 1:
        exposures = np.zeros(n)
        for i in range(0, n):
            with open(mibfiles[i], 'r') as h:
                line = h.readline(200)
            exposures[i] = np.float32(line.split(',')[10])*1000
    else:
        exposures = np.zeros(n)
        with open(mibfiles,'rb') as h:    
            for i in range(0, n):
                hdr_temp = np.fromfile(h, 'int8', 384)
                hdr_temp = ''.join([chr(item) for item in hdr_temp]).split(',')
                exposures[i] = hdr_temp[10]
                _ = np.fromfile(h, np.uint16, 256**2)
    return exposures

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

def parse_mib_header(mibfile):
    mib_hdr = {}
    with open(mibfile, 'r') as h:
        hdr_temp = np.fromfile(h, 'int8', 384)
        hdr_temp = ''.join([chr(item) for item in hdr_temp]).split(',')

    mib_hdr['HeaderID'] = hdr_temp[0]
    mib_hdr['AcquisitionSequenceNumber'] = hdr_temp[1]
    mib_hdr['DataOffset'] = np.int32(hdr_temp[2])
    mib_hdr['NumChips'] = np.int32(hdr_temp[3])
    mib_hdr['PixDimX'] = np.int32(hdr_temp[4])
    mib_hdr['PixDimY'] = np.int32(hdr_temp[5])
    mib_hdr['PixDepth'] = hdr_temp[6]
    mib_hdr['SensorLayout'] = hdr_temp[7][3:]
    mib_hdr['ChipSelect'] = np.int32(hdr_temp[8])
    mib_hdr['TimeStamp'] = hdr_temp[9]
    mib_hdr['ShutterTime'] = np.float32(hdr_temp[10])
    mib_hdr['Counter'] = np.int32(hdr_temp[11])
    mib_hdr['ColourMode'] = np.int32(hdr_temp[12])
    mib_hdr['GainMode'] = np.int32(hdr_temp[13])
    mib_hdr['Thresholds'] = np.zeros(8, np.float32)
    for i in range(0, 8):
        mib_hdr['Thresholds'][i] = np.float32(hdr_temp[14+i])
    mib_hdr['DACs'] = {}
    mib_hdr['DACs']['Format'] = hdr_temp[22]
    mib_hdr['DACs']['Thresh0'] = np.uint16(hdr_temp[23])
    mib_hdr['DACs']['Thresh1'] = np.uint16(hdr_temp[24])
    mib_hdr['DACs']['Thresh2'] = np.uint16(hdr_temp[25])
    mib_hdr['DACs']['Thresh3'] = np.uint16(hdr_temp[26])
    mib_hdr['DACs']['Thresh4'] = np.uint16(hdr_temp[27])
    mib_hdr['DACs']['Thresh5'] = np.uint16(hdr_temp[28])
    mib_hdr['DACs']['Thresh6'] = np.uint16(hdr_temp[29])
    mib_hdr['DACs']['Thresh7'] = np.uint16(hdr_temp[30])
    mib_hdr['DACs']['Preamp'] = np.uint8(hdr_temp[31])
    mib_hdr['DACs']['Ikrum'] = np.uint8(hdr_temp[32])
    mib_hdr['DACs']['Shaper'] = np.uint8(hdr_temp[33])
    mib_hdr['DACs']['Disc'] = np.uint8(hdr_temp[34])
    mib_hdr['DACs']['DiscLS'] = np.uint8(hdr_temp[35])
    mib_hdr['DACs']['ShaperTest'] = np.uint8(hdr_temp[36])
    mib_hdr['DACs']['DACDiscL'] = np.uint8(hdr_temp[37])
    mib_hdr['DACs']['DACTest'] = np.uint8(hdr_temp[38])
    mib_hdr['DACs']['DACDISCH'] = np.uint8(hdr_temp[39])
    mib_hdr['DACs']['Delay'] = np.uint8(hdr_temp[40])
    mib_hdr['DACs']['TPBuffIn'] = np.uint8(hdr_temp[41])
    mib_hdr['DACs']['TPBuffOut'] = np.uint8(hdr_temp[42])
    mib_hdr['DACs']['RPZ'] = np.uint8(hdr_temp[43])
    mib_hdr['DACs']['GND'] = np.uint8(hdr_temp[44])
    mib_hdr['DACs']['TPRef'] = np.uint8(hdr_temp[45])
    mib_hdr['DACs']['FBK'] = np.uint8(hdr_temp[46])
    mib_hdr['DACs']['Cas'] = np.uint8(hdr_temp[47])
    mib_hdr['DACs']['TPRefA'] = np.uint16(hdr_temp[48])
    mib_hdr['DACs']['TPRefB'] = np.uint16(hdr_temp[49])
    mib_hdr['ExtID'] = hdr_temp[50]
    mib_hdr['ExtTimeStamp'] = hdr_temp[51]
    mib_hdr['ExtID'] = np.float64(hdr_temp[52][:-2])
    mib_hdr['ExtCounterDepth'] = np.uint8(hdr_temp[53])
    return mib_hdr

def get_merlin_data(mibfiles, hdrfile, dmfile=None, skip_frames=None, scanX=None, scanY=None,
                    use_fpd=True, scan_calibration=1.0, show_progressbar=True):

    # Use exposure times to determine the number of extra frames at
    # the beginning and end of the dataset.
    if skip_frames is None:
        exposures = get_exposure_times(mibfiles)
        skip_frames = np.argmax(exposures)

    # Determine total number of frames. If a list of .mib files is provided, 
    # the total number of frames is simply the number of files. If a single .mib
    # file is provided, the total number of frames is determined based on the file size.
    if type(mibfiles) is list:
        mibfiles = sort_mibs(mibfiles)
        total_frames = len(mibfiles)
    else:
        total_frames = int(os.path.getsize(mibfiles) / (2*(256**2) + 384))

    # Determine scan parameters.
    if dmfile:
        dm = hs.load(dmfile)
        nframes = dm.data.ravel().shape[0]
        scanX = [dm.data.shape[0], 'x', 'nm']
        scanY = [dm.data.shape[1], 'y', 'nm']
        scan_calibration = dm.axes_manager[0].scale
        if dm.axes_manager[0].units.lower() != 'nm':
            scan_calibration = scan_calibration * 1000
    elif type(scanX) is list:
        nframes = scanX[0] * scanY[0]
    else:
        nframes = int(total_frames - skip_frames)
        scanX = [nframes, 'x', 'pixels']
        scanY = [1, 'y', 'pixels']
    
    extra_frames = total_frames - nframes - skip_frames
    
    if not use_fpd:
        if type(mibfiles) is list:
            print('MIB')
        print('DM file: %s' % dmfile)
        print('Header file: %s' % hdrfile)
        print('Total number of frames: %s' % total_frames)
        print('Extra frames at beginning of scan: %s' % skip_frames)
        print('Extra frames at end of scan: %s' % extra_frames)
        print('Resulting data shape: [%s, %s, %s, %s]' % (scanX[0], scanY[0], 256, 256))

    # Read data using the fpd module
    if use_fpd:
        if dmfile:
            data = MerlinBinary(binfns=mibfiles,
                                hdrfn=hdrfile,
                                dmfns=dmfile,
                                ds_start_skip=skip_frames,
                                row_end_skip=0,
                                sort_binary_file_list=False)
        else:
            data = MerlinBinary(binfns=mibfiles,
                                hdrfn=hdrfile,
                                dmfns = [],
                                ds_start_skip=skip_frames,
                                scanXalu=scanX,
                                scanYalu=scanY,
                                row_end_skip=0,
                                sort_binary_file_list=False)
        data = data.to_array()

    # Read data using NumPy
    else:
        # Parse the header file to determine the counter depth.
        hdr = parse_hdr(hdrfile)
        if hdr['CounterDepth'] == '6' or hdr['CounterDepth']=='1':
            data_type = np.uint8
        elif hdr['CounterDepth'] == '12':
            data_type = np.uint16
        elif hdr['CounterDepth'] == '24':
            data_type = np.uint32

        data = np.zeros([nframes,256**2], data_type)
        if type(mibfiles) is list:
            for i in tqdm.tqdm(range(0, nframes), disable=(not show_progressbar)):
                h = open(mibfiles[i+skip_frames], 'rb')
                data[i,:] = np.fromfile(h, dtype=data_type, offset=384)
                h.close()
        else:
            with open(mibfiles, 'rb') as h:
                for i in tqdm.tqdm(range(0, nframes+skip_frames), disable=(not show_progressbar)):
                    if i < skip_frames:
                        _ = np.fromfile(h, dtype=data_type, count=256**2, offset=384)
                    else:
                        data[i-skip_frames,:] = np.fromfile(h, dtype=data_type, count=256**2, offset=384)

        data = data.reshape([scanY[0], scanX[0], 256, 256])

    # Convert data to a PyXem ElectronDiffractoin2D signal
    data = pxm.ElectronDiffraction2D(data)
    
    # Set the spatial calibration.  If no DM file is provided,
    # the calibration is set to 1.0 nm.
    data.set_scan_calibration(scan_calibration)
    
    # Diffraction calibration is set to 1.0 A^-1
    data.set_diffraction_calibration(1.0)
    return data