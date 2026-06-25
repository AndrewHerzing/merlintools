
import numpy as np
import subprocess
import tempfile


#--------------------------------------------------
def writeGSF(filename, data, XReal=1.0, YReal=1.0, Title=None, 
             XYUnits='m', ZUnits=None, open_file=False):
    '''
    Write Gwyddion Simple Field (.gsf) file to disk. For gsf details, see:
    
    http://gwyddion.net/documentation/user-guide-en/gsf.html
    
    Parameters
    ----------
    filename : string or None
        Filename to write file to. '.gsf' is automatically appended if missing.
        If None, uses temporary file. Files should be manually removed.
    data : ndarray of dimensionality 2
        Image to be written.
    XReal : float
        Physical x-size in XYUnits.
    YReal : float
        Physical y-size in XYUnits.
    XYUnits : character
        Lateral base unit, 'm' or 'A'.
    ZUnits : string
        Z base unit.
    Title : string
        Image title.
    open_file : Bool
        Controls if file is opened in Gwyddion.
    
    Returns
    ----------
    filename : string
        Name of file written to disk. 
    
    '''

    if XYUnits is not 'm':
        raise NotImplementedError("Only 'm' XYUnits is implemented")

    if Title is None:
        Title = 'Unknown'
        
    data = data.astype('f4')
    # IEEE 32bit single-precision floating point numbers in 
    # little-endian byte order, by row from top to bottom and 
    # in each row from left to right.
    YRes, XRes = data.shape
    
    magic = 'Gwyddion Simple Field 1.0'
    
    hdr_lines = [magic]
    hdr_lines += ['XRes = %d' %(XRes)]
    hdr_lines += ['YRes = %d' %(YRes)]
    hdr_lines += ['XReal = %e' %(XReal)]
    hdr_lines += ['YReal = %e' %(YReal)]    
    hdr_lines += ['XYUnits = %s' %(XYUnits)]   
    hdr_lines += ['Title = %s' %(Title)]   
    if ZUnits is not None:
        hdr_lines += ['ZUnits = %s' %(ZUnits)]
    
    hdr_str = '\n'.join(hdr_lines)+'\n'
    nul_len = 4 - len(hdr_str)%4
    hdr_str += '\0'*nul_len
    hdr_str = hdr_str.encode('utf-8')
    
    data_str = data.tobytes()
    
    file_data = hdr_str + data_str
    if filename is None:
        f = tempfile.NamedTemporaryFile(suffix='.gsf', delete=False)
        filename = f.name
        f.write(file_data)
    else:
        if filename.endswith('.gsf') is False:
            filename += '.gsf'  
        with open(filename, "bw") as f:
            f.write(file_data)
    
    if open_file:
        subprocess.Popen(['gwyddion', '--no-splash', '--remote-new', filename])
    #print(hdr_str)
    #return hdr_str
    
    return filename


#--------------------------------------------------
def readGSF(filename):
    '''
    Read Gwyddion Simple Field (.gsf) file. For gsf details, see:
    
    http://gwyddion.net/documentation/user-guide-en/gsf.html
    
    Parameters
    ----------
    filename : string or None
        Filename of read to file. 
        '.gsf' is automatically appended if missing.
    
    Returns
    ----------
    data : 32bit float array of dimensionality 2
        Image data
    dsf : dictionary
        Dictionary of strings and floats of all header values
    
    Notes
    -----
    The file format specifies:
    
    Mandatory:
        data : ndarray of dimensionality 2
            Image
        XRes : int
            x size
        YRes : int
            y size
    Optional:
        XReal : float
            Physical x-size in XYUnits
            Default 1.0
        YReal : float
            Physical y-size in XYUnits
            Default 1.0
        XOffset : float
            Horizontal (x) offset in XYUnits
            Default 0.0
        YOffset : float
            Vertical (y) offset in XYUnits
            Default 0.0
        XYUnits : character
            Lateral base unit 'm' or 'A'
            Default 'm'
        ZUnits : string
            Z base unit.
            Default None
        Title : string
            Image title.
            Default None

    '''
      
    hdr = b""
    data = b""
    hd = True
    with open(filename, 'rb') as f:
        #f.read(110)
        # read file byte by byte
        byte = f.read(1)
        while byte != b"":
            #print(byte)
            #while len(hdr)%4!=0
            if hd:
                hdr += byte
                if byte == b'\0':
                    # find 1st null is end of header
                    hd = False
            else:
                # go to int 4 size, if needed
                if len(hdr)%4 != 0:
                    hdr += byte
                else:
                    #read data
                    data += byte
                    data += f.read()
                    break
            byte = f.read(1)
    hdr = hdr.decode('utf-8')
    hdr = hdr.strip('\0').strip().split('\n')
    # convert data to array. TODO: check size
    data = np.fromstring(data, dtype='f4')
    
    # check magic
    assert(hdr.pop(0) == 'Gwyddion Simple Field 1.0')
    
    # parse header
    ds = {k.strip():v.strip() for k, v in (x.split('=') for x in hdr)
          if k.strip().lower() in ['title', 'xyunits', 'zunits']}
    df = {k.strip():float(v.strip()) for k, v in (x.split('=') for x in hdr)
          if k.strip().lower() not in ['title', 'xyunits', 'zunits']}
    
    XRes = int(df['XRes'])
    YRes = int(df['YRes'])
    data.shape = (YRes, XRes)
    
    dsf = ds.copy()
    dsf.update(df)
    
    return data, dsf



##--------------------------------------------------
#if __name__ == '__main__':
    #t = np.random.random((100, 500))
    #sy, sx = t.shape
    #filename = writeGSF(filename=None, data=t, XReal=1.0*sx, 
                        #YReal=1.0*sy, Title=None, XYUnits='m', 
                        #ZUnits=None, open_file=True)
    
    #filename = writeGSF(filename='fn', data=t, XReal=1.0*sx, 
                        #YReal=1.0*sy, Title=None, XYUnits='m', 
                        #ZUnits=None, open_file=False)
    
    ##plt.matshow(t, cmap='gray')
    #readGSF(filename='fn.gsf')

