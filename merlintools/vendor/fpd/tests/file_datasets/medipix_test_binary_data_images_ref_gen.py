

# conversion script to generate reference images

import numpy as np
import os
import h5py
import matplotlib.pylab as plt
plt.ion()

from fpd.fpd_file import binary_to_hdf5 


tags = ['001_12bit_non_raw_16x8',
        '002_12bit_raw_16x8',
        '003_6bit_non_raw_16x8',
        '004_6bit_raw_16x8',
        '005_1bit_non_raw_16x8',
        '006_1bit_raw_16x8']

bp = './medipix_test_binary_data_images/'


# generate hdf5 files
for tag in tags:
    binfns = tag + '.mib' #'.bin' #'.mib'
    hdrfn = tag + '.hdr'
    dmfns = tag + '.dm3'

    binfns = os.path.join(bp, binfns)
    hdrfn = os.path.join(bp, hdrfn)
    dmfns = os.path.join(bp, dmfns)
    
    binary_to_hdf5(binfns, hdrfn, dmfns=[dmfns], h5fn=None, row_end_skip=1, ow=True)


# save diffraction images to file
fpd_sum_dif_ims = []
for tag in tags:
    with h5py.File(os.path.join(bp, tag + '.hdf5')) as h:
        im = h['fpd_expt/fpd_sum_dif/data'][:]
    fpd_sum_dif_ims.append(im)
    plt.matshow(im)
    plt.title(tag)

kwds = dict(zip(tags, fpd_sum_dif_ims))
np.savez_compressed('medipix_test_binary_data_images_fpd_sum_dif_ims.npz', **kwds)


# check result
fpd_sum_dif_ims = np.load('medipix_test_binary_data_images_fpd_sum_dif_ims.npz')
for tag in tags:
    ref_im = fpd_sum_dif_ims[tag]
    #plt.matshow(ref_im)
    with h5py.File(os.path.join(bp, tag + '.hdf5')) as h:
        new_im = h['fpd_expt/fpd_sum_dif/data'][:]
    assert (ref_im == new_im).all()

