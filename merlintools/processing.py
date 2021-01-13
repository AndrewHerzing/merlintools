import hyperspy
import numpy as np
import fpd.fpd_processing as fpdp

def get_radial_profile(ds):
    radial_mean = [None] * ds.shape[0]*ds.shape[1]
    idx = 0
    min_length = np.inf
    r_pix_min = None
    for i in range(0, ds.shape[0]):
        for j in range(0,ds.shape[1]):
            r_pix, radial_mean[idx] = fpdp.radial_profile(ds[i,j,:,:], com_yx[:,i,j], plot=False, spf=1)
            if radial_mean[idx].shape[0] < min_length:
                min_length = radial_mean[idx].shape[0]
                r_pix_min = r_pix
            idx+=1

    result = np.zeros([ds.shape[0]*ds.shape[1], min_length])
    for i in range(0, len(radial_mean)):
        result[i, :] = radial_mean[i][0:min_length]
    result = result.reshape([ds.shape[0], ds.shape[1], min_length])
    s = hs.signals.Signal1D(result)
    return s