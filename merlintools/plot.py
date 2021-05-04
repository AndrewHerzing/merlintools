import numpy as np
import matplotlib.pylab as plt


def plot_most_intense_patterns(dataset, im, ndiffs=10, logscale=True):
    """
    Plot the diffraction patterns at the ndiffs most intense locations
    in an image.

    Args
    ----------
    dataset : NumPy array
        4D-STEM dataset.
    im : NumPy array
        Image to use for selection of locations.
    ndiffs : integer
        Number of patterns to plot.
    logscale : boolean
        If True, plot patterns on a log scale.

    Returns
    ----------
    locs : NumPy array
        Locations of most intense patterns.

    """
    nrows = int(np.ceil(ndiffs/3))
    idx = np.ravel(im).argsort()[::-1][0:ndiffs]
    fig, ax = plt.subplots(nrows, 3, figsize=(9, 3*nrows))
    n = 0
    locs = np.zeros([ndiffs, 2], np.int32)
    for i in ax.flat:
        if n < ndiffs:
            locs[n, :] = np.unravel_index(idx[n], im.shape)
            diff = dataset[locs[n][0], locs[n][1], :, :]
            if logscale:
                diff = np.log(diff+1)
                vmax = diff.max()
                vmin = diff.min()
                i.imshow(diff, vmin=vmin, vmax=vmax, cmap='inferno')
            else:
                vmax = diff.max()
                vmin = diff.min()
                i.imshow(diff, vmin=vmin, vmax=vmax, cmap='inferno')
        i.axis('off')
        n += 1
    plt.tight_layout()
    return locs
