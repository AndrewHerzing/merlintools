import numpy as np
import matplotlib.pylab as plt


def plot_most_intense_patterns(dataset, im, ndiffs=10):
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

    Returns
    ----------
    fig : Matplotlib Figure
        Figure containing the plotted patterns.

    """
    nrows = int(np.ceil(ndiffs/3))
    idx = np.ravel(im).argsort()[::-1][0:ndiffs]
    fig, ax = plt.subplots(nrows, 3, figsize=(9, 3*nrows))
    n = 0
    for i in ax.flat:
        if n < ndiffs:
            loc = np.unravel_index(idx[n], [100, 99])
            diff = dataset[loc[0], loc[1]+1, :, :]
            i.imshow(np.log(diff+1), vmin=0.5, vmax=3, cmap='inferno')
        i.axis('off')
        n += 1
    plt.tight_layout()
    return fig
