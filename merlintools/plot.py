# -*- coding: utf-8 -*-
#
# This file is part of MerlinTools

"""
plot module for MerlinTools package.

@author: Andrew Herzing
"""

import numpy as np
import matplotlib.pylab as plt
from matplotlib import patches


def plot_most_intense_patterns(dataset, im, ndps=10, logscale=True):
    """
    Plot the diffraction patterns at the most intense locations in an image.

    Args
    ----------
    dataset : NumPy array
        4D-STEM dataset.
    im : NumPy array
        Image to use for selection of locations.
    ndps : integer
        Number of patterns to plot.
    logscale : boolean
        If True, plot patterns on a log scale.

    Returns
    ----------
    locs : NumPy array
        Locations of most intense patterns.

    """
    nrows = int(np.ceil(ndps / 3))
    idx = np.ravel(im).argsort()[::-1][0:ndps]
    fig, ax = plt.subplots(nrows, 3, figsize=(9, 3 * nrows))
    n = 0
    locs = np.zeros([ndps, 2], np.int32)
    for i in ax.flat:
        if n < ndps:
            locs[n, :] = np.unravel_index(idx[n], im.shape)
            diff = dataset[locs[n][0], locs[n][1], :, :]
            if logscale:
                diff = np.log(diff + 1)
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

def plot_q_windows(data, q_ranges, log_scale=True, colors=None, alpha=0.5, center_lines=False,
                   legend=False, figsize=(8,6)):
    """
    Display plot of data and q windows.

    Args
    ----------
    data : NumPy array
        4D-STEM dataset
    q_vals : list
        List of tuples defining q range for each window
    log_scale : bool
        If True, display plot on a semi-log scale.
    colors : list
        Colors to use for plot
    alpha : float
        Alpha level for windows
    center_lines : bool
        If True, display dashed lines at center of windows
    legend : bool
        If True, add legend to plot
    figsize : tuple
        Size of figure to display

    Returns
    ----------
    fig : Matplotlib Figure
        Handle for plot
    """
    if colors is None:
        colors = ['blue','green','purple','magenta','cyan']
    if type(q_ranges) is not list:
        q_ranges = [q_ranges]
    
    fig, ax = plt.subplots(1, figsize=figsize)
    if log_scale:
        ax.semilogy(data['radial_profile'][0], data['radial_profile'][1].sum((0,1)),'ro')
    else:
        ax.plot(data['radial_profile'][0], data['radial_profile'][1].sum((0,1)),'ro')     

    ylim = ax.get_ylim()
    
    rois = [None]*len(q_ranges)
    q_mid = [None]*len(q_ranges)
    for i in range(len(q_ranges)):
        q_mid[i] = (q_ranges[i][1] + q_ranges[i][0])/2
        if q_ranges[i][0] > 0.0:
            if data['qcal_units'] == 'A^-1':
                patch_label = r'%.2f $\AA^{-1}$' % q_mid[i]
            elif data['qcal_units'] == 'nm^-1':
                patch_label = r'%.2f $nm^{-1}$' % q_mid[i]
            else:
                patch_label = r'%.2f $pixels^{-1}$' % q_mid[i]
        else:
            patch_label = "BF"            
        rois[i] = patches.Rectangle((q_ranges[i][0],ylim[0]), width=q_ranges[i][1] - q_ranges[i][0],
                                    height=ylim[1]-ylim[0], alpha=alpha, color=colors[i],
                                    label=patch_label)
        ax.add_patch(rois[i])
    if center_lines:
        for i in range(len(q_ranges)):
            if q_ranges[i][0] > 0.0:
                ax.axvline(q_mid[i], color='black', linestyle='--')
    if legend:
        ax.legend(handles=[i for i in rois], loc='best')
    ax.set_ylabel('log Radially Averaged Intensity (counts)')
    if data['qcal_units'] == 'A^-1':
        ax.set_xlabel(r'q ($\AA^{-1}$)')
    elif data['qcal_units'] == 'nm^-1':
        ax.set_xlabel(r'q ($nm^{-1}$)')
    else:
        ax.set_xlabel(r'$pixels^{-1}$')
    
    return fig

def plot_q_images(data, crop=True, labels=None, figsize=None):
    nimages = len(data['images'])
    if labels is None:
        labels = [None]*nimages
        for i in range(0, len(labels)):
            if data['q_vals'][i] == 0.0:
                labels[i] = 'BF'
            else:
                labels[i] = ('q=%.2f A^-1' % data['q_vals'][i])
    if figsize is None:
        figsize = (5*nimages, 5)
    fig,ax = plt.subplots(1, nimages, figsize=figsize)
    for i in range(0, nimages):
         if crop:
            ax[i].imshow(data['images'][i][:-1,1:], cmap='inferno')
         else:
             ax[i].imshow(data['images'][i], cmap='inferno')
         ax[i].set_title(labels[i])
    [i.axis('off') for i in ax.reshape(-1)]
    _ = plt.suptitle('%s ; FOV: %.0f nm' % (data['data']['filename'].split('/')[-2],
                                            float(data['data']['nt'].DM0[2].data[-1] * 1000)))
    plt.tight_layout()
    return fig, ax

def adjust_clim(figure, axis_number, clim):
    figure.axes[axis_number].get_images()[0].set_clim(clim)
    return