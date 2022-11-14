#!/usr/bin/env python3
"""Demo integral-field spectroscopy / hyperspectral imaging using matplotlib"""
__author__ = 'fdeugenio'

import warnings

import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.widgets
import matplotlib.cm
import numpy as np
np.random.seed(8614)

from astropy.modeling.models import Gaussian2D
from scipy.ndimage import gaussian_filter1d
from scipy.stats import binned_statistic_2d

def generate_data(n_sample=5000, n_outliers=0.1):
    """Generate mock data and model
    Returns
    -------
    cube : float 3-d array
        The mock data.
    modl : float 3-d array
        The mock data, without Gaussian noise added.
    """

    # make data: correlated + noise
    x = np.random.randn(n_sample)
    y = 1.2 * x + np.random.randn(n_sample) / 2 + 10.
    z = x**2 - y + np.random.randn(n_sample) / 2
    n_outliers = int(n_outliers) if n_outliers>=1 else int(n_outliers * n_sample)
    z_outliers = np.random.choice(np.arange(n_sample), n_outliers, replace=False)
    z[z_outliers] = np.random.normal(10000., 1., size=n_outliers)

    return x, y, z



def main():

    fig = plt.figure(figsize=(15.6, 4))
    fig.canvas.manager.set_window_title('vicube')
    gs = fig.add_gridspec(
        2, 2, height_ratios=(0.5,10), width_ratios=(1, 3))

    ax0 = fig.add_subplot(gs[1, 0])
    ax1 = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[0, 0])

    ax1.yaxis.set_label_position('right')
    ax1.yaxis.tick_right()

    x, y, z = generate_data(n_sample=100000, n_outliers=10)

    stat, xbins, ybins, belong = binned_statistic_2d(
        x, y, z, bins=(30, 30), statistic=np.mean, expand_binnumbers=True)
    dx = np.diff(xbins)[0] 
    dy = np.diff(ybins)[0] 

    #_img_ = ax.hexbin(x, y, C=z, gridsize=20)
    vmin, vmax = np.nanpercentile(stat, (5, 95))
    _img_ = ax0.pcolormesh(xbins, ybins, stat.T, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(_img_, cax=cax, orientation='horizontal')
    cax.xaxis.tick_top()
    cax.xaxis.set_label_position('top')

    hist_z = z[(belong[0]==1) & (belong[1]==1)]
    z_bins = np.linspace(hist_z.min(), hist_z.max(), 50)
    z_bc   = (z_bins[1:]+z_bins[:-1])/2.
    line_data = ax1.step(z_bc, np.histogram(hist_z, bins=z_bins)[0], 'k-', alpha=0.5, where='mid')[0]
    #line_modl = ax1.step(modl[15, 15, :], 'r-', alpha=0.5)[0]
    #high = z>1000
    #ax0.scatter(x[high], y[high], c=z[high], vmin=vmin, vmax=vmax, edgecolor='k')

    selector  = matplotlib.patches.Rectangle(
        (xbins[1], ybins[1]), dx, dy, edgecolor='r', facecolor='none', lw=1.0)
    ax0.add_artist(selector)
    selector.set_visible(True)

    def update_plot(event):
        _x_, _y_ = event.xdata, event.ydata

        selector.set_visible(True)
        selector.set_xy((_x_-dx/2, _y_-dy/2))

        i = np.digitize(_x_, xbins)
        j = np.digitize(_y_, ybins)
        hist_z = z[(belong[0]==i) & (belong[1]==j)]

        if len(hist_z)==0: return
        z_bins = np.linspace(hist_z.min(), hist_z.max(), 50)
        z_bc   = (z_bins[1:]+z_bins[:-1])/2.
        h      = np.histogram(hist_z, bins=z_bins)[0]
        #print('***', np.max(hist_z), np.max(z), '***')
        line_data.set_xdata(z_bc)
        line_data.set_ydata(h)

        #line_modl.set_ydata(modl[i, j, :])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax1.set_xlim(z_bins[0], z_bins[-1])
            ax1.set_ylim(0, np.max(h))

    def hover(event):
        if event.inaxes != ax0:
            selector.set_visible(False)
            fig.canvas.draw_idle()
            return
        update_plot(event)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.show()
    plt.subplots_adjust(wspace=0, hspace=0)



if __name__ == "__main__":
    main()
