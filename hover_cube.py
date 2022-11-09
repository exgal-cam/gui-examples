#!/usr/bin/env python3
"""Demo integral-field spectroscopy / hyperspectral imaging using matplotlib"""
__author__ = 'fdeugenio'

import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.widgets
import matplotlib.cm
import numpy as np
np.random.seed(8614)

from astropy.modeling.models import Gaussian2D
from scipy.ndimage import gaussian_filter1d

def generate_data(mean_x, mean_y, std_x, std_y, theta, n_pixels):
    """Generate mock data and model
    Returns
    -------
    cube : float 3-d array
        The mock data.
    modl : float 3-d array
        The mock data, without Gaussian noise added.
    """

    # Generate a realistic galaxy image.
    g2d = Gaussian2D(
        1., mean_x, mean_y, std_x, std_y, theta=theta)
    x, y = [np.linspace(-5., 5., 30) for _ in 'ab']
    xx, yy = np.meshgrid(x, y) 
    img = g2d(xx, yy)

    # Generate a realistic galaxy spectrum X-D
    spec = np.zeros(n_pixels)
    wave = np.arange(spec.size)

    spec = 10./(wave/1000.+1.)**2
    spec[:155] /= 4.
    spec = gaussian_filter1d(spec, 10)
    for w,f,s in zip((250, 1022, 1257, 2200, 2250),
                     (5, -5, 12, 37, 6),
                     (15, 18, 18, 18, 25)):
        spec += f*np.exp(-0.5*((wave-w)/s)**2)

    # Observing the model galaxy with noise.
    modl = img[:, :, None] * spec[None, None, :]
    cube = np.random.normal(
        img[:, :, None]*spec[None, None, :],    # mean
        )
    img  = np.nanmedian(cube, axis=2)

    return cube, modl



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

    cube, modl = generate_data(
        0., 0., 1., 2., np.pi/6., 2500)
    img = np.nanmedian(cube, axis=2)

    _img_ = ax0.imshow(img, origin='lower')
    plt.colorbar(_img_, cax=cax, orientation='horizontal')
    cax.xaxis.tick_top()
    cax.xaxis.set_label_position('top')

    line_data = ax1.step(cube[15, 15, :], 'k-', alpha=0.5)[0]
    line_modl = ax1.step(modl[15, 15, :], 'r-', alpha=0.5)[0]

    selector  = matplotlib.patches.Rectangle(
        (14.5, 14.5), 1, 1, edgecolor='r', facecolor='none', lw=1.0)
    ax0.add_artist(selector)
    selector.set_visible(True)

    def update_plot(event):
        selector.set_visible(True)
        i, j = int(event.xdata), int(event.ydata)
        line_data.set_ydata(cube[i, j, :])
        line_modl.set_ydata(modl[i, j, :])
        selector.set_xy((i-.5, j-.5))

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
