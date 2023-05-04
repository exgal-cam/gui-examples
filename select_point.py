#!/usr/bin/env python3
"""Demo integral-field spectroscopy / hyperspectral imaging using matplotlib"""
__author__ = 'fdeugenio'
"""Inspired from matplotlib documentation here https://matplotlib.org/stable/api/backend_bases_api.html#matplotlib.backend_bases.PickEvent"""

import matplotlib.pyplot as plt 
import numpy as np
from numpy.random import rand

def callback(input_arg0, input_arg1):
    print(f'Doing something with {input_arg0}')

def main():
    fig, ax = plt.subplots()

    # Extremely realistic star-forming main-sequence data.
    mstar = np.logspace(9.3, 12, 500)
    sfrms = np.random.normal(mstar/1e10, mstar/5e9)
    sfr_q = np.random.normal(mstar/1e12, mstar/5e10)
    qmask = (mstar>1e10) & (np.round(np.log10(mstar)*10)%3==0)
    sfr   = np.where(qmask, sfr_q, sfrms)
    ids   = np.random.randint(100000, 1000000, size=mstar.size)
    print(qmask.sum())
    ax.plot(mstar, sfr, 'k.', picker=3)
    ax.loglog()
    ax.set_ylabel('$SFR \; \mathrm{[M_\odot \, yr^{-1}]}$')
    ax.set_xlabel('$M_\star \; \mathrm{[M_\odot]}$')
    # 3, for example, is tolerance for picker i.e, how far a mouse click from
    # the plotted point can be registered to select nearby data point/points.

    def on_pick(event):
        line = event.artist
        xdata, ydata = line.get_data()
        ind = event.ind[0]
        print(f'ID: {ids[ind]:< 8d}, M*={xdata[ind]/1e10:.3f}, SFR={ydata[ind]:.3f}')
        callback(ids[ind], 'Something')

    cid = fig.canvas.mpl_connect('pick_event', on_pick)

    plt.show()

if __name__=="__main__":
    main()


