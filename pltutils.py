# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


def tickoff(ax=None):
    if ax is not None:
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        try:
            ax.zaxis.set_major_formatter(NullFormatter())
        except:
            pass
    else:
        plt.tick_params(
            axis='both',        # changes apply to the x-axis
            which='both',       # both major and minor ticks are affected
            bottom='off',       # ticks along the bottom edge are off
            top='off',          # ticks along the top edge are off
            labelbottom='off')  # labels along the bottom edge are off


def show3d(data, t, ax, view_init=None, cmap=plt.cm.Spectral, linewidth=0.,
           markersize=10.):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    ax.scatter(x, y, z, c=t, cmap=cmap, lw=linewidth, s=markersize)
    if view_init is not None:
        ax.view_init(*view_init)
    tickoff(ax)
