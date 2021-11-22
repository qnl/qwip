from typing import Union, Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.colors import Colormap, ListedColormap, LogNorm
from matplotlib.figure import Figure

from qwip.calibration.readout import ReadoutCalibration
from qwip.visualization.utils import (
    get_alpha_colormap,
    get_colormap_with_dropout,
    get_berkeley_colormap
)


def plot_readout_histogram(
    data: dict[str, np.ndarray],
    *,
    cmap: Union[str, Colormap] = 'Greys',
    seq_axis: Optional[int] = None,
    share_axis: bool = True,
    auto_axis: bool = True,
    logscale : bool = True,
    alpha: float = 0.8
) -> Figure:
    """Plots readout histograms from IQ data.

    This plotting utility can be configured

    Args:
        data (dict): A mapping with IQ data.
        cmap (str|Colormap): A colormap or a reference to a registered colormap.
        seq_axis (int|None): The data axis that corresponds to the sequence
            element dimension. If None, the all IQ data is plotted with a single
            colormap. Otherwise, if an axis is specified, IQ data from each
            element will be shown with a different color.
        share_axis (bool): Whether the x and y axes should be shared among the
            different subplots.
        logscale (bool): Whether the colorscale should be log normalized.
        alpha (float): The transparency level of the color scale.

    Returns:
        (Figure): A matplotlib Figure.
    """
    fig, axes = plt.subplots(1, len(data), figsize=(4*3, 3), sharex=share_axis, sharey=share_axis)
    
    # histogram colormap
    cmap = get_colormap_with_dropout(cmap, alpha=alpha)

    limits = 0
    for ax, (key, IQ) in zip(axes, data.items()):
        ax.set_title(f'{key}')

        if share_axis:
            limits = max(limits, np.max(np.abs(IQ)) * 1.1)
        else:
            limits = np.max(np.abs(IQ)) * 1.1

        extent = limits*np.array([-1, 1, -1, 1]) if auto_axis else None

        if seq_axis is None:
            ax.hexbin(
                *IQ.reshape(-1, 2).T,
                cmap=cmap,
                extent=extent,
                norm=LogNorm() if logscale else None,
            )
        else:
            _plot_readout_by_element(
                ax, IQ, seq_axis, extent, logscale, alpha
            )

        ax.set_aspect('equal')

    fig.tight_layout()

    return fig

def _plot_readout_by_element(
    ax,
    IQ: np.ndarray,
    seq_axis: int,
    extent: list,
    logscale: bool,
    alpha: float
) -> None:
    n_elements = IQ.shape[seq_axis]

    for i in range(n_elements):
        idx = tuple(slice(None, None, None) if dim != seq_axis else i for dim in range(len(IQ.shape)))
        ax.hexbin(
            *IQ[idx].reshape(-1, 2).T,
            cmap=get_colormap_with_dropout(get_alpha_colormap(f'C{i}', upper=alpha)),
            extent=extent,
            norm=LogNorm() if logscale else None
        )

def plot_decision_boundary(
    fig: Figure,
    gmm_data: dict[str, dict[str, np.ndarray]],
    *,
    colors: Union[str, list] = get_berkeley_colormap().colors,
    alpha: float = 0.3,
) -> Figure:
    """Plots GMM decision boundaries on an existing figure.

    Args:
        fig (Figure): The figure on which to draw the decision boundaries.
        gmm_data (dict): The GMM means and covariances for each subplot.
        colors (str|list): A string reference to a registered colormap or a list
            of colors. The first N colors will be used to denote the regions,
            where N is the number of qubit states.
        alpha (float): The transparency level of the shaded regions.

    Returns:
        (Figure): A matplotlib Figure.
    """

    for ax, params in zip(fig.axes, gmm_data.values()):
        gmm = ReadoutCalibration.get_gmm_model(params['means'], params['covariances'])

        N = gmm.n_components
        if isinstance(colors, str):
            cmap = mpl.cm.get_cmap(colors)
            cmap = ListedColormap([cmap.colors[i] for i in range(N)], name='readout', N=N)
        else:
            cmap = ListedColormap(colors[:N], name='readout', N=N)

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        nx, ny = (200, 200)

        xs, ys = np.meshgrid(np.linspace(xmin, xmax, nx),
                             np.linspace(ymin, ymax, ny))

        classified = gmm.predict(np.c_[xs.reshape(-1), ys.reshape(-1)]).reshape(xs.shape)

        for state in range(N):
            ax.contour(xs, ys, classified, 
                       [0.5*(state + (state + 1)%N)], # mean of each pair
                       linewidths=2,
                       alpha=0.3,
                       colors='k')

        im = ax.pcolormesh(
            xs, ys, classified, cmap=cmap, shading='nearest', alpha=alpha, zorder=-1
        )
        fig.colorbar(im, ax=ax, label='State', ticks=np.arange(N), pad=0.04, fraction=0.046)

        ax.grid(True)
        fig.tight_layout()
    return fig