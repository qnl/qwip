import itertools as it
from typing import TYPE_CHECKING, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, ListedColormap, LogNorm
from matplotlib.figure import Figure
from pandas import DataFrame, Series

from qwip.processing.classification import GMMData
from qwip.visualization.utils import TColor, get_berkeley_colormap, get_colormap

if TYPE_CHECKING:
    from qwip.processing.processors import GMMClassification, IQResult


def plot_readout_IQ(
    data: "IQResult",
    /,
    ax: Axes | None = None,
    groupby: str | None = None,
    cmap: Colormap | TColor = "Greys",
    logscale: bool = True,
    bins: int = 30,
    title: str = None,
) -> Figure:
    """Plots an IQResult as a 2d histogram.

    Args:
        data: The IQResult.
        ax: The `Axes` on which to plot the data.
        groupby: A string specifying whether to group the data. This is passed to the
            pandas `groupby` function.
        cmap: A colormap specifier. See `qwip.visualization.utils.get_colormap` for more
            details.
        bins: The number of bins to use in the 2d histogram.
        title: The subplot title.

    Returns:
        The matplotlib `Figure` that the subplot belongs to.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.grid(False)
    ax.set_aspect("equal")
    norm = LogNorm() if logscale else None
    dropout = {} if logscale else dict(threshold=0.02, smoothing=0.04)
    cmap = get_colormap(cmap, dropout=dropout)

    if title:
        ax.set_title(title)

    if groupby is None:
        IQ = data.data.to_numpy().flatten()
        counts, _, _, im = ax.hist2d(
            IQ.real, IQ.imag, norm=norm, density=True, cmap=cmap, bins=bins
        )

    else:
        for i, (label, subset) in enumerate(data.data.groupby(groupby)):
            IQ = subset.to_numpy().flatten()
            cmap = get_colormap(f"C{i}", dropout=dropout)
            counts, _, _, im = ax.hist2d(
                IQ.real, IQ.imag, norm=norm, density=True, cmap=cmap, bins=bins
            )

    ax.axis("square")

    return fig


def plot_GMM(
    gmm: "GMMClassification",
    ax: Axes | None = None,
    mesh: int = 200,
    legend: bool = True,
    means_kw: dict = dict(marker="*", mec="k", mew=0.3),
    contour_kw: dict = dict(linewidths=1, colors="k"),
) -> Figure:
    """Plots GMM means and decision boundaries.

    Args:
        gmm: The GMM processor that holds the GMM data.
        ax: The `Axes` object on which to plot the data. If none is supplied, a new
            figure is created.
        mesh: The number of points to use in the mesh for plotting the decision
            boundary.
        legend: If true, adds a legend to the plot.
        means_kw: Keyword arguments are passed to `Axes.plot` when plotting the means.
        contour_kw: Keyword arguments are passed to `Axes.contour` when plotting the
            decision boundary.

    Returns:
        The matplotlib `Figure` that the subplot belongs to.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    for i, m in enumerate(gmm.means):
        ax.plot(*m, color=f"C{i}", label=f"{i}", **means_kw)

    xs, ys = np.meshgrid(
        np.linsapce(*ax.get_xlim(), mesh), np.linspace(*ax.get_ylim(), mesh)
    )
    pts = np.stack([xs.flatten(), ys.flatten()]).T
    classified = gmm.get_model().predict(pts).reshape(xs.shape)

    pairs = it.combinations(range(gmm.num_states), 2)
    boundaries = sorted(np.mean(pair) for pair in pairs)

    ax.contour(xs, ys, classified, boundaries, **contour_kw)

    if legend:
        ax.legend()

    return fig


def plot_decision_boundary(
    fig: Figure,
    gmm_data: dict[str, GMMData],
    *,
    colors: Union[str, list] = get_berkeley_colormap().colors,
    alpha: float = 0.2,
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

    for ax, gmm_data in zip(fig.axes, gmm_data.values()):
        gmm = gmm_data.gmm_model()

        N = gmm.n_components
        if isinstance(colors, str):
            cmap = mpl.cm.get_cmap(colors)
            cmap = ListedColormap(
                [cmap.colors[i] for i in range(N)], name="readout", N=N
            )
        else:
            cmap = ListedColormap(colors[:N], name="readout", N=N)

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        nx, ny = (200, 200)

        xs, ys = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny))

        classified = gmm.predict(np.c_[xs.reshape(-1), ys.reshape(-1)]).reshape(
            xs.shape
        )

        for state in range(N):
            ax.contour(
                xs,
                ys,
                classified,
                [0.5 * (state + (state + 1) % N)],  # mean of each pair
                linewidths=2,
                alpha=0.3,
                colors="k",
            )

        im = ax.pcolormesh(
            xs, ys, classified, cmap=cmap, shading="nearest", alpha=alpha, zorder=-1
        )
        fig.colorbar(
            im, ax=ax, label="State", ticks=np.arange(N), pad=0.04, fraction=0.046
        )

        ax.grid(True)
        fig.tight_layout()
    return fig
