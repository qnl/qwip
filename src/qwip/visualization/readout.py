import itertools as it
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LogNorm
from matplotlib.figure import Figure

from qwip.utils import deprecated
from qwip.visualization.utils import TColor, get_colormap

if TYPE_CHECKING:
    from qwip.processing.processors import GMMClassification, IQResult


def plot_IQ(
    data: "IQResult",
    /,
    ax: Axes | None = None,
    x_axis: str = "frequency",
    log_mag: bool = True,
    electrical_delay: float = 0,
    unwrap: bool = False,
) -> Figure:
    """Plots an IQResult as a function of a specified index level.

    This is intended for plotting averaged IQ values vs frequency. It will create three
    panels, plotting the amplitude, phase, and a smith chart of the IQ data.

    Args:
        data: The IQResult.
        ax: The `Axes` on which to plot the data.
        x_axis: The index level to use as the x-axis.
        log_mag: Whether or not to plot the amplitude in dB (power). This is passed
            directly to `IQResult.amplitude`.
        electrical_delay: If non-zero, applies an electrical delay to the data before
            plotting. See `IQResult.electrical_delay`.
        unwrap: Whether or not to unwrap the phase. This is passed directly to
            `IQResult.phase`.

    Returns:
        The matplotlib `Figure` that the subplot belongs to.
    """

    grid = [["amplitude", "IQ"], ["phase", "IQ"]]

    if ax is None:
        fig, axes = plt.subplot_mosaic(grid, figsize=(8, 4), layout="constrained")
    else:
        fig = ax.get_figure()
        gridspec = ax.get_subplotspec()
        ax.remove()
        subfig = fig.add_subfigure(gridspec)
        axes = subfig.subplot_mosaic(grid)

    axes["phase"].sharex(axes["amplitude"])
    axes["amplitude"].tick_params(labelbottom=False, length=0)
    axes["IQ"].set_aspect("equal", adjustable="datalim")

    data = data.electrical_delay(electrical_delay, frequency=x_axis)
    combined = pd.concat(
        [
            data.amplitude(log=log_mag),
            data.phase(unwrap=unwrap),
            data.data,
        ],
        axis="columns",
        keys=["amplitude", "phase", "IQ"],
    )

    if x_axis not in combined.index.names:
        raise ValueError(f"'{x_axis}' is not in index levels {data.index.names}")

    if len(combined.index.names) > 1:
        groupby = [idx for idx in data.index.names if idx != x_axis]
        for label, df in combined.groupby(groupby):
            df = df.droplevel(groupby)

            axes["amplitude"].plot(df["amplitude"])
            axes["phase"].plot(df["phase"])
            axes["IQ"].plot(np.real(df["IQ"]), np.imag(df["IQ"]))
    else:
        axes["amplitude"].plot(combined["amplitude"])
        axes["phase"].plot(combined["phase"])
        axes["IQ"].plot(np.real(combined["IQ"]), np.imag(combined["IQ"]))

    axes["amplitude"].set_ylabel("Amplitude" + (" (dB)" if log_mag else ""))
    axes["phase"].set_ylabel("Phase")
    axes["IQ"].set_title("IQ")
    axes["IQ"].set_xlabel("I")
    axes["IQ"].set_ylabel("Q")
    axes["amplitude"].margins(x=0)
    axes["phase"].set_xlabel(x_axis if x_axis.isupper() else x_axis.capitalize())

    return fig


def plot_IQ_histogram(
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


@deprecated(
    version="23.10.0", removed="23.12.0", message="Use `plot_IQ_histogram` instead."
)
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

    !!! Warning
        Deprecated since version 23.10.0. `plot_readout_IQ` will be removed in 23.12.0.

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
    return plot_IQ_histogram(
        data,
        ax=ax,
        groupby=groupby,
        cmap=cmap,
        logscale=logscale,
        bins=bins,
        title=title,
    )


def plot_GMM(
    gmm: "GMMClassification",
    ax: Axes | None = None,
    mesh: int = 200,
    legend: bool = False,
    means_kw: dict = dict(marker="*", mec="k", mew=0.3, ls=" "),
    contour_kw: dict = dict(linewidths=1, colors="k"),
    legend_kw: dict = {},
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
        legend_kw: Keyword arguments are pased to `Axes.legend`.

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
        np.linspace(*ax.get_xlim(), mesh), np.linspace(*ax.get_ylim(), mesh)
    )
    pts = np.stack([xs.flatten(), ys.flatten()]).T
    classified = gmm.get_model().predict(pts).reshape(xs.shape)

    pairs = it.combinations(range(gmm.num_states), 2)
    boundaries = sorted(np.mean(pair) for pair in pairs)

    ax.contour(xs, ys, classified, boundaries, **contour_kw)

    if legend:
        ax.legend(**legend_kw)

    return fig


__all__ = ["plot_IQ", "plot_IQ_histogram", "plot_GMM"]
