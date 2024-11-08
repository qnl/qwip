"""Visualization utilities.

Useful helper functions for working with matplotlib.
"""

import itertools as it
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable
from functools import wraps
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import (
    Colormap,
    LinearSegmentedColormap,
    ListedColormap,
    to_hex,
    to_rgb,
)
from matplotlib.figure import Figure, FigureBase

TColor = str | tuple[float, float, float] | tuple[float, float, float, float]


## Grid


def find_closest_factors(n: int, /) -> tuple[int, int]:
    """Given an integer n, finds the factors of n that are closest to sqrt(n)

    Args:
        n: The integer to factor.

    Returns:
        `a, b` such that `a <= b` and `a * b == n`.
    """
    if not n > 0:
        raise ValueError(f"n must be greater than 0, got {n}")

    trial_factors = 1 + np.arange(np.floor(np.sqrt(n)))

    remainder, factors = np.modf(n / trial_factors)

    As = trial_factors[remainder == 0].astype(int)
    Bs = factors[remainder == 0].astype(int)

    return As[-1], Bs[-1]


def get_grid_size(
    N: int,
    nrows: int | None = None,
    ncols: int | None = None,
    ratio: float | None = 6.0,
) -> tuple[int, int]:
    """Determines a grid size for N subplots.

    Args:
        N: The number of subplots requested.
        nrows: If not `None`, specifies the total number of rows.
        ncols: If not `None`, specifies the total number of columns.
        ratio: The maximum allowed value for `ncols / nrows` when both are `None`. This
            is used so that large prime values of `N` still yield reasonable grid
            configurations.

    Returns:
        A tuple `(nrows, ncols)`.
    """
    if nrows is None and ncols is None:
        nrows, ncols = find_closest_factors(N)
        if ratio and ncols / nrows > ratio:
            ncols = np.ceil(np.sqrt(N)).astype(int)
            nrows = np.ceil(N / ncols).astype(int)
    elif nrows is None:
        nrows = np.ceil(N / ncols).astype(int)
    elif ncols is None:
        ncols = np.ceil(N / nrows).astype(int)
    elif nrows * ncols < N:
        raise ValueError(
            f"Not enough axes in the specified grid. nrows * ncols must be greater than"
            f" N. Got {nrows} * {ncols} = {nrows * ncols} < {N}."
        )

    return nrows, ncols


def get_axes_by_position(
    axes: Iterable[Axes],
    axis: Literal["x", "y"],
    selection: Callable[[Iterable[float]], float] = min,
) -> list[Axes]:
    """Determines the subset of axes that meet the given selection criteria.

    This function first groups all the axes by either `x0` or `y0` of the axes bounding
    box. It will then use the selection function to determine which position should
    be returned.

    Args:
        axes: An iterable of `Axes` to select from.
        axis: Whether to select based on the x or y coordinate.
        selection: A selection function used to determine which axes to return.

    Returns:
        A list of `Axes` that match the given selection criteria.
    """

    if (axis := axis.lower()) not in "xy":
        raise ValueError(f"`axis` must be either 'x' or 'y', got '{axis}'.")

    groups = defaultdict(list)

    for ax in axes:
        bbox = ax.get_position()
        key = bbox.x0 if axis == "x" else bbox.y0
        groups[key].append(ax)

    return groups[selection(groups)]


def make_dict_grid(
    keys: Collection[str] | dict[str, Any],
    nrows: int | None = None,
    ncols: int | None = None,
    ratio: float | None = 6.0,
    layout: str = "constrained",
    **kwargs: Any,
) -> tuple[Figure, dict]:
    """Makes a figure subplot mosaic with subplots labeled by keys.

    Args:
        keys: The labels for the figure subplots.
        nrows: If not `None`, specifies the total number of rows.
        ncols: If not `None`, specifies the total number of columns.
        ratio: The maximum allowed value for `ncols / nrows` when both are `None`.
        layout: Specifies the figure layout. See `matplotlib.pyplot.subplot_mosaic` for
            more details.
        **kwargs: Additional keyword arguments are passed to
            `matplotlib.pyplot.subplot_mosaic`.

    Returns:
        The created figure and subplot dictionary.
    """
    nrows, ncols = get_grid_size(len(keys), nrows, ncols, ratio)

    grid = []
    index = it.product(range(nrows), range(ncols))

    for (row, col), key in it.zip_longest(index, keys, fillvalue="."):
        if col == 0:
            grid.append([])

        grid[row].append(key)

    fig, axes = plt.subplot_mosaic(grid, layout=layout, **kwargs)

    return fig, axes


def make_list_grid(
    N: int,
    nrows: int | None = None,
    ncols: int | None = None,
    ratio: float | None = 6.0,
    hide_unused: bool = True,
    layout: str = "constrained",
    **kwargs: Any,
) -> tuple[Figure, np.ndarray]:
    """Makes a figure subplot grid with at least N subplots.

    Args:
        N: The number of subplots requested.
        nrows: If not `None`, specifies the total number of rows.
        ncols: If not `None`, specifies the total number of columns.
        hide_unused: If `True`, will hide all extra subplots. The N "active" subplots
            start at the top left corner and go down the grid in row-major order.
        layout: Specifies the figure layout. See `matplotlib.pyplot.subplots` for more
            details.
        **kwargs: Additional keyword arguments are passed to
            `matplotlib.pyplot.subplots`.

    Returns:
        A numpy array of empty Axes objects.
    """
    nrows, ncols = get_grid_size(N, nrows, ncols, ratio)

    fig, axes = plt.subplots(nrows, ncols, layout=layout, **kwargs)

    if nrows == ncols == 1:
        axes = np.array([axes])

    if N < nrows * ncols and hide_unused:
        for ax in axes.flat[N:]:
            ax.axis("off")

    return fig, axes


def grid_plotter(
    dataset: dict[str, Any],
    plotter: Callable | None = None,
    fig: dict[str, Figure] | None = None,
    subplot_kwargs: dict = {},
    grid_kwargs: dict = {},
    sort: bool = True,
) -> Figure:
    """Plots a dataset on a grid.

    Given a dataset mapping of keys to data, plots the data on a figure subplot using
    the specified plotting function.

    Args:
        dataset: A dictionary mapping keys to data. The values can be MeasurementResult
            instances, pandas dataframes, or numpy arrays, as long as the axes plotter
            can handle the data type.
        plotter: A plotting function that plots the data on a given `Axes` or `SubFigure`.
        subplot_kwargs: A dictionary of parameters that are passed to `plotter`
        grid_kwargs: A dictionary of parameters that are passed to `make_dict_grid`.
        sort: If `True`, the dataset keys will be sorted first.

    Returns:
        A matplotlib `Figure` with the resulting subplots.
    """
    N = len(dataset)

    if sort:
        dataset = dict(sorted(dataset.items()))

    if fig is None:
        fig, subplots = make_dict_grid(dataset.keys(), **grid_kwargs)
    elif fig.subfigs:
        subplots = subfigure_dict(fig)
    else:
        subplots = axes_dict(fig)

    for key, subplot_data in dataset.items():
        canvas = subplots[key]

        match canvas:
            case Axes():
                subplot_kwargs["ax"] = canvas
            case FigureBase():
                subplot_kwargs["fig"] = canvas

        plotter(subplot_data, **subplot_kwargs)

    return fig


def axes_dict(fig: Figure):
    return {ax.get_label(): ax for ax in fig.axes}


def subfigure_dict(fig: Figure):
    return {subfig.get_label(): subfig for subfig in fig.subfigs}


def basic_canvas(fig_kwargs={}, subplot_kwargs={}):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, fig: Figure | None = None, ax: Axes | None = None, **kwargs):
            if ax:
                ...
            elif fig:
                ax = fig.subplots(**subplot_kwargs)
            else:
                fig, ax = plt.subplots(**fig_kwargs, **subplot_kwargs)

            f = func(*args, ax=ax, **kwargs)

            if not ax.get_title():
                ax.set_title(ax.get_label())

            return f

        return wrapper

    return decorator


def figure_canvas(fig_kwargs={}): ...


def mosaic_canvas(mosaic: list[list[str]], /, fig_kwargs={}, subplot_kwargs={}):
    labels = {label for row in mosaic for label in row}

    def decorator(func):
        @wraps(func)
        def wrapper(*args, fig: Figure | None = None, ax: Axes | None = None, **kwargs):
            if fig is None and ax is None:
                fig, axes = plt.subplot_mosaic(mosaic, **fig_kwargs, **subplot_kwargs)
                return func(*args, fig=fig, **kwargs)

            if ax:  # replace ax with subfigure in same location
                outer_fig = ax.get_figure()
                fig = outer_fig.add_subfigure(ax.get_subplotspec())
                fig.set_label(ax.get_label())
                ax.remove()

            # fig is guaranteed to exist here
            if fig.axes:
                axes = axes_dict(fig)

                if set(axes) != labels:
                    raise ValueError(
                        f"{func} expects canvas with labels {labels} but got {set(axes)}"
                    )
            else:
                fig.subplot_mosaic(mosaic, **subplot_kwargs)

            f = func(*args, fig=fig, **kwargs)

            if fig.get_suptitle():
                fig.suptitle(fig.get_label())

            return f

        return wrapper

    return decorator


## Watermark


def ax_labels(axes: dict, fontsize: int = 30, **kwargs):
    kwargs = (
        dict(
            ha="center",
            va="center",
            fontsize=fontsize,
            color="darkgrey",
            alpha=0.5,
        )
        | kwargs
    )

    for k, ax in axes.items():
        ax.text(0.5, 0.5, k, transform=ax.transAxes, **kwargs)


## Legend


def all_legend_handles_labels(
    axes: Iterable[Axes],
) -> tuple[list, list]:
    handles, labels = zip(*(ax.get_legend_handles_labels() for ax in axes))

    return list(it.chain(*handles)), list(it.chain(*labels))


## Colors


def get_colormap(
    cmap_or_color: Colormap | TColor, dropout: dict | None = None
) -> Colormap:
    """Returns a colormap.

    This helper function allows for easy construction of a colormap. If a colormap is
    given it will be returned with no modifications. Otherwise if a string is given, the
    function will first try to resolve it to a colormap and then fallback to resolving
    it as a color.

    Args:
        cmap_or_color: Can be anything that resolves to either a registered colormap or
            color.
        dropout: If `None`, no dropout is applied. Otherwise, a dropout is applied with
            `get_colormap_with_dropout` using the supplied parameters. Pass in an empty
            dictionary for default parameters. This is only applied when a color is
            given.

    Returns:
        The resulting colormap.
    """
    if isinstance(cmap_or_color, Colormap):
        return cmap_or_color

    try:
        cmap = plt.get_cmap(cmap_or_color)
        return cmap
    except ValueError:
        pass

    try:
        cmap = get_alpha_colormap(cmap_or_color)
    except ValueError as e:
        raise ValueError(f"'{cmap_or_color}' is not a valid color or colormap.") from e

    if dropout is not None:
        cmap = get_colormap_with_dropout(cmap, **dropout)

    return cmap


def get_colormap_with_dropout(
    cmap: str | Colormap,
    threshold: float = 0.2,
    smoothing: float = 0.1,
    alpha: float = None,
) -> Colormap:
    """Constructs a colormap with transparent dropout for low values.

    This is useful for plotting multiple 2D histograms with on the same axis.

    Args:
        cmap (str|Colormap): A colormap or reference to a registered colormap.
        threshold (float): The value below which the colormap will dropout to
            a transparent value. Should be in [0.0, 1.0]
        smoothing (float): A window around the threshold where the colormap will
            linearly interpolate between the original colormap value and the
            dropout.
        alpha (float): The transparency to apply to the original colormap. This
            is necessary because setting alpha outside this function will
            override the dropout.

    Returns:
        (Colormap): The new colormap.
    """

    if isinstance(cmap, str):
        cmap = mpl.cm.get_cmap(cmap)

    n_points = round(100)

    start_interp = threshold - 0.5 * smoothing
    end_interp = threshold + 0.5 * smoothing

    colors = [
        (0.0, (1.0, 1.0, 1.0, 0.0)),
        (start_interp, (1.0, 1.0, 1.0, 0.0)),
        *[(v, cmap(v, alpha)) for v in np.linspace(end_interp, 1, n_points)],
    ]

    new_cmap = LinearSegmentedColormap.from_list(f"{cmap.name}_dropout", colors)

    return new_cmap


def get_alpha_colormap(
    color: TColor, lower: float = 0.1, upper: float = 1.0
) -> Colormap:
    """Constructs a colormap from a color with varying transparency.

    The transparency will be interpolated between lower and upper. The resulting
    colormap can always be reversed with the `reversed(name)` method of the
    colormap.

    Args:
        color (color-like): A str or a tuple of values specifying a color.
            Anything that can be resolved with `matplotlib.colors.to_rgb` can
            be given.
        lower (float): The lower bound for alpha.
        upper (float): The upper bound for alpha.

    Returns:
        (Colormap): A Colormap
    """
    color = to_rgb(color)
    return ListedColormap(
        [(*color, a) for a in np.linspace(lower, upper, 100)],
        name=f'alpha_{to_hex(color).strip("#").upper()}',
    )


def get_berkeley_colormap() -> Colormap:
    """Returns a qualitative colormap with UC Berkeley colors.

    Colors are specified here: https://brand.berkeley.edu/colors/.

    Returns:
        (Colormap): A colormap.
    """
    colors = [
        "#3b7ea1",  # Founder's Rock
        "#c4820e",  # Medalist
        "#859438",  # Soybean
        "#ed4e33",  # Golden Gate
        "#00a598",  # Lap Lane
        "#003262",  # Berkeley Blue
        "#ddd5c7",  # Bay Fog
        "#6C3302",  # South Hall
        "#cfdd45",  # Ion
        "#00b0da",  # Lawrence
    ]

    return ListedColormap(colors, name="berkeley", N=len(colors))


def get_lbnl_colormap() -> Colormap:
    """Returns a qualitative colormap with LBNL colors.

    Colors are specified here: https://creative.lbl.gov/visual-identity/

    Returns:
        (Colormap): A colormap.
    """
    colors = [
        "#007681",  # Teal
        "#D57800",  # Orange
        "#74AA50",  # Green
        "#E8927C",  # Dusty Rose
        "#672E45",  # Burgundy
        "#4298B5",  # Blue
        "#EAAA00",  # Yellow
        "#AC9F3C",  # Olive Green
        "#B1B3B3",  # Light Grey
        "#5D4777",  # Purple
        "#E04E39",  # Red
    ]

    return ListedColormap(colors, name="lbnl", N=len(colors))


__all__ = ["grid_plotter", "make_dict_grid", "make_list_grid"]
