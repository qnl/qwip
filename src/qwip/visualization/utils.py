"""Useful utility functions for working visualizing data with matplotlib.
"""

import itertools as it
from collections.abc import Iterable

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

TColor = str | tuple[float, float, float] | tuple[float, float, float, float]


## Grid


def find_closest_factors(n: int, /):
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


def make_grid(N, nrows=None, ncols=None, hide_unused=True, **kwargs) -> np.ndarray:
    """Makes a figure subplot grid with at least N subplots.

    Args:
        N: The number of subplots requested.
        nrows: If not `None`, specifies the total number of rows.
        ncols: If not `None`, specifies the total number of columns.
        hide_unsued: If `True`, will hide all extra subplots. The N "active" subplots
            start at the top left corner and go down the grid in row-major order.
        **kwargs: Additional keyword arguments are passed to `plt.subplots`

    Returns:
        A numpy array of empty Axes objects.
    """
    if nrows is None and ncols is None:
        nrows, ncols = find_closest_factors(N)
    elif nrows is None:
        nrows = np.ceil(N / ncols).astype(int)
    elif ncols is None:
        ncols = np.ceil(N / nrows).astype(int)
    elif nrows * ncols < N:
        raise ValueError(
            f"Not enough axes in the specified grid. nrows * ncols must be greater than"
            f" N. Got {nrows} * {ncols} = {nrows * ncols} < {N}."
        )

    fig, axes = plt.subplots(nrows, ncols, **kwargs)

    if nrows == ncols == 1:
        axes = np.array([axes])

    if N < nrows * ncols and hide_unused:
        for ax in axes.flat[N:]:
            ax.axis("off")

    fig.tight_layout()

    return fig, axes


## Legend


def all_legend_handles_labels(
    axes: Iterable[Axes],
) -> tuple[list, list]:
    handles, labels = zip(*(ax.get_legend_handles_labels() for ax in axes))

    return list(it.chain(*handles)), list(it.chain(*labels))


## Colors


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
