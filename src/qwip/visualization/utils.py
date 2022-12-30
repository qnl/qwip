"""Useful utility functions for working visualizing data with matplotlib.
"""

import itertools as it
from collections.abc import Iterable

import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.colors import (
    Colormap,
    LinearSegmentedColormap,
    ListedColormap,
    to_rgb,
    to_hex
)

TColor = str | tuple[float, float, float] | tuple[float, float, float, float]

def all_legend_handles_labels(
    axes: Iterable[Axes],
) -> tuple[list, list]:
    handles, labels = zip(*(ax.get_legend_handles_labels() for ax in axes))

    return list(it.chain(*handles)), list(it.chain(*labels))

def get_colormap_with_dropout(
    cmap: str | Colormap,
    threshold: float = 0.2,
    smoothing: float = 0.1,
    alpha: float = None
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
        *[(v, cmap(v, alpha)) for v in np.linspace(end_interp, 1, n_points)]
    ]

    new_cmap = LinearSegmentedColormap.from_list(
        f'{cmap.name}_dropout',
        colors
    )

    return new_cmap

def get_alpha_colormap(
    color: TColor,
    lower: float = 0.1,
    upper: float = 1.0
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
        name=f'alpha_{to_hex(color).strip("#").upper()}'
    )

def get_berkeley_colormap() -> Colormap:
    """Returns a qualitative colormap with UC Berkeley colors.
    
    Colors are specified here: https://brand.berkeley.edu/colors/.

    Returns:
        (Colormap): A colormap.
    """
    colors = [
        '#3b7ea1', # Founder's Rock
        '#c4820e', # Medalist
        '#859438', # Soybean
        '#ed4e33', # Golden Gate
        '#00a598', # Lap Lane
        '#003262', # Berkeley Blue
        '#ddd5c7', # Bay Fog
        '#6C3302', # South Hall
        '#cfdd45', # Ion
        '#00b0da', # Lawrence
    ]
    
    return ListedColormap(colors, name='berkeley', N=len(colors))

def get_lbnl_colormap() -> Colormap:
    """Returns a qualitative colormap with LBNL colors.
    
    Colors are specified here: https://creative.lbl.gov/visual-identity/

    Returns:
        (Colormap): A colormap.
    """
    colors = [
        '#007681', # Teal
        '#D57800', # Orange
        '#74AA50', # Green
        '#E8927C', # Dusty Rose
        '#672E45', # Burgundy
        '#4298B5', # Blue
        '#EAAA00', # Yellow
        '#AC9F3C', # Olive Green
        '#B1B3B3', # Light Grey
        '#5D4777', # Purple
        '#E04E39', # Red
    ]

    return ListedColormap(colors, name='lbnl', N=len(colors))