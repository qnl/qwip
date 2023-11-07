import itertools as it

import numpy as np
import pandas as pd
from attrs import define
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from qwip.visualization.utils import get_axes_by_position, make_dict_grid


@define
class DataFramePlotter:
    """Plots multi-indexed dataframes as a 2D color plot."""

    @staticmethod
    def get_indices(
        index: pd.DataFrame | pd.Series | pd.Index,
        labels: tuple[str, str] | None = None,
    ) -> tuple[np.array, np.array]:
        """Gets indices along each axis of a multiindex with two levels.

        Args:
            index: An index, Series or DataFrame, with unique index values.
            labels: A tuple of string labels specifying which index levels to plot along
                the x and y axes. If `None`, defaults to the first two levels of the
                multiindex.

        Returns:
            A tuple of sorted and unique index values along x and y.
        """
        if isinstance(index, pd.DataFrame | pd.Series):
            index = index.index

        if not labels:
            labels = index.names

        ys = index.unique(labels[0]).sort_values()
        xs = index.unique(labels[1]).sort_values()
        return xs, ys

    @staticmethod
    def get_values(
        xs: np.ndarray,
        ys: np.ndarray,
        df: pd.DataFrame,
        fill_value: float = np.nan,
    ) -> pd.DataFrame:
        """Reindex a multi-indexed dataframe into a full grid.

        Args:
            xs: An array of x-axis values.
            ys: An array of y-axis values.
            df: The dataframe to reindex.
            fill_value: Used to fill missing values, defaults to `np.nan`.

        Returns:
            A dataframe that has been reindexed such that all index values within the
            have a value.
        """
        df = df.droplevel([n for n in df.index.names if n != xs.name and n != ys.name])
        return df.reindex(index=list(it.product(ys, xs)), fill_value=fill_value)

    def plot(
        self,
        df: pd.DataFrame,
        fill_value: float = np.nan,
        labels: tuple[str, str] | None = None,
        cmap: str | Colormap = "RdBu",
        colorbar: bool = True,
        fig_kwargs: dict = {},
        pc_kwargs: dict = {},
    ) -> Figure:
        """Plots a dataframe with a 2D multiindex as a 2D color plot.

        Args:
            df: The dataframe to plot.
            fill_value: This is passed to `get_values`.
            labels: This is passed to `get_indices`.
            cmap: The colormap to use.
            colorbar: If `True`, draws a colorbar for each subplot.
            fig_kwargs: Arguments to pass to `make_dict_grid`.
            pc_kwargs: Arguments to pass to `pcolormesh`.

        Returns:
            The matplotlib `Figure` that contains the plots.
        """
        fig, axes = make_dict_grid(df.columns, sharex=True, sharey=True, **fig_kwargs)

        xs, ys = type(self).get_indices(df, labels=labels)
        fulldf = type(self).get_values(xs, ys, df, fill_value=fill_value)

        pc_kwargs = pc_kwargs | dict(cmap=cmap)

        for col in fulldf:
            ax = axes[col]
            ax.grid(False)
            series = fulldf[col]
            im = ax.pcolormesh(
                xs, ys, series.to_numpy().reshape(len(ys), len(xs)), **pc_kwargs
            )

            if colorbar:
                cb = fig.colorbar(im)

            ax.set_title(col)

        for ax in get_axes_by_position(axes.values(), "y"):
            ax.set_xlabel(xs.name)
        for ax in get_axes_by_position(axes.values(), "x"):
            ax.set_ylabel(ys.name)

        return fig


__all__ = ["DataFramePlotter"]
