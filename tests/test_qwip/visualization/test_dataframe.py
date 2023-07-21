import itertools as it

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from qwip.visualization.dataframe import DataFramePlotter


class TestDataFramePlotter:
    def test_get_indices(self):
        X = -np.arange(15)
        Y = np.arange(20)
        index = pd.MultiIndex.from_tuples(it.product(Y, X), names=["Y", "X"])

        xs, ys = DataFramePlotter.get_indices(index)

        assert (xs == np.sort(pd.Index(X, name="X"))).all()
        assert (ys == np.sort(pd.Index(Y, name="Y"))).all()

    def test_get_values(self):
        radius = 10
        xs = np.linspace(-10, 10)
        ys = np.linspace(-10, 10, 40)

        index = []
        values = []
        for x, y in it.product(xs, ys):
            r = np.sqrt(x**2 + y**2)

            if r > radius:
                continue

            index.append((y, x))
            values.append((r, r**2))

        df = pd.DataFrame(
            np.array(values),
            columns=["distance", "distance squared"],
            index=pd.MultiIndex.from_tuples(index, names=["y", "x"]),
        )
        fulldf = DataFramePlotter.get_values(
            pd.Index(xs, name="x"), pd.Index(ys, name="y"), df
        )

        expected_index = pd.MultiIndex.from_tuples(it.product(ys, xs), names=["y", "x"])
        rs = ys[:, np.newaxis] ** 2 + xs[np.newaxis, :] ** 2
        r = np.sqrt(rs)

        expected = pd.DataFrame(r.flatten(), columns=["distance"], index=expected_index)
        expected["distance squared"] = rs.flatten()
        expected.loc[expected["distance"] > radius] = np.nan

        assert_frame_equal(fulldf, expected)
