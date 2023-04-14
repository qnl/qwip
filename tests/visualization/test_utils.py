import numpy as np
import pytest

from qwip.visualization.utils import find_closest_factors, make_grid


@pytest.mark.parametrize(
    "n,result",
    [
        (1, (1, 1)),
        (3, (1, 3)),
        (4, (2, 2)),
        (6, (2, 3)),
        (8, (2, 4)),
        (24, (4, 6)),
        (32, (4, 8)),
    ],
)
def test_find_closest_factors(n, result):
    assert find_closest_factors(n) == result


def test_make_grid_01():
    fig, axes = make_grid(1)

    assert axes.shape == (1,)


def test_make_grid_02():
    fig, axes = make_grid(5, nrows=2)

    assert axes.shape == (2, 3)
    assert axes[1, 2].axison is False
    assert np.all([ax.axison for ax in axes.flat[:5]])


def test_make_grid_03():
    with pytest.raises(ValueError):
        fig, axes = make_grid(5, nrows=2, ncols=2)
