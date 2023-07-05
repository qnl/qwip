import numpy as np
import pytest

from qwip.visualization.utils import (
    ax_labels,
    find_closest_factors,
    get_grid_size,
    make_dict_grid,
    make_list_grid,
)


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


@pytest.mark.parametrize(
    "kwargs,expect",
    [
        (dict(N=5), (1, 5)),
        (dict(N=7), (3, 3)),
        (dict(N=8), (2, 4)),
        (dict(N=11, ncols=5), (3, 5)),
        (dict(N=11, nrows=4), (4, 3)),
        (dict(N=11, ratio=None), (1, 11)),
    ],
)
def test_get_grid_size(kwargs, expect):
    assert get_grid_size(**kwargs) == expect


def test_grid_size_exception():
    with pytest.raises(ValueError):
        get_grid_size(N=11, nrows=3, ncols=3)


@pytest.mark.parametrize("keys", [({f"R{i}": None for i in range(7)}), ({"ax0": None})])
def test_make_dict_grid(keys):
    import matplotlib.pyplot as plt

    fig, axes = make_dict_grid(keys)
    assert axes.keys() == keys.keys()
    ax_labels(axes)
    plt.show()


def test_make_grid_01():
    fig, axes = make_list_grid(1)

    assert axes.shape == (1,)


def test_make_grid_02():
    fig, axes = make_list_grid(5, nrows=2)

    assert axes.shape == (2, 3)
    assert axes[1, 2].axison is False
    assert np.all([ax.axison for ax in axes.flat[:5]])
