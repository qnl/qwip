import itertools as it

import matplotlib.pyplot as plt
import numpy as np
import pytest

from qwip.visualization.utils import (
    find_closest_factors,
    get_axes_by_position,
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


def test_get_axes_by_position():
    _, axes = make_dict_grid([f"R{r}C{c}" for r, c in it.product(range(3), repeat=2)])

    left = [ax.get_label() for ax in get_axes_by_position(axes.values(), "x")]
    right = [ax.get_label() for ax in get_axes_by_position(axes.values(), "x", max)]
    bottom = [ax.get_label() for ax in get_axes_by_position(axes.values(), "y")]
    top = [ax.get_label() for ax in get_axes_by_position(axes.values(), "y", max)]

    assert left == ["R0C0", "R1C0", "R2C0"]
    assert right == ["R0C2", "R1C2", "R2C2"]
    assert bottom == ["R2C0", "R2C1", "R2C2"]
    assert top == ["R0C0", "R0C1", "R0C2"]


@pytest.mark.parametrize("keys", [({f"R{i}": None for i in range(7)}), ({"ax0": None})])
def test_make_dict_grid(keys):
    fig, axes = make_dict_grid(keys)

    assert axes.keys() == keys.keys()
    assert [ax.get_label() for ax in fig.axes] == list(keys)


def test_make_list_grid_01():
    fig, axes = make_list_grid(1)

    assert axes.shape == (1,)


def test_make_list_grid_02():
    fig, axes = make_list_grid(5, nrows=2)

    assert axes.shape == (2, 3)
    assert axes[1, 2].axison is False
    assert np.all([ax.axison for ax in axes.flat[:5]])
