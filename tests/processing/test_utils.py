"""Tests for qwip.processing.utils"""
import numpy as np
import pytest
from loguru import logger
from numpy.random import default_rng
from numpy.testing import assert_allclose, assert_array_equal

from qwip.flatdict import FlatDict
from qwip.processing.process import ProcessSettings

logger.enable("qwip")


class TestFormatLegacyHeterodyne:
    @pytest.fixture
    def meas(self):
        return {
            "raw_trace": None,
            "joint": None,
            "x_axis": None,
            "settings": {"long": "unreadable", "settings": "dictionary"},
            "R0": {
                "Heterodyne": np.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5)),
            },
            "R1": {"Heterodyne": np.arange(2 * 256 * 20 * 1).reshape((2, 256, 20, 1))},
        }

    @pytest.fixture
    def reformat(self):
        psettings = ProcessSettings(
            name="reformat", process_type="utils.FormatLegacyHeterodyne"
        )

        return psettings.get_process()

    def test_get_keys(self, reformat, meas):
        output = reformat(meas)

        assert tuple(output.keys()) == ("R0", "R1")

    def test_data(self, reformat, meas):
        output = reformat(meas)

        assert_allclose(output["R0"][..., 0], meas["R0"]["Heterodyne"][0])
        assert_allclose(output["R0"][..., 1], meas["R0"]["Heterodyne"][1])
        assert_allclose(output["R1"][..., 0], meas["R1"]["Heterodyne"][0])
        assert_allclose(output["R1"][..., 1], meas["R1"]["Heterodyne"][1])


class TestRename:
    @pytest.fixture
    def data(self):
        return {f"R{i}": np.ones(3) * i for i in range(10)}

    @pytest.fixture
    def psettings(self):
        psettings = ProcessSettings(name="rename", process_type="utils.Rename")

        return psettings

    @pytest.mark.parametrize(
        "mapping,keys",
        [({"R0": "new/R0"}, ["new/R0"] + [f"R{i}" for i in range(1, 10)])],
    )
    def test_rename_by_mapping(self, psettings, data, mapping, keys):
        rename = psettings.get_process(rename=mapping)

        output = rename(data)

        assert list(output.keys()) == keys

    def test_rename_by_callable(self, psettings, data):
        rename = psettings.get_process(rename=lambda x: f"new/{x}")

        output = rename(data)
        assert list(output.keys()) == [
            "new/R0",
            "new/R1",
            "new/R2",
            "new/R3",
            "new/R4",
            "new/R5",
            "new/R6",
            "new/R7",
            "new/R8",
            "new/R9",
        ]


class TestFilterData:
    @pytest.fixture
    def data(self):
        return {f"R{i}": np.ones(3) * i for i in range(10)}

    @pytest.fixture
    def psettings(self):
        psettings = ProcessSettings(name="filter", process_type="utils.FilterData")

        return psettings

    @pytest.mark.parametrize(
        "regex,keys",
        [
            ("R1", ["R1"]),
            ("R[0-4]", [f"R{i}" for i in range(5)]),
            ("R[4|2]", ["R2", "R4"]),
            ("Q\\d+", []),
        ],
    )
    def test_filter_by_regex(self, psettings, data, regex, keys):
        filter = psettings.get_process(filter=regex)

        output = filter(data)

        assert list(output.keys()) == keys

    @pytest.mark.parametrize(
        "container,keys", [(["R1"], ["R1"]), (["Q1", "R1"], ["R1"]), (["Q1"], [])]
    )
    def test_filter_by_container(self, psettings, data, container, keys):
        filter = psettings.get_process(filter=container)

        output = filter(data)
        assert list(output.keys()) == keys

    def test_filter_by_callable(self, psettings, data):
        filter = psettings.get_process(filter=lambda x: False)

        output = filter(data)
        assert list(output.keys()) == []

        filter = psettings.get_process(filter=lambda x: True)

        output = filter(data)
        assert list(output.keys()) == list(data.keys())


class TestCollectData:
    @pytest.fixture
    def psettings(self):
        psettings = ProcessSettings(name="collect", process_type="utils.CollectData")

        return psettings

    def test_process_outer(self, psettings):
        collect = psettings.get_process()
        psettings.inputs = ("a", "b", "c")

        output = collect({"R0": 0}, {"R0": 1}, {"R0": 2})

        assert output == FlatDict({"a": {"R0": 0}, "b": {"R0": 1}, "c": {"R0": 2}})

    def test_label_outer(self, psettings):
        collect = psettings.get_process(outer_key="label")

        psettings.inputs = ("a", "b", "c")
        output = collect({"R0": 0}, {"R0": 1}, {"R1": 2})

        assert output == FlatDict({"R0": {"a": 0, "b": 1}, "R1": {"c": 2}})
