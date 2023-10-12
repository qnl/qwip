import itertools as it

import numpy as np
import pandas as pd
import pytest

from qwip.backends.vna import VNABackend, VNAExecutable, sweep_parameters
from qwip.processing.processors import IQResult


class TestVNAExecutable:
    def test_center(self):
        exe = VNAExecutable()

        with pytest.raises(ValueError):
            exe.center = 7e9

        exe.start = 6e9
        exe.stop = 7e9
        exe.center = 7e9
        assert (exe.start, exe.stop) == (6.5e9, 7.5e9)
        assert exe.center == 7e9

    def test_span(self):
        exe = VNAExecutable()

        with pytest.raises(ValueError):
            exe.span = 1e9

        exe.start = 5e9
        exe.stop = 6e9
        exe.span = 2e9

        assert (exe.start, exe.stop) == (4.5e9, 6.5e9)
        assert exe.span == 2e9


def test_empty_sweep_parameters():
    exe_list = [VNAExecutable()] * 5
    sweep = sweep_parameters(exe_list)

    assert sweep.equals(pd.RangeIndex(0, 5))


def test_single_sweep_parameters():
    exe_list = [VNAExecutable(power=p) for p in [-50, -40, -30]]

    sweep = sweep_parameters(exe_list)

    assert sweep.equals(
        pd.MultiIndex.from_tuples([(-50,), (-40,), (-30,)], names=["power"])
    )


def test_repeats_sweep_parameters():
    exe_list = [VNAExecutable(power=p, averages=1) for p in (-50, -40, -30)]

    sweep = sweep_parameters(exe_list)

    assert sweep.equals(
        pd.MultiIndex.from_tuples([(-50,), (-40,), (-30,)], names=["power"])
    )


def test_nan_sweep_parameters():
    exe_list = [VNAExecutable(averages=1), VNAExecutable(averages=2), VNAExecutable()]

    sweep = sweep_parameters(exe_list)
    assert sweep.equals(pd.MultiIndex.from_tuples([(1,), (2,), (np.nan,)]))


def test_multi_sweep_parameters():
    exe_list = [
        VNAExecutable(power=p, averages=a) for p, a in it.product([-50, -40], [1, 2, 3])
    ]

    sweep = sweep_parameters(exe_list)
    assert sweep.equals(
        pd.MultiIndex.from_product(([-50, -40], [1, 2, 3]), names=["power", "averages"])
    )


@pytest.mark.skip_instrument("vna")
class TestVNABackend:
    @pytest.fixture
    def backend(self, instrument_server):
        vna = instrument_server["vna"]

        return VNABackend(vna=vna)

    def test_upload_none(self, backend):
        backend.vna.start(6e9)
        backend.vna.stop(7e9)
        backend.vna.points(1001)
        backend.vna.power(-70)
        backend.vna.averages(1)
        backend.vna.electrical_delay(0)
        backend.vna.if_bandwidth(1000)
        backend.vna.trace("S21")

        exe = VNAExecutable()
        backend.upload(exe)

        assert exe == VNAExecutable(
            start=6e9,
            stop=7e9,
            points=1001,
            power=-70,
            averages=1,
            delay=0,
            if_bandwidth=1000,
            meas="S21",
        )

    def test_upload_explicit(self, backend):
        exe = VNAExecutable(
            start=5e9,
            stop=6e9,
            points=501,
            power=-60,
            averages=2,
            delay=100e-9,
            if_bandwidth=500,
            meas="S11",
        )
        backend.upload(exe)

        assert backend.vna.start() == 5e9
        assert backend.vna.stop() == 6e9
        assert backend.vna.points() == 501
        assert backend.vna.power() == -60
        assert backend.vna.averages() == 2
        assert backend.vna.averages_enabled() is True
        assert backend.vna.electrical_delay() == 100e-9
        assert backend.vna.if_bandwidth() == 500
        assert backend.vna.trace() == "S11"

    def test_acquire(self, backend):
        result = backend.acquire()[backend.vna.name]

        assert isinstance(result, IQResult)
        assert list(result.data.index.names) == ["frequency"]
        assert len(result.data) == backend.vna.points()
