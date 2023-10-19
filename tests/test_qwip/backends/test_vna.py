import itertools as it
from contextlib import nullcontext

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

    @pytest.mark.parametrize(
        "parameters,error",
        [
            (dict(power=np.arange(-50, -10)), False),
            (dict(center=np.linspace(5.5e9, 5.6e9, 21)), False),
            (
                dict(power=np.arange(-50, -10), center=np.linspace(5.5e9, 5.6e9, 40)),
                False,
            ),
            (dict(power=np.arange(-50, -10), averages=np.ones(5)), ValueError),
        ],
    )
    def test_sweep(self, parameters, error):
        exe = VNAExecutable(start=5e9, stop=6e9, points=100)

        maybe_err = pytest.raises(error) if error else nullcontext()

        with maybe_err:
            sweep = exe.sweep(**parameters)
            assert len({len(sweep)} | set(len(val) for val in parameters.values())) == 1

            for param, values in parameters.items():
                assert values.tolist() == [getattr(e, param) for e in sweep]


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

        return VNABackend(device=vna)

    def test_upload_none(self, backend):
        backend.device.start(6e9)
        backend.device.stop(7e9)
        backend.device.points(1001)
        backend.device.power(-70)
        backend.device.averages(1)
        backend.device.electrical_delay(0)
        backend.device.if_bandwidth(1000)
        backend.device.trace("S21")

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

        assert backend.device.start() == 5e9
        assert backend.device.stop() == 6e9
        assert backend.device.points() == 501
        assert backend.device.power() == -60
        assert backend.device.averages() == 2
        assert backend.device.averages_enabled() is True
        assert backend.device.electrical_delay() == 100e-9
        assert backend.device.if_bandwidth() == 500
        assert backend.device.trace() == "S11"

    def test_acquire(self, backend):
        result = backend.acquire()[backend.device.name]

        assert isinstance(result, IQResult)
        assert list(result.data.index.names) == ["frequency"]
        assert len(result.data) == backend.device.points()
