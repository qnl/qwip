import re

from pathlib import Path
import pytest
import numpy as np
import qwip

from numpy.testing import assert_allclose

from qwip.sequencer.waveform import (
    update_fields,
    Waveform,
    BasicWaveform,
    Channel,
    MarkerWaveform,
    ModulationFrequency,
    ModulatedWaveform,
    CosineRampWaveform,
    CWWaveform,
    DRAG,
    DCWaveform,
    SquareWaveform,
    GaussianWaveform
)

from qwip.sequencer.sequence import (
    SequenceElement
)

import matplotlib.pyplot as plt

@pytest.fixture
def data_file(request):
    fspath = Path(request.fspath)

    file = fspath.name
    name = re.match(r'test_(?P<name>.*)\.py', str(file)).group('name')
    datadir = fspath.parent / name

    if request.cls:
        datadir = datadir / request.cls.__name__.lstrip('Test')

    return datadir / f'{request.node.name.lstrip("test_")}.txt'

class TestBasicWaveforms:
    @pytest.mark.parametrize(
        'name,expect',
        [
            (None, 'BasicWaveform'),
            ('name', 'name')
        ]
    )
    def test_name(self, name, expect):
        w = BasicWaveform(name=name) if name else BasicWaveform()
        assert w.name == expect

    def test_create(self):
        w = BasicWaveform()
        
        assert w.name == 'BasicWaveform'
        assert w.channels == tuple()
        assert w.width == 0
        assert w.amplitude == 1
        assert w.t0 == 0

        w = BasicWaveform(name='name')
        assert w.name == 'name'

    def test_convert(self):
        w = BasicWaveform(channels=(0, 'Q1'), width=1, amplitude='A')

        assert w.channels == (Channel('0'), Channel('Q1'))
        assert w.width == 1.0 and isinstance(w.width, float)
        assert w.amplitude == 'A'

    @pytest.mark.parametrize(
        'kwargs,expect',
        [
            (dict(), dict(width='w', amplitude='amp', t0=0)),
            (dict(w=10, amp=20), dict(width=10, amplitude=20, t0=0)),
            (dict(width=10, amplitude=20, t0=1), dict(width=10, amplitude=20, t0=1)),
            (dict(random=10), dict(width='w', amplitude='amp', t0=0, random=10))
        ]
    )
    def test_update_fields(self, kwargs, expect):
        w = BasicWaveform(width='w', amplitude='amp')

        assert update_fields(w, **kwargs) == expect


class TestCWWaveform:
    @pytest.mark.parametrize(
        'ts,phase_jumps,expected',
        [
            (
                np.arange(10),
                np.array([(0, 0), (4.5, 1)]),
                np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
            ),
            (
                np.arange(10) / 2,
                np.array([(0, 0), (3, 1)]),
                np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
            ),
            (
                np.arange(10) * 5,
                np.array([(0, 0), (51, 1)]),
                np.zeros(10)
            ),
            (
                np.arange(5),
                np.array([(0, 0), (4, 1)]),
                np.array([0, 0, 0, 0, 1])
            ),
            (
                np.arange(10),
                np.array([(0, 0), (1.9, 1), (4.7, -1), (20, -2)]),
                np.array([0, 0, 1, 1, 1, -1, -1, -1, -1, -1])
            ),
            (
                np.arange(10) + 10,
                np.array([(0, 0), (9, 1), (10, -1), (15, 1), (19, -2)]),
                np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, -2])
            ),
            (
                np.arange(10) + 10,
                np.array([(0, 0), (9, 1), (14.5, -1), (20, -2)]),
                np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
            ),
            (
                np.arange(10) + 10,
                np.array([(0, 0), (9, 1), (20, -2)]),
                np.ones(10)
            )
        ]
    )
    def test_integrated_phase(self, ts, phase_jumps, expected):
        modwave = CWWaveform(
            frequency='Q0'
        )

        phase_tracker = {
            ModulationFrequency('Q0'): phase_jumps
        }

        integrated_phase = modwave.compute_integrated_phase(ts, phase_tracker)

        assert_allclose(integrated_phase, expected)

    @pytest.mark.parametrize(
        'ts,phase_jumps',
        [
            (np.linspace(0, 5, 21), np.array([(0, 0)])),
            (np.linspace(0, 5, 21), np.array([(0, 0), (1.25, 90), (2.5, -90)])),
            (np.linspace(1.25, 6.25, 21), np.array([(0, 0), (1, -90)]))
        ]
    )
    def test_modulation(self, ts, phase_jumps, data_file):
        expected = np.loadtxt(str(data_file))

        w = CWWaveform(
            frequency=ModulationFrequency(0.2),
            channels=('I', 'Q')
        )

        phase_tracker = {
            ModulationFrequency(0.2): phase_jumps
        }

        wave = w(ts, phase_tracker=phase_tracker, phase_unit='degrees')
        assert_allclose(wave, expected)

        w_single_channel = w.evolve(channels=('I',))
        wave = w_single_channel(ts, phase_tracker=phase_tracker, phase_unit='degrees')
        assert_allclose(wave, expected[0])

        w_three_channel = w.evolve(channels=('a', 'b', 'c'))
        wave = w_three_channel(ts, phase_tracker=phase_tracker, phase_unit='degrees')
        assert_allclose(wave, np.stack([expected[0] for i in range(3)]))

    @pytest.mark.parametrize(
        'ts,phase_jumps',
        [
            (np.linspace(0, 10, 51), np.array([(0, 0), (1, 90)])), 
        ]
    )
    def test_units(self, ts, phase_jumps):
        pj_deg = phase_jumps
        pj_rad = phase_jumps * np.array([[1, np.pi / 180]])

        mod_freq = ModulationFrequency(0.5)

        w = CWWaveform(
            frequency=mod_freq,
            channels=('I', 'Q')
        )

        wave_d = w(ts, phase_tracker={mod_freq: pj_deg}, phase_unit='degrees')
        wave_r = w(ts, phase_tracker={mod_freq: pj_rad}, phase_unit='radians')

        assert_allclose(wave_d, wave_r)

class TestModulatedWaveform:
    @pytest.mark.parametrize(
        'env,mod,expected',
        [
            (
                dict(t0=1, width=10),
                dict(amplitude=2),
                dict(t0=1, width=10, amplitude=2)
            ),
            (
                dict(width=20, amplitude=2),
                dict(amplitude=0.5),
                dict(t0=0, width=20, amplitude=1)
            )
        ]
    )
    def test_properties(self, env, mod, expected):
        env = SquareWaveform(**env)
        mod = CWWaveform(frequency='f', **mod)

        wave = ModulatedWaveform(envelope=env, mod_freq=mod)

        assert dict((p, getattr(wave, p)) for p in expected.keys()) == expected

    @pytest.mark.parametrize(
        'ts,phase_jumps',
        [
            (np.arange(240) / 2.4e9, np.array([(0, 0)])),
            (np.arange(240) / 2.4e9, np.array([(0, 0), (20e-9, 180)])),
            (np.arange(240) / 2.4e9, np.array([(0, 0), (60e-9, 90)])),
        ]
    )
    def test_IQ_modulation(self, ts, phase_jumps, data_file):
        expected = np.loadtxt(str(data_file))

        freq = ModulationFrequency(100e6)
        env = SquareWaveform(width=40e-9)
        mod = CWWaveform(
            frequency=freq,
            channels=('I', 'Q')
        )

        phase_tracker = {
            freq: phase_jumps
        }

        w = ModulatedWaveform(envelope=env, mod_freq=mod)

        ts = np.arange(240) / 2.4e9
        wave = w(ts, t0=40e-9, phase_tracker=phase_tracker)

        assert_allclose(wave, expected)

    @pytest.mark.parametrize(
        'wave,val',
        [
            (
                ModulatedWaveform(
                    name='X90',
                    envelope=GaussianWaveform(width=40e-9, amplitude=0.5),
                    mod_freq=CWWaveform(
                        frequency='f',
                        channels=('I', 'Q')
                    )
                ),
                dict(
                    name='X90',
                    envelope=dict(
                        width=4e-8,
                        amplitude=0.5,
                        __class__='GaussianWaveform'
                    ),
                    mod_freq=dict(
                        channels=[dict(name='I'), dict(name='Q')],
                        frequency=dict(offset='f'),
                        __class__='CWWaveform'
                    ),
                    __class__='ModulatedWaveform'
                )
            ),
        ]
    )
    def test_serialization(self, wave, val):
        unstruct = qwip.converter.unstructure(wave)
        restruct = qwip.converter.structure(unstruct, ModulatedWaveform)

        assert unstruct == val
        assert restruct == wave

class TestDRAGWaveform:
    def test_timepoints(self):
        env = DRAG(
            envelope=GaussianWaveform(width=20e-9),
            lmbda=-2/(2*np.pi*200e6)
        )

        freq = CWWaveform(
            frequency=ModulationFrequency(500e6),
            channels=('I', 'Q')
        )

        w = ModulatedWaveform(envelope=env, mod_freq=freq)

        ts = np.arange(480) / 2.4e9

        vals = w(ts, t0=100e-9)

        fig, axes = plt.subplots(2, 1)
        ax = axes[0]
        ax.plot(ts*1e9, vals[0])
        ax.plot(ts*1e9, vals[1])
        ax.grid(True)
        ax = axes[1]
        ks, fs = w.fft(ts, t0=100e-9)

        ax.plot(ks/1e6, np.abs(fs))
        ax.grid(True)
        ax.axvline(400, color='k', linestyle='--')
        plt.show()
