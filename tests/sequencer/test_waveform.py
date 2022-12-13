import pytest
import numpy as np

from numpy.testing import assert_allclose

from qwip.sequencer.waveform import (
    Waveform,
    Channel,
    MarkerWaveform,
    ModulationFrequency,
    ModulatedWaveform,
    DCWaveform,
    SquareWaveform,
    GaussianWaveform
)

from qwip.sequencer.sequence import (
    SequenceElement
)

import matplotlib.pyplot as plt

class TestWaveforms:
    @pytest.mark.parametrize(
        'name,expect',
        [
            (None, 'Waveform'),
            ('name', 'name')
        ]
    )
    def test_name(self, name, expect):
        w = Waveform(name=name) if name else Waveform()
        assert w.name == expect

    def test_create(self):
        w = Waveform()
        
        assert w.name == 'Waveform'
        assert w.channels == tuple()
        assert w.width == 0
        assert w.amplitude == 1
        assert w.t0 == 0

        w = Waveform(name='name')
        assert w.name == 'name'

    def test_convert(self):
        w = Waveform(channels=(0, 'Q1'), width=1, amplitude='A')

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
        w = Waveform(width='w', amplitude='amp')

        assert w.update_fields(**kwargs) == expect

    def test_square(self):
        import numpy as np
        s = GaussianWaveform(width='w', amplitude=10, cutoff=4)

        ts = np.linspace(0, 100)

class TestModulatedWaveform:

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
        modwave = ModulatedWaveform(
            envelope=SquareWaveform(),
            mod_freq='Q0'
        )

        phase_tracker = {
            ModulationFrequency('Q0'): phase_jumps
        }

        integrated_phase = modwave.compute_integrated_phase(ts, phase_tracker)

        assert_allclose(integrated_phase, expected)

    @pytest.mark.parametrize(
        'ts,phase_jumps,expected',
        [
            (
                np.linspace(0, 5, 21),
                np.array([(0, 0)]),
                np.array([[
                        1.0000000e+00,  9.5105654e-01,  8.0901700e-01,  5.8778524e-01,
                        3.0901697e-01, -4.3711388e-08, -3.0901703e-01, -5.8778518e-01,
                        -8.0901706e-01, -9.5105648e-01, -1.0000000e+00, -9.5105654e-01,
                        -8.0901694e-01, -5.8778507e-01, -3.0901709e-01,  1.1924881e-08,
                        3.0901712e-01,  5.8778507e-01,  8.0901694e-01,  9.5105654e-01,
                        1.0000000e+00
                    ],
                    [
                        0.0000000e+00,  3.0901700e-01,  5.8778524e-01,  8.0901700e-01,
                        9.5105654e-01,  1.0000000e+00,  9.5105648e-01,  8.0901700e-01,
                        5.8778518e-01,  3.0901703e-01, -8.7422777e-08, -3.0901697e-01,
                        -5.8778536e-01, -8.0901712e-01, -9.5105648e-01, -1.0000000e+00,
                        -9.5105648e-01, -8.0901712e-01, -5.8778530e-01, -3.0901694e-01,
                        1.7484555e-07
                    ]],
                    dtype=np.float32
                )
            ),
            (
                np.linspace(0, 5, 21),
                np.array([(0, 0), (1.25, 90), (2.5, -90)]),
                np.array([[
                        1.0000000e+00,  9.5105654e-01,  8.0901700e-01,  5.8778524e-01,
                        3.0901697e-01, -1.0000000e+00, -9.5105654e-01, -8.0901694e-01,
                        -5.8778507e-01, -3.0901709e-01, -4.3711388e-08, -3.0901703e-01,
                        -5.8778518e-01, -8.0901706e-01, -9.5105648e-01, -1.0000000e+00,
                        -9.5105654e-01, -8.0901694e-01, -5.8778507e-01, -3.0901709e-01,
                        1.1924881e-08
                    ],
                    [
                        0.0000000e+00,  3.0901700e-01,  5.8778524e-01,  8.0901700e-01,
                        9.5105654e-01, -8.7422777e-08, -3.0901697e-01, -5.8778536e-01,
                        -8.0901712e-01, -9.5105648e-01,  1.0000000e+00,  9.5105648e-01,
                        8.0901700e-01,  5.8778518e-01,  3.0901703e-01, -8.7422777e-08,
                        -3.0901697e-01, -5.8778536e-01, -8.0901712e-01, -9.5105648e-01,
                        -1.0000000e+00
                    ]],
                    dtype=np.float32
                )
            ),
            (
                np.linspace(1.25, 6.25, 21),
                np.array([(0, 0), (1, -90)]),
                np.array([[
                        1.0000000e+00,  9.5105654e-01,  8.0901700e-01,  5.8778524e-01,
                        3.0901697e-01, -4.3711388e-08, -3.0901703e-01, -5.8778518e-01,
                        -8.0901706e-01, -9.5105648e-01, -1.0000000e+00, -9.5105654e-01,
                        -8.0901694e-01, -5.8778507e-01, -3.0901709e-01,  1.1924881e-08,
                        3.0901712e-01,  5.8778507e-01,  8.0901694e-01,  9.5105654e-01,
                        1.0000000e+00
                    ],
                    [
                        0.0000000e+00,  3.0901700e-01,  5.8778524e-01,  8.0901700e-01,
                        9.5105654e-01,  1.0000000e+00,  9.5105648e-01,  8.0901700e-01,
                        5.8778518e-01,  3.0901703e-01, -8.7422777e-08, -3.0901697e-01,
                        -5.8778536e-01, -8.0901712e-01, -9.5105648e-01, -1.0000000e+00,
                        -9.5105648e-01, -8.0901712e-01, -5.8778530e-01, -3.0901694e-01,
                        1.7484555e-07
                    ]],
                    dtype=np.float32
                )
            )
        ]
    )
    def test_modulation(self, ts, phase_jumps, expected):
        s = DCWaveform(amplitude=1)
        m = ModulatedWaveform(
            envelope=s,
            mod_freq=ModulationFrequency(0.2),
            channels=('I', 'Q')
        )

        phase_tracker = {
            ModulationFrequency(0.2): phase_jumps
        }

        wave = m(ts, phase_tracker=phase_tracker, phase_unit='degrees')

        assert_allclose(wave, expected)

        m_single_channel = m.evolve(channels=('I',))
        wave = m_single_channel(ts, phase_tracker=phase_tracker, phase_unit='degrees')
        assert_allclose(wave, expected[0])

        m_three_channel = m.evolve(channels=('a', 'b', 'c'))
        wave = m_three_channel(ts, phase_tracker=phase_tracker, phase_unit='degrees')
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

        m = ModulatedWaveform(
            envelope=DCWaveform(amplitude=1),
            mod_freq=mod_freq,
            channels=('I', 'Q')
        )

        wave_d = m(ts, phase_tracker={mod_freq: pj_deg}, phase_unit='degrees')
        wave_r = m(ts, phase_tracker={mod_freq: pj_rad}, phase_unit='radians')

        assert_allclose(wave_d, wave_r)