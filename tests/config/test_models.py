import pytest
import sqlalchemy as sa

import qwip

from qwip.sequencer.elements import SequenceElement
from qwip.sequencer.waveform import (
    GaussianWaveform,
    DRAG,
    CWWaveform,
    VirtualZWaveform,
    CosineRampWaveform,
    ModulatedWaveform,
)
from qwip.config.models import (
    WaveformModel,
    WaveformLocationModel,
    ConstraintModel,
    SequenceElementModel,
)

from .fixtures import (
    configdb,
    models,
    session,
    reset_models
)

class TestWaveformModel:
    WAVEFORMS = dict(
        X90=ModulatedWaveform(
            name='X90',
            envelope=CosineRampWaveform(
                width=20e-9,
                ramp=2.5e-9,
                amplitude=0.15
            ),
            modulation=CWWaveform(
                channels=('I', 'Q'),
                frequency='mod_GE'
            )
        ),
        drag=DRAG(
            name='drag',
            lmbda=1,
            envelope=GaussianWaveform(
                width=20e-9
            )
        )
    )

    def test_select_none(self, session, reset_models):
        waveforms = session.scalars(sa.select(WaveformModel)).one_or_none()

        assert waveforms is None

    def test_insert_select(self, session, reset_models):
        models = []
        for wave in self.WAVEFORMS.values():
            model = WaveformModel.from_waveform(wave)
            session.add(model)
            models.append(model)
            
        session.flush()

        num_total = session.scalar(
            sa.select(sa.func.count())
            .select_from(WaveformModel)
        )

        assert num_total == 5

        num_toplevel = session.scalar(
            sa.select(sa.func.count())
            .select_from(WaveformModel)
            .where(WaveformModel.parent_id == None)
        )

        assert num_toplevel == 2

        results = session.scalars(
            sa.select(WaveformModel)
            .where(WaveformModel.parent_id == None)
            .order_by(WaveformModel.waveform_id)
        ).all()

        assert models == results
        
        results = session.scalars(
            sa.select(WaveformModel)
            .where(WaveformModel.parent_id != None)
            .order_by(WaveformModel.waveform_id)
        )

        assert [w.key for w in results] == ['envelope', 'modulation', 'envelope']

class TestSequenceElements:
    @pytest.fixture
    def x90_se(self):
        z_correction = VirtualZWaveform(mod_key='mod_GE', phase='z_phase')
        x90 = ModulatedWaveform(
            name='X90',
            envelope=CosineRampWaveform(
                width=20e-9,
                ramp=2.5e-9,
                amplitude=0.15
            ),
            modulation=CWWaveform(
                channels=('I', 'Q'),
                frequency='mod_GE'
            )
        )
        se = SequenceElement.fromtuples(
            [('t0', z_correction), ('t0', x90), ('t0' + x90.width, z_correction)],
            t0=0
        )
        return se

    def test_insert_select(self, session, reset_models, x90_se):
        se_model = SequenceElementModel.from_sequence_element(x90_se, name='x90')
        session.add(se_model)
        session.flush()

        num_waves = session.scalar(
            sa.select(sa.func.count())
            .select_from(WaveformModel)
            .where(WaveformModel.parent_id == None)
        )
        num_pairs = session.scalar(
            sa.select(sa.func.count())
            .select_from(WaveformLocationModel)
        )

        assert num_waves == 3
        assert num_pairs == 3

        new_model = session.scalars(sa.select(SequenceElementModel)).one()

        assert se_model == new_model

    def test_delete(self, session, reset_models, x90_se):
        se_model = SequenceElementModel.from_sequence_element(x90_se, name='x90')
        session.add(se_model)
        session.flush()

        session.delete(se_model)
        session.flush()

        num_waves = session.scalar(
            sa.select(sa.func.count())
            .select_from(WaveformModel)
            .where(WaveformModel.parent_id == None)
        )
        num_pairs = session.scalar(
            sa.select(sa.func.count())
            .select_from(WaveformLocationModel)
        )

        assert num_waves == 3
        assert num_pairs == 0
