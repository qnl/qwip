import pytest
import sqlalchemy as sa
import sympy as sym

import qwip
from qwip.config.models import (
    ConstraintModel,
    Folder,
    OperationLocationModel,
    OperationModel,
    Parameter,
    TimelineModel,
)
from qwip.sequencer.timeline import Timeline
from qwip.sequencer.waveform import (
    DRAG,
    CosineRampWaveform,
    CWWaveform,
    GaussianWaveform,
    ModulatedWaveform,
    TriggeredWaveform,
    VirtualZWaveform,
)


class TestFolder:
    def test_select_insert(self, session, models):
        hardware = Folder(name="hardware")
        lo = Folder(name="local_oscillators", parent=hardware)
        dc = Folder(name="dc_sources", parent=hardware)

        session.add(hardware)
        session.flush()

        results = session.scalars(sa.select(Folder)).all()

        assert len(results) == 3
        assert results == [hardware, lo, dc]

    def test_select_none(self, session, models):
        folders = session.scalars(sa.select(Folder)).one_or_none()
        assert folders is None

    def test_select_condition(self, session, models):
        hardware = Folder(name="hardware")
        lo = Folder(name="local_oscillators", parent=hardware)
        dc = Folder(name="dc_sources", parent=hardware)
        yoko = Folder(name="yokos", parent=dc)
        qubits = Folder(name="qubits")

        session.add_all([hardware, qubits])
        session.flush()

        assert session.scalars(sa.select(Folder)).all() == [
            hardware,
            qubits,
            lo,
            dc,
            yoko,
        ]

        assert session.scalars(
            sa.select(Folder).where(Folder.parent_id == None)
        ).all() == [hardware, qubits]
        assert session.scalars(
            sa.select(Folder).where(Folder.parent_id == hardware.folder_id)
        ).all() == [lo, dc]


class TestParameter:
    def test_select(self, session, models):
        params = session.scalars(sa.select(Parameter)).one_or_none()
        assert params is None


class TestOperationModel:
    WAVEFORMS = dict(
        X90=ModulatedWaveform(
            name="X90",
            envelope=CosineRampWaveform(width=20e-9, ramp=2.5e-9, amplitude=0.15),
            modulation=CWWaveform(channel="IQ", frequency="mod_GE"),
        ),
        drag=DRAG(name="drag", lmbda=1, envelope=GaussianWaveform(width=20e-9)),
        triggered=TriggeredWaveform(
            width="width",
            target=Timeline.from_layers(
                [
                    VirtualZWaveform(phase="zphase", frame="Q0"),
                    ModulatedWaveform(
                        envelope=GaussianWaveform(width="width", amplitude="amplitude"),
                        modulation=CWWaveform(
                            channel="Q0", frequency="frame", phase="phase"
                        ),
                    ),
                    VirtualZWaveform(phase="zphase", frame="Q0"),
                ]
            ),
        ),
    )

    def test_select_none(self, session, models):
        waveforms = session.scalars(sa.select(OperationModel)).one_or_none()

        assert waveforms is None

    def test_insert_select(self, session, models):
        models = []
        for wave in self.WAVEFORMS.values():
            model = OperationModel.from_operation(wave)
            session.add(model)
            models.append(model)

        session.flush()

        num_total = session.scalar(
            sa.select(sa.func.count()).select_from(OperationModel)
        )

        assert num_total == 6

        num_toplevel = session.scalar(
            sa.select(sa.func.count())
            .select_from(OperationModel)
            .where(OperationModel.parent_id == None)
        )

        assert num_toplevel == 3

        results = session.scalars(
            sa.select(OperationModel)
            .where(OperationModel.parent_id == None)
            .order_by(OperationModel.operation_id)
        ).all()

        assert models == results

        results = session.scalars(
            sa.select(OperationModel)
            .where(OperationModel.parent_id != None)
            .order_by(OperationModel.operation_id)
        )

        assert [w.key for w in results] == ["envelope", "modulation", "envelope"]

    def test_round_trip(self, session, models):
        models = []
        for wave in self.WAVEFORMS.values():
            model = OperationModel.from_operation(wave)
            session.add(model)
            models.append(model)

        session.flush()

        results = session.scalars(
            sa.select(OperationModel)
            .where(OperationModel.parent_id == None)
            .order_by(OperationModel.operation_id)
        ).all()

        reloaded = [w.to_operation() for w in results]
        original = list(self.WAVEFORMS.values())
        assert reloaded[:-1] == original[:-1]
        assert reloaded[-1].target == original[-1].target


class TestTimelines:
    @pytest.fixture
    def x90_tmln(self):
        z_correction = VirtualZWaveform(frame="mod_GE", phase="z_phase")
        x90 = ModulatedWaveform(
            name="X90",
            envelope=CosineRampWaveform(width=20e-9, ramp=2.5e-9, amplitude=0.15),
            modulation=CWWaveform(channel="IQ", frequency="mod_GE"),
        )
        tmln = Timeline.from_layers(
            [z_correction, x90, z_correction],
            t0="t0",
            width=x90.width,
            constraints={"t0"},
        )

        return tmln

    @pytest.fixture
    def tmln_no_width(self, x90_tmln):
        tmln = x90_tmln.copy()
        tmln.width = None
        return tmln

    def test_insert_select(self, session, models, x90_tmln):
        tmln_model = TimelineModel.from_timeline(x90_tmln, name="x90")
        session.add(tmln_model)
        session.flush()

        num_waves = session.scalar(
            sa.select(sa.func.count())
            .select_from(OperationModel)
            .where(OperationModel.parent_id == None)
        )
        num_pairs = session.scalar(
            sa.select(sa.func.count()).select_from(OperationLocationModel)
        )

        assert num_waves == 3
        assert num_pairs == 3

        new_model = session.scalars(sa.select(TimelineModel)).one()

        assert tmln_model == new_model

    def test_delete(self, session, models, x90_tmln):
        tmln_model = TimelineModel.from_timeline(x90_tmln, name="x90")
        extra_wave = OperationModel.from_operation(
            CosineRampWaveform(amplitude=0.5, width=20e-9)
        )
        session.add(extra_wave)
        session.add(tmln_model)
        session.flush()

        session.delete(tmln_model)
        session.flush()

        num_waves = session.scalar(
            sa.select(sa.func.count())
            .select_from(OperationModel)
            .where(OperationModel.parent_id == None)
        )
        num_pairs = session.scalar(
            sa.select(sa.func.count()).select_from(OperationLocationModel)
        )

        assert num_waves == 1
        assert num_pairs == 0

    def test_round_trip(self, session, models, tmln_no_width):
        tmln_model = TimelineModel.from_timeline(tmln_no_width, name="x90")

        session.add(tmln_model)
        session.flush()

        new_model = session.scalars(sa.select(TimelineModel)).one()
        new_tmln = new_model.to_timeline()

        assert new_tmln == tmln_no_width
