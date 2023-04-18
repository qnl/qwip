import pytest
from numpy.random import default_rng

from qwip.qpu import FakeBackend
from qwip.sequencer import ReadoutMarker, Sequence, SequenceElement


class TestQuantumBackend:
    @pytest.fixture
    def backend(self, qpu_01, seed):
        backend = FakeBackend(rng=default_rng(seed))
        backend.update_parameters(qpu_01)

        return backend

    @pytest.fixture
    def sequencer(qpu_01):
        return qpu_01.sequencer

    @pytest.fixture
    def compiled(self, sequencer):
        def generate_cseq(shape):
            ro = SequenceElement()
            seq = Sequence.empty(shape)
            seq = seq + SequenceElement().add_waveform(ReadoutMarker(), 10e-9)

            return sequencer.compile(seq, readout=ro)

        return generate_cseq

    @pytest.fixture
    def sequence(self):
        def generate_seq(shape):
            seq = Sequence.empty(shape)
            seq = seq + SequenceElement().add_waveform(ReadoutMarker(), 10e-9)

            return seq

        return generate_seq

    def test_end_to_end(self, qpu_01, backend, compiled):
        # End to end tests require a database -- need to implement offline database
        # for proper testing.
        ...
