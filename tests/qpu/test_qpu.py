import pytest
from numpy.random import default_rng

from qwip.sequencer import Sequence, SequenceElement, ReadoutMarker
from qwip.qpu import FakeBackend

class TestQuantumBackend:
    @pytest.fixture
    def backend(self, qpu, seed):
        backend = FakeBackend(rng=default_rng(seed))
        backend.update_parameters(qpu)

        return backend
    
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

    def test_end_to_end(self, qpu, backend, compiled):
        # End to end tests require a database -- need to implement offline database
        # for proper testing. 
        ...
