from qwip.sequencer import Sequence

seq = Sequence.empty((20, 4, 5, 2))  # (1)
seq1 = seq[::2, 1:3, ...]
assert seq1.shape == (10, 2, 5, 2)
