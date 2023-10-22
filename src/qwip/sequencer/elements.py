from qwip.sequencer.timeline import Timeline
from qwip.utils import deprecated


class SequenceElement(Timeline):
    ...


SequenceElement.__init__ = deprecated(
    version="23.10.0",
    removed="24.1.0",
    message="Use `qwip.sequencer.Timeline` instead.",
)(SequenceElement.__init__)

__all__ = ["SequenceElement"]
