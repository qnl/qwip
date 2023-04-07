import re
from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

from qtrl.managers import MetaManager

from qwip.qpu.systems import ReadoutResonator
from qwip.sequencer.compilation import CompiledSequence
from qwip.settings.settings import qdefine

if TYPE_CHECKING:
    from qwip.qpu.qpu import QPU


class QuantumBackend(metaclass=ABCMeta):
    @abstractmethod
    def upload(self, cseq: CompiledSequence, **kwargs) -> None:
        ...

    @abstractmethod
    def acquire(self, cseq: CompiledSequence, **kwargs) -> dict:
        ...

    @abstractmethod
    def update_parameters(self, qpu: "QPU", **kwargs):
        ...


@qdefine
class SimulatorBackend(QuantumBackend):
    ...


@qdefine
class QTRLBackend(QuantumBackend):
    """A hardware backend that interface with QTRL."""

    meta: MetaManager

    def upload(self, cseq: CompiledSequence, **kwargs) -> None:
        self.meta.write_sequence(cseq)

    def acquire(self, cseq: CompiledSequence, repetitions: int = 512, **kwargs) -> dict:
        acquisition_kwargs = dict(n_reps=repetitions, save_data=False) | kwargs

        return self.meta.acquire(**acquisition_kwargs)

    def update_parameters(self, qpu: "QPU"):
        """Updates parameters from the QPU.

        The QTRL backend requires that all readout frequencies are specified in the
        variables config because this is where the ADCManager pulls the demod
        frequencies from.
        """
        for sys in qpu.subsystems.values():
            match sys:
                case ReadoutResonator(name=n, frequency=f):
                    if not (m := re.match(r"R(\d+)", n)):
                        raise ValueError(
                            f"Resonator names must be of the form 'R\\d+' for the "
                            f"QTRL backend. Got {n}"
                        )

                    self.meta.variables[f"Q{m[1]}/res_freq"] = f
