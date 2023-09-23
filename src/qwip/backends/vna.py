from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import pendulum

from qwip.attrs import qdefine
from qwip.backends.backend import QuantumBackend
from qwip.processing.processors import IQResult

if TYPE_CHECKING:
    from qwip.qpu.qpu import QPU


@runtime_checkable
class VNA(Protocol):
    def get_complex_data(self, run: bool = True) -> np.ndarray:
        ...

    def points(self) -> int:
        ...

    def start(self) -> float:
        ...

    def stop(self) -> float:
        ...

    def trace(self) -> str:
        ...


@qdefine
class VNABackend(QuantumBackend):
    vna: VNA

    def upload(self, exe, **kwargs):
        ...

    def acquire(self, exe: None = None, **kwargs) -> dict:
        timestamp = pendulum.now()

        frequencies = np.linspace(self.vna.start(), self.vna.stop(), self.vna.points())
        IQ = self.vna.get_complex_data(**kwargs)
        df = pd.DataFrame(IQ, index=pd.Index(frequencies, name="frequency"))

        result = IQResult(name=self.vna.trace(), data=df, timestamp=timestamp)

        return {self.vna.name: result}

    def update_parameters(self, qpu: "QPU", **kwargs):
        ...
