from typing import TYPE_CHECKING, Protocol, runtime_checkable

import attrs
import numpy as np
import pandas as pd
import pendulum
from attrs import field

import qwip
from qwip.attrs import qdefine, qfrozen
from qwip.backends.backend import QuantumBackend
from qwip.processing.processors import IQResult
from qwip.sequencer.compilation import QuantumExecutable

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

    def power(self) -> float:
        ...

    def averages(self) -> float:
        ...

    def averages_enabled(self) -> bool:
        ...

    def electrical_delay(self) -> float:
        ...

    def if_bandwidth(self) -> float:
        ...


@qdefine
class VNAExecutable(QuantumExecutable):
    """Executable for a generic VNA

    Specifies parameters for a VNA measurement. A value of `None` means that the current
    value for that parameter is used. A parameter alias can be specified in the field's
    metadata dictionary if the parameter name on the VNA differs from the attribute on
    the exe.

    Attributes:
        start: The start frequency.
        stop: The end frequency.
        points: The number of points in the sweep.
        power: The source power in dBm.
        averages: The number of averages to take. If > 1, averaging will automatically
            be enabled.
        delay: The electrical delay in seconds.
        if_bandwidth: The IF bandwidth.
        meas: The measurement type ('S11', 'S21', ...).
    """

    start: float | None = None
    stop: float | None = None
    points: int | None = None
    power: float | None = None
    averages: int | None = None
    delay: float | None = field(
        default=None, metadata=dict(parameter="electrical_delay")
    )
    if_bandwidth: float | None = None
    meas: str | None = field(default=None, metadata=dict(parameter="trace"))

    @property
    def center(self) -> float | None:
        if self.start is None or self.stop is None:
            return None

        return (self.start + self.stop) / 2

    @center.setter
    def center(self, center) -> None:
        if (span := self.span) is None:
            raise ValueError("Cannot set center when span is None.")

        self.start = center - span / 2
        self.stop = center + span / 2

    @property
    def span(self) -> float | None:
        if self.start is None or self.stop is None:
            return None

        return self.stop - self.start

    @span.setter
    def span(self, span) -> None:
        if (center := self.center) is None:
            raise ValueError("Cannot set span when center is None.")

        self.start = center - span / 2
        self.stop = center + span / 2

    def sweep(self, **kwargs):
        ...


def sweep_parameters(exe_list: list[VNAExecutable]) -> pd.Index:
    """Returns an index with the parameters that are swept.

    Any parameters that remain the same across all executables are not included in the
    returned index.

    Args:
        exe_list: A list of `VNAExecutable`

    Returns:
        A pandas multi-index.
    """
    params = pd.DataFrame([qwip.converter.unstructure(exe) for exe in exe_list])

    for col in params.columns:
        if params[col].nunique() == 1:
            del params[col]

    if params.empty:
        return pd.RangeIndex(0, len(exe_list))

    return pd.MultiIndex.from_frame(params)


@qdefine
class VNABackend(QuantumBackend):
    """A backend for interfacing with a Vector Network Analyzer.

    The main purpose of the VNABackend is to provide a consistent framework for
    processing results and saving data between frequency domain and time domain
    measurements. Since some measurements (resonator spectroscopy, punchout, etc.) can
    be done with both frequency and time domain instruments, this allows us to share the
    processing and analysis code.

    Attributes:
        vna: A VNA instrument for interfacing with a Vector Network Analyzer.
    """

    vna: VNA

    def upload(self, exe: VNAExecutable, **kwargs):
        """Sets the parameters for a frequency sweep.

        Any parameters that are `None` in the `VNAExecutable` will be left unchanged on
        the instrument, and the current value will be set in the exe.

        Args:
            exe: The `VNAExecutable` to pull the parameters from.
        """

        for f in attrs.fields(type(exe)):
            try:
                parameter = getattr(self.vna, f.metadata.get("parameter", f.name))
            except AttributeError:
                continue

            value = getattr(exe, f.name)

            if value is None:
                setattr(exe, f.name, parameter())

                # Letting averages <= 1 be equivalent to no averaging disabled
                if f.name == "averages" and not self.vna.averages_enabled():
                    setattr(exe, f.name, 1)
            else:
                parameter(value)

                if f.name == "averages":
                    self.vna.averages_enabled(value > 1)

    def acquire(self, **kwargs) -> dict:
        """Acquires the complex IQ data from the VNA.

        Args:
            **kwargs: Keyword arguments are passed to `get_complex_data`.

        Returns:
            A dictionary of `IQResult`.
        """
        timestamp = pendulum.now()

        frequencies = np.linspace(self.vna.start(), self.vna.stop(), self.vna.points())

        power_on = self.vna.power_on()
        self.vna.power_on(True)
        IQ = self.vna.get_complex_data(**kwargs)
        self.vna.power_on(power_on)

        df = pd.DataFrame(
            IQ, index=pd.Index(frequencies, name="frequency"), columns=["IQ"]
        )
        result = IQResult(name=self.vna.trace(), data=df, timestamp=timestamp)

        return {self.vna.name: result}

    def update_parameters(self, qpu: "QPU", **kwargs):
        ...

    @property
    def exe_formats(self) -> set[type[QuantumExecutable]]:
        return {VNAExecutable}
