from typing import Any, Callable, TypeVar

import numpy as np
from attrs import field

import qwip
from qwip.config.database import (
    configschema,
    ConfigFolder,
    ValidatedConfigFolder
)

Target = TypeVar('Target', bound=str)

@configschema
class MixerSchema(ValidatedConfigFolder):
    channel_I: int
    channel_Q: int
    voltage_I: float
    voltage_Q: float

@configschema
class LocalOscillatorSchema(ValidatedConfigFolder):
    power: float = 0
    frequency: float = 0 
    phase: float = 0

@configschema
class DCSchema(ValidatedConfigFolder):
    current: float = 0
    range: float

@configschema
class HardwareSchema(ValidatedConfigFolder):
    local_oscillators: ConfigFolder[str, LocalOscillatorSchema]
    mixer_nulling: ConfigFolder[str, MixerSchema]
    current_sources: ConfigFolder[str, DCSchema]
    num_dac_channels: int

@configschema
class ClassificationSchema(ValidatedConfigFolder):
    n_states: int
    means: np.ndarray
    covariances: np.ndarray
    rotation: float = 0

@configschema
class ReadoutSchema(ValidatedConfigFolder):
    drives: ConfigFolder[Target, str]
    classification: ConfigFolder[Target, ClassificationSchema]
    excited_state_promotion: bool = False

    
@configschema
class QuantumModelSchema(ValidatedConfigFolder):
    model_class: str
    parameters: ConfigFolder[str, Any]

@configschema
class PulsesSchema(ValidatedConfigFolder):
    targets: tuple[Target]
    pulse_key: str
    variables: ConfigFolder[str, float]
    
@configschema
class NativeGate(ValidatedConfigFolder):
    pulse_key: str
    parametrizable: bool = False
    dimensions: list[int] = field(factory=lambda: [2])
    unitary: Callable[..., np.ndarray]

@configschema
class CompilationSchema(ValidatedConfigFolder):
    channel_map: ConfigFolder[str, int]
    channel_delays: ConfigFolder[str, float]
    X90: ConfigFolder[Target, NativeGate]
    EF_X90: ConfigFolder[Target, NativeGate]
    Z: ConfigFolder[Target, NativeGate]
    EF_Z: ConfigFolder[Target, NativeGate]
    CZ: ConfigFolder[Target, NativeGate]
    iSWAP: ConfigFolder[Target, NativeGate]


@configschema
class ConfigSchema(ValidatedConfigFolder):
    """Top level configuration settings.

    Attributes:
        version: The QWiP version/library schema version used.
        qwip_commit: The git commit (if available) for QWiP.
        sample_id: The sample id that the database corresponds to.
        hardware: Hardware subfolder
        processing: Processing subfolder.
        model: Model subfolder.
        pulses: Pulses subfolder.
        compilation: Compiation subfolder.
    """
    version: str = qwip.qsettings["version"]
    qwip_commit: str | None = qwip.qsettings["src/commit"]
    sample_id: str
    targets: tuple[Target, ...] = field(factory=tuple)

    hardware: HardwareSchema
    readout: ConfigFolder[str, ReadoutSchema]
    models: ConfigFolder[Target, QuantumModelSchema]
    pulses: ConfigFolder[str, PulsesSchema]
    compilation: CompilationSchema
    extra: ConfigFolder

__all__ = [
    "ConfigSchema",
    "CompilationSchema",
    "HardwareSchema",
    "ProcessingSchema",
    "PulsesSchema",
    "CompilationSchema"
]