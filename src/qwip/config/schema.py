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
class QuantumSystemSchema(ValidatedConfigFolder):
    system_class: str
    parameters: ConfigFolder[str, Any]

@configschema
class PulsesSchema(ValidatedConfigFolder):
    targets: tuple[Target]
    pulse_key: str
    variables: ConfigFolder[str, str | float]
    
@configschema
class NativeGateSchema(ValidatedConfigFolder):
    pulse_key: str
    parametrizable: bool = False
    dimensions: list[int] = field(factory=lambda: [2])
    unitary: Callable[..., np.ndarray]

@configschema
class ChannelInfoSchema(ValidatedConfigFolder):
    name: str
    index: int
    group: str
    subchannel: int = 0
    delay: float = 0

@configschema
class ChannelGroupSchema(ValidatedConfigFolder):
    name: str
    channels: list[str]
    sample_rate: float

@configschema
class CompilationSchema(ValidatedConfigFolder):
    channels: ConfigFolder[str, ChannelInfoSchema]
    channel_groups: ConfigFolder[str, ChannelGroupSchema]
    X90: ConfigFolder[Target, NativeGateSchema]
    EF_X90: ConfigFolder[Target, NativeGateSchema]
    Z: ConfigFolder[Target, NativeGateSchema]
    EF_Z: ConfigFolder[Target, NativeGateSchema]
    CZ: ConfigFolder[Target, NativeGateSchema]
    iSWAP: ConfigFolder[Target, NativeGateSchema]


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
    subsystems: ConfigFolder[Target, QuantumSystemSchema]
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