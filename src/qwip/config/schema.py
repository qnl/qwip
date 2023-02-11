from qwip.config.database import (
    configschema,
    ConfigFolder,
    ValidatedConfigFolder
)

@configschema
class HardwareSchema:
    ...
    
@configschema
class ProcessingSchema:
    ...
    
@configschema
class QuantumModelSchema:
    ...

@configschema
class PulsesSchema:
    ...
    
@configschema
class CompilationSchema:
    ...

@configschema
class ConfigSchema:
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
    version: str = qwip.qsettings['version']
    qwip_commit: str
    sample_id: str

    hardware: HardwareSchema
    processing: ProcessingSchema
    model: QuantumModelSchema
    pulses: PulsesSchema
    compilation: CompilationSchema

__all__ = [
    "ConfigSchema",
    "CompilationSchema",
    "HardwareSchema",
    "ProcessingSchema",
    "PulsesSchema",
    "CompilationSchema"
]