from qwip.config.database import (
    configschema,
    SettingsFolder,
    ValidatedSettingsFolder
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
    version: str = qwip.qsettings['version']
    qwip_commit: str
    sample_id: str

    hardware: HardwareSchema
    processing: ProcessingSchema
    model: QuantumModelSchema
    pulses: PulsesSchema
    compilation: CompilationSchema
