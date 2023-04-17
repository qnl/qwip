from pathlib import Path

from attrs import field

from qwip.attrs import qdefine
from qwip.settings import Settings


@qdefine
class InstrumentSettings(Settings):
    ...


@qdefine
class DatabaseSettings(Settings):
    host: str
    username: str
    password: str = field(repr=lambda: "*****")
    dbname: str | None = None


@qdefine
class StaticSettings(Settings):
    hardware: str
    database: DatabaseSettings
    legacy_config_path: Path
