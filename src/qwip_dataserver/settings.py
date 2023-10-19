from functools import lru_cache
from pathlib import Path

from pydantic import ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    DATASERVER_ROOT: Path = Path("./")

    @field_validator("DATASERVER_ROOT")
    @classmethod
    def path_exists(cls, v: Path, info: ValidationInfo) -> Path:
        if not v.exists():
            raise ValueError(f"Path '{v}' does not exist!")

        if not v.is_dir():
            raise ValueError(f"'{v}' is not a directory.")

        return v.resolve()


@lru_cache
def get_settings():
    return Settings()
