import json
from contextlib import contextmanager
from typing import Literal

import attr
import cattr
import tomli as tomllib

from qwip.attrs import qdefine
from qwip.flatdict import FlatDict, FlatMapping


@qdefine(auto_attribs=False)  # pylint: disable=redundant-keyword-arg
class Settings(FlatMapping):
    """A validated dataclass object."""

    def _get_mapping_type(self, key) -> type:
        field = getattr(attr.fields(type(self)), key, None)
        return field.type if field else FlatDict

    def __proxy_setitem__(self, key, val):
        if not hasattr(self, key):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            )

        setattr(self, key, val)

    def __proxy_getitem__(self, key):
        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            pass

        raise KeyError(f"'{key}'")

    def __proxy_delitem__(self, key):
        try:
            return object.__delattr__(self, key)
        except AttributeError:
            pass

        raise KeyError(f"'{key}'")

    def __iter__(self):
        yield from (k for k in iter(self.__slots__) if not k.startswith("_"))

    def __len__(self):
        return len(self.__slots__)

    def validate(self) -> None:
        """Calls all validators attached to field attributes."""
        attr.validate(self)

    @contextmanager
    def context(self, settings: dict | None = None, validate: bool = True):
        """Context manager for temporarily changing parameters.

        Args:
            settings (dict): A dictionary with a subset of keys to temporarily
                update the `Settings` object with.
            validate (bool): Whether or not to validate assignments within the
                context block.
        """
        orig = self.copy()
        try:
            if not validate:
                attr.set_run_validators(False)
            if settings:
                self.update(settings)
            yield
        finally:
            attr.set_run_validators(True)
            self.update(orig)

    def todict(self) -> dict:
        return cattr.unstructure(self)

    def toflatdict(self) -> FlatDict:
        return FlatDict(self.todict())

    def get_keys(self, *keys):
        return FlatDict(super().get_keys(*keys))

    @classmethod
    def fromdict(cls, d: dict):
        return cls(d)

    @classmethod
    def schema(cls) -> dict:
        """Returns a JSON schema in dictionary format for the `Settings` class."""
        return schema(cls)

    @classmethod
    def load(cls, file: str, file_fmt: Literal["json", "toml"] = "toml"):
        """Creates a `Settings` object from either a JSON or TOML file.

        Args:
            file: A path to the settings file.
            file_fmt: A string specifying the file format of the settings file.
        """
        match file_fmt:
            case "toml":
                with open(file, "rb") as f:
                    s = cls(**tomllib.load(f))
            case "json":
                with open(file, "r") as f:
                    s = cls(**json.load(f))
            case _:
                raise ValueError(f"Invalid file format: {file_fmt}.")

        return s

    def save(self, file, file_fmt="yaml"):
        pass
