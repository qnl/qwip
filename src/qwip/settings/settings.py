import functools
import json

from collections.abc import Mapping
from contextlib import contextmanager
from typing import Optional

import attr, cattr


from attrs import define

from qwip import yaml
from qwip.settings.base import SettingsBase
from qwip.flatdict import FlatDict
from qwip.settings.validation import add_type_validators
from qwip.settings.serialization import add_type_converters
from qwip.settings.schema import schema

def qwip_field_transform(cls, fields):
    fields = add_type_converters(cls, fields)
    fields = add_type_validators(cls, fields)
    return fields

qdefine = functools.partial(
    define,
    auto_attribs=True,
    kw_only=True,
    field_transformer=qwip_field_transform,
    on_setattr=[attr.setters.convert, attr.setters.validate]
)

@contextmanager
def disable_validation():
    """Context manager for temporarily disabling validation on all attrs classes."""
    try:
        attr.set_run_validators(False)
        yield
    finally:
        attr.set_run_validators(True)

danger = disable_validation

@qdefine(auto_attribs=False) # pylint: disable=redundant-keyword-arg
class Settings(SettingsBase):
    """A validated dataclass object."""

    def __setitem__(self, key, val):
        keys = key.strip(self._delim).split(self._delim)

        if len(keys) == 1:
            setattr(self, keys[0], val)
        else:
            subgroup = self
            remainder = keys
            for i, subkey in enumerate(keys[:-1]):
                base = remainder[0]
                remainder = keys[i+1:]

                if subkey in subgroup:
                    subgroup = getattr(subgroup, subkey)
                else:
                    break
            else:
                base = remainder[0]

                if isinstance(subgroup, (Settings, FlatDict)):
                    subgroup.__setitem__(base, val)
                else:
                    setattr(subgroup, base, val)

                return

            key = self._delim.join(remainder)
            setattr(subgroup, base, FlatDict({key: val}))

    def update(self, *args, **kwargs):
        if len(args) > 1:
            raise TypeError(f'update expected at most 1 argument, got {len(args)}')
        
        def _update(key, value):
            if isinstance(value, Mapping):
                self[key].update(value)
            else:
                self[key] = value

        if args:
            other = args[0]
            
            if isinstance(other, Mapping):
                for key, value in other.items():
                    _update(key, value)
            else:
                for key, value in other:
                    _update(key, value)

        for key, value in kwargs.items():
            _update(key, value)

    def validate(self) -> None:
        """Calls all validators attached to field attributes."""
        attr.validate(self)
    
    @contextmanager
    def context(self, settings: Optional[dict] = None, validate: bool = True):
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
        """Returns a JSON schema in dictionary format for the `Settings` class.
        """
        return schema(cls)

    @classmethod
    def load(cls, file: str, file_fmt: str = 'yaml'):
        """Creates a `Settings` object from either a JSON or YAML file.

        file:
        """
        if file_fmt == 'yaml':
            with open(file, 'r') as f:
                s = cls(**yaml.load(f))
        elif file_fmt == 'json':
            with open(file, 'r') as f:
                s = cls(**json.load(f))

        return s

    def save(self, file, file_fmt='yaml'):
        pass