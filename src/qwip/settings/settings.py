import functools
import json

from collections.abc import Mapping
from contextlib import contextmanager
from typing import Optional

import attr, cattr

from attrs import define, frozen

import qwip
from qwip import yaml
from qwip.settings.base import SettingsBase
from qwip.flatdict import FlatDict
from qwip.settings.validation import add_type_validators
from qwip.settings.serialization import add_type_converters
from qwip.settings.schema import schema
from qwip._cattr import make_attrs_structure_fn, make_attrs_unstructure_fn

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

qfrozen = functools.partial(
    frozen,
    auto_attribs=True,
    kw_only=True,
    field_transformer=qwip_field_transform,
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

    def __iter__(self):
        yield from (k for k in iter(self.__slots__) if not k.startswith('_'))
        
    def __len__(self):
        return len(self.__slots__)

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
