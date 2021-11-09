import functools
import attr, cattr

from collections.abc import Mapping
from copy import deepcopy
from contextlib import contextmanager

from attr import attrs

from qwip.settings.base import SettingsBase
from qwip.parameters import Parameters
from qwip.settings.validation import add_type_validators
from qwip.settings.serialization import add_type_converters


def qwip_field_transform(cls, fields):
    fields = add_type_converters(cls, fields)
    fields = add_type_validators(cls, fields)
    return fields

qattrs = functools.partial(
    attrs,
    auto_attribs=True,
    kw_only=True,
    slots=True,
    field_transformer=qwip_field_transform,
    on_setattr=attr.setters.validate
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

@qattrs(auto_attribs=False) # pylint: disable=redundant-keyword-arg
class Settings(SettingsBase):

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

                if isinstance(subgroup, list):
                    subgroup = subgroup[int(subkey)]
                else:
                    if subkey in subgroup:
                        subgroup = getattr(subgroup, subkey)
                    else:
                        break
            else:
                base = remainder[0]
                base = int(base) if isinstance(subgroup, list) else base

                if isinstance(subgroup, (list, Settings, Parameters)):
                    subgroup.__setitem__(base, val)
                else:
                    setattr(subgroup, base, val)

                return

            key = self._delim.join(remainder)
            setattr(subgroup, base, Parameters({key: val}))

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

    def validate(self):
        attr.validate(self)
    
    @contextmanager
    def context(self, settings=None, validate=True):
        """Context manager for temporarily changing parameters.
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

    def todict(self):
        return cattr.unstructure(self)

    def toparameter(self):
        return Parameters(self.todict())

    def get_keys(self, *args, keys=[]):
        return Parameters(super().get_keys(*args, keys=keys))

    @classmethod
    def fromdict(cls, d):
        return cls(d)