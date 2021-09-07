import functools
import attr, cattr

from collections.abc import Mapping
from copy import deepcopy
from contextlib import contextmanager

from attr import attrs

from qwip.settings.base import SettingsBase
from qwip.settings.parameters import (
    Parameters, FlatKeysView, FlatValuesView, FlatItemsView
)
from qwip.settings.validation import add_type_validators
from qwip.settings.serialization import add_type_converters


def qwip_field_transform(cls, fields):
    fields = add_type_converters(cls, fields)
    fields = add_type_validators(cls, fields)
    return fields

qwip_attrs = functools.partial(
    attrs,
    auto_attribs=True,
    kw_only=True,
    slots=True,
    field_transformer=qwip_field_transform,
    on_setattr=attr.setters.validate
)


@qwip_attrs(auto_attribs=False) # pylint: disable=redundant-keyword-arg
class Settings(SettingsBase):

    def __setitem__(self, key, val):
        keys = key.strip(self._delim).split(self._delim)

        if len(keys) == 1:
            setattr(self, keys[0], val)
        else:
            subgroup = self
            for i, subkey in enumerate(keys[:-1]):
                remainder = keys[i+1:]
                base = remainder[0]

                if isinstance(subgroup, list):
                    subgroup = subgroup[int(subkey)]
                else:
                    if subkey in subgroup:
                        subgroup = getattr(subgroup, subkey)
                    else:
                        break
            else:
                base = int(base) if isinstance(subgroup, list) else base

                if isinstance(subgroup, (list, Settings, Parameters)):
                    subgroup.__setitem__(base, val)
                else:
                    setattr(subgroup, base, val)

                return

            key = self._delim.join(remainder[1:])
            setattr(subgroup, base, Parameters({key: val}))
    
    def __iter__(self):
        for name in self.__slots__.__iter__():
            if not name.startswith('_'):
                yield name

    def __flatiter__(self, base=None):
        def flat_enumerate(maybe_lst, index=()):
            if isinstance(maybe_lst, list):
                for i, next in enumerate(maybe_lst):
                    yield from flat_enumerate(next, index=(*index, i))
            else:
                yield index, maybe_lst

        for k, v in self.items():
            if hasattr(v, '__flatiter__') and v:
                next_base = k if base is None else  f'{base}{self._delim}{k}'
                yield from v.__flatiter__(base=next_base)
            elif isinstance(v, list) and v:
                for idx, next_v in flat_enumerate(v):
                    listkey = self._delim.join(str(i) for i in idx)
                    if hasattr(next_v, '__flatiter__') and next_v:
                        next_base = f'{k}{self._delim}{listkey}'
                        if base:
                            next_base = base + listkey + next_base
                        yield from next_v.__flatiter__(base=next_base)
                    else:
                        yield f'{k}{self._delim}{listkey}' if base is None else f'{base}{self._delim}{k}{self._delim}{listkey}'
            else:
                yield k if base is None else f'{base}{self._delim}{k}' 
    
    def __len__(self):
        return self.__slots__.__len__()

    def _repr_html_(self):
        return (
            '<table style="min-width:200px">\n'
            '<thead><th>Key</th><th>Value</thead>'
            + '\n'.join([f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in self.flatitems()])
            + '</table>'
        )

    def flatkeys(self):
        """Returns a `View` of flattened keys.

        Nested `Parameters` are flattened and keys joined with a `'.'`.

        Returns:
            FlatKeysView: An iterator that returns all flattened keys in the
                `Parameters` object.
        """
        return FlatKeysView(self)
    
    def flatvalues(self):
        """Returns a `View` of flattened values.

        All `Parameters` objects contained within this object are iterated over

        Returns:
            FlatValuesView: An iterator that returns all values in the `Parameters`
                object, including values in nested `Parameters`.
        """
        return FlatValuesView(self)
    
    def flatitems(self):
        """Returns a `View` of flattened items.

        Nested `Parameters` are flattened and keys joined with a `'.'`.

        Returns:
            FlatItemsView: An iterator that returns a tuple of `(flatkey, val)`
                for all values in the `Parameters` object.
        """
        return FlatItemsView(self)

    def copy(self):
        """Returns a deep copy of the parameters object"""
        return deepcopy(self)

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
    
    @contextmanager
    def context(self, settings=None, validate=True):
        """Context manager for temporarily changing parameters.
        """
        orig = self.copy()
        try:
            if settings:
                self.update(settings)
            yield
        finally:
            self.update(orig)

    def todict(self):
        return cattr.unstructure(self)

    def toparameter(self):
        return Parameters(self.todict())

    @classmethod
    def fromdict(cls, dict):
        return cattr.structure(dict, cls)