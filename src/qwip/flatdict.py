"""A module for implementing a FlatDict object."""
import html

from collections.abc import Mapping, MutableMapping, KeysView, ValuesView, ItemsView
from copy import deepcopy
from contextlib import contextmanager
from typing import Any, Generic, TypeVar, Union, get_args, get_origin

import attr
import numpy as np
from cattr.gen import make_mapping_structure_fn, make_mapping_unstructure_fn

import qwip
from qwip.typing import is_annotated_type, is_optional_type, is_generic_type, issubtype

class FlatKeysView(KeysView):
    """A flattened key view."""
    __slots__ = ('_levels',)
    def __init__(self, mapping, levels=None):
        self._levels = levels
        super().__init__(mapping)

    def __iter__(self):
        flatkeys = type(self._mapping).__flatiter__(self._mapping, levels=self._levels)
        if self._levels is not None and self._levels < 0:
            keys = []
            n = -self._levels
            for k in flatkeys:
                split = k.rsplit(self._mapping._delim, maxsplit=n)
                shortened = split[0]
                if not keys:
                    keys.append(shortened)
                elif keys[-1].startswith(shortened):
                    keys[-1] = shortened
                elif not shortened.startswith(keys[-1]):
                    keys.append(shortened)

            for k in keys:
                yield k
        else:
            yield from flatkeys
    
    def __repr__(self):
        return f'{self.__class__.__name__}({list(self.__iter__())}))'
        
class FlatValuesView(ValuesView):
    """A flattened values view."""
    __slots__ = ('_levels',)
    def __init__(self, mapping, levels=None):
        self._levels = levels
        super().__init__(mapping)

    def __iter__(self):
        for key in self._mapping.flatkeys(levels=self._levels):
            yield self._mapping[key]
    
    def __repr__(self):
        return f'{self.__class__.__name__}({list(self.__iter__())})'
        
class FlatItemsView(ItemsView):
    """A flattened items view."""
    __slots__ = ('_levels',)
    def __init__(self, mapping, levels=None):
        self._levels = levels
        super().__init__(mapping)

    def __iter__(self):
        for key in self._mapping.flatkeys(levels=self._levels):
            yield (key, self._mapping[key])

    def __repr__(self):
        return f'{self.__class__.__name__}({list(self.__iter__())})'


class FlatMapping(Mapping):
    """A base class for a mapping object that allows "flat" access.

    Values can be accessed like `flatmap['one/two/three']` or
    `flatmap.one.two.three` in lieu of `flatmap['one']['two']['three']`.

    Attributes:
        _delim (str): A class variable that specifies the character used to
            separate nested mappings.
    """

    __slots__ = tuple()
    _delim: str = '/'

    def __getitem__(self, key):
        """Returns the item stored at `key`.
        
        Example:
            ```
            x.__getitem__(y) <==> x[y]
            x['one/two/three'] <==> x['one']['two']['three']
            ```
        """
        subkeys = self.split(self.strip(key))
        val = self
        for subkey in subkeys:
            # First we check if val has a defined non-nested get
            try:
                val = object.__getattribute__(val, '__proxy_getitem__')(subkey)
                continue
            except AttributeError as e:
                pass

            # If val is self it should always have a defined non-nested get function
            # so we raise a key error.
            if val is self:
                raise KeyError(f"'{key}'")

            # Otherwise val may be some other dictionary-like object so we check if
            # it has a __getitem__ defined.
            try:
                val = object.__getattribute__(val, '__getitem__')(subkey)
                continue
            except AttributeError as e:
                pass

            raise KeyError(f"'{key}'")

        return val

    def __proxy_getitem__(self, key):
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __contains__(self, key):
        try:
            self[key]
        except (KeyError, AttributeError):
            return False
        else:
            return True

    def __flatiter__(self, levels=None, base=None):
        if levels == 0:
            return
        
        next_levels = levels - 1 if levels is not None and levels > 0 else levels
        for k, v in self.items():
            if hasattr(v, '__flatiter__') and v and (levels is None or levels < 0 or levels > 1):
                next_base = k if base is None else  f'{base}{self._delim}{k}'
                
                yield from v.__flatiter__(base=next_base, levels=next_levels)
            else:
                yield k if base is None else f'{base}{self._delim}{k}'

    def __repr__(self):
        return '{' + ', '.join([f'{repr(k)}: {repr(v)}' for k, v in self.items()]) + '}' 
    
    def flatkeys(self, levels=None):
        """Returns a `View` of flattened keys.

        Nested `FlatMappings` are flattened and keys joined with a `'.'`.

        Returns:
            FlatKeysView: An iterator that returns all flattened keys in the
                `FlatMapping` object.
        """
        return FlatKeysView(self, levels=levels)
    
    def flatvalues(self, levels=None):
        """Returns a `View` of flattened values.

        All `FlatMapping` objects contained within this object are iterated over

        Returns:
            FlatValuesView: An iterator that returns all values in the 
                `FlatMapping` object, including values in nested `FlatMappings`.
        """
        return FlatValuesView(self, levels=levels)
    
    def flatitems(self, levels=None):
        """Returns a `View` of flattened items.

        Nested `FlatMappings` are flattened and keys joined with a `'.'`.

        Returns:
            FlatItemsView: An iterator that returns a tuple of `(flatkey, val)`
                for all values in the `FlatMapping` object.
        """
        return FlatItemsView(self, levels=levels)

    def copy(self):
        """Returns a deep copy of the `FlatMapping` object
        
        Returns:
            (FlatMapping): A copy of the `FlatMapping` 
        """
        return deepcopy(self)

    def _repr_html_(self):
        """Returns a formatted HTML table with the flattened keys and values.

        This allows function `display(flatmap)` to return a formatted view of
        the `FlatMapping`.
        """
        return (
                f'<table style="min-width:200px">\n'
                f'<thead><th>{type(self).__name__}</th></thead>'
                f'<thead><th>Key</th><th>Value</thead>'
                + '\n'.join([
                    f'<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>' 
                        for k, v in self.flatitems()
                ])
                + '</table>'
            )

    def get_keys(self, *keys) -> dict:
        """Returns a subset of the `FlatMapping` with the specified keys."""
        subset = {}
        subset.update({key: self[key] for key in keys})

        return subset

    def strip(self, key):
        return key.strip(self._delim)

    def split(self, key):
        return key.split(self._delim)

    def rsplit(self, key, maxsplit=-1):
        return key.rsplit(self._delim, maxsplit)

KT = TypeVar('KT', bound=str)
VT = TypeVar('VT')

class FlatDict(FlatMapping, MutableMapping, dict, Generic[KT, VT]): # type:ignore
    """A mapping object that supports key chaining and attribute access.

    Values can be accessed like `params['one/two/three']` or
    `params.one.two.three` in lieu of `params['one']['two']['three']`.

    Attributes:
        _delim (str): A class variable that specifies the character used to
            separate nested mappings.
    """
    __slots__ = tuple()

    def  __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __getattr__(self, key):
        try:
            return self.__getitem__(key)
        except KeyError as e:
            raise AttributeError(
                f"'{type(self).__name__} object has no attribute '{key}'"
            ) from e

    def __proxy_getitem__(self, key):
        return dict.__getitem__(self, key)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            new_value = self._get_mapping_type(name)()
            new_value.update(value)
            value = new_value

        if hasattr(self, name) and name not in self:
            object.__setattr__(self, name, value)
        else:
            self.__setitem__(name, value)

    def __proxy_setitem__(self, key, val):
        dict.__setitem__(self, key, val)

    def _get_mapping_type(self, key: str) -> type:
        return type(self)

    def __setitem__(self, key, val):
        keys = key.strip(self._delim).split(self._delim)

        if isinstance(val, Mapping) and not isinstance(val, FlatMapping):
            new_val = self.__class__()
            new_val.update(val)
            val = new_val

        if len(keys) == 1:
            self.__proxy_setitem__(keys[0], val)
            return

        subgroup = self
        remainder = keys
        for i, subkey in enumerate(keys[:-1]):
            base = remainder[0]
            remainder = keys[i+1:]

            if isinstance(subgroup, FlatMapping):
                if subkey in subgroup:
                    subgroup = getattr(subgroup, subkey)
                else:
                    break
            else:
                raise TypeError(
                    f'Cannot assign key {keys[i] + self._delim + remainder} to base {subgroup} of type {type(subgroup).__name__}.'
                )
        else: # Finished for loop
            base = remainder[0]
            if isinstance(subgroup, FlatMapping):
                subgroup.__setitem__(base, val)
            else:
                try:
                    setattr(subgroup, base, val)
                except AttributeError as e:
                    raise TypeError(
                        f'Cannot assign key {base} to base {repr(subgroup)} of type {type(subgroup).__name__}'
                    ) from e

            return
        
        # need to create parameters
        key = self._delim.join(remainder)
        subgroup.__setitem__(base, self._get_mapping_type(key=key)({key: val}))

    def __delitem__(self, key):
        key = key.split(self._delim) if isinstance(key, str) else [key]
        
        val = self
        for subkey in key[:-1]:
            val = val.__getitem__(subkey)
        
        dict.__delitem__(val, key[-1])

    def __iter__(self):
        yield from dict.__iter__(self)

    def __len__(self):
        return dict.__len__(self)

    def todict(self) -> dict:
        """Recursively converts the `FlatDict` object to a dictionary.

        Returns:
            dict: The converted `FlatDict`.
        """
        return {k: v.todict() if isinstance(v, FlatDict) else v for k, v in self.items()}

    def toflatdict(self, levels=None) -> dict:
        """Converts the flattenned `FlatDict` object to a dictionary.

        Returns:
            dict: The flattened `FlatDict`.
        """
        return {k: v for k, v in self.flatitems(levels=levels)}
    
    @contextmanager
    def context(self, update=None):
        """Context manager for temporarily changing values."""
        orig = self.copy()
        try:
            if update is not None:
                self.update(update)
            yield
        finally:
            self.update(orig)

# Register structuring/unstructuring on qwip converter
def make_flatdict_structure_fn(cls):
    structure_fn = make_mapping_structure_fn(
        cls, qwip.converter, structure_to=get_origin(cls) or cls
    )

    levels = None
    _cls = cls

    if args := get_args(_cls):
        VT = args[1]
        while is_optional_type(VT) or is_annotated_type(VT):
            VT = get_args(VT)[0]
            
        VT = get_origin(VT) or VT

        if VT is Any:
            levels = None
        elif issubclass(VT, Mapping) or attr.has(VT):
            levels = 1

    _cls = get_origin(_cls) or _cls

    def new_structure_fn(obj, cls):
        override = get_args(cls)[1] if is_annotated_type(cls) else levels

        if isinstance(obj, Mapping):
            obj = _cls(obj).toflatdict(levels=override)
        return structure_fn(obj, cls)

    return new_structure_fn

qwip.converter.register_structure_hook_factory(
    lambda cls: is_generic_type(cls, FlatDict),
    make_flatdict_structure_fn
)