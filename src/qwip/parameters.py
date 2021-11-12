"""A module for implementing a Parameters object."""

from collections.abc import Mapping, MutableMapping, KeysView, ValuesView, ItemsView
from copy import deepcopy
from contextlib import contextmanager
from typing import Generic, TypeVar

class FlatKeysView(KeysView):
    """A flattened key view."""
    def __iter__(self):
        yield from type(self._mapping).__flatiter__(self._mapping)
    
    def __repr__(self):
        return f'{self.__class__.__name__}({list(self.__iter__())}))'
        
class FlatValuesView(ValuesView):
    """A flattened values view."""
    def __iter__(self):
        for key in self._mapping.flatkeys():
            yield self._mapping[key]
    
    def __repr__(self):
        return f'{self.__class__.__name__}({list(self.__iter__())})'
        
class FlatItemsView(ItemsView):
    """A flattened items view."""
    def __iter__(self):
        for key in self._mapping.flatkeys():
            yield (key, self._mapping[key])

    def __repr__(self):
        return f'{self.__class__.__name__}({list(self.__iter__())})'


class FlatMapping(Mapping):
    __slots__ = tuple()
    _delim: str = '/'

    def __getitem__(self, key):
        key = self.split(self.strip(key))
        val = self
        for subkey in key:
            if isinstance(val, list):
                val = val[int(subkey)]
            else:
                try:
                    val = getattr(val, subkey)
                except AttributeError as e:
                    raise KeyError(str(e).rsplit(' ', maxsplit=1)[-1].strip('\'')) from AttributeError
        return val

    def __iter__(self):
        for name in self.__slots__.__iter__():
            if not name.startswith('_'):
                yield name

    def __len__(self):
        return self.__slots__.__len__()

    def __contains__(self, key):
        try:
            self[key]
        except (KeyError, AttributeError):
            return False
        else:
            return True

    def __flatiter__(self, base=None):
        def flat_enumerate(maybe_lst, index=()):
            if isinstance(maybe_lst, list):
                for i, nxt in enumerate(maybe_lst):
                    yield from flat_enumerate(nxt, index=(*index, i))
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

    def __repr__(self):
        return '{' + ', '.join([f'{repr(k)}: {repr(v)}' for k, v in self.items()]) + '}' 
    
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

    def _repr_html_(self):
        return (
            '<table style="min-width:200px">\n'
            '<thead><th>Key</th><th>Value</thead>'
            + '\n'.join([f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in self.flatitems()])
            + '</table>'
        )

    def get_keys(self, *args, keys=[]):
        subset = {}
        subset.update({key: self[key] for key in args})
        subset.update({key: self[key] for key in keys})

        return subset

    def strip(self, key):
        return key.strip(self._delim)

    def split(self, key):
        return key.split(self._delim)

    def rsplit(self, key, maxsplit=-1):
        return key.rsplit(self._delim, maxsplit)

KT = TypeVar('KT')
VT = TypeVar('VT')

class Parameters(FlatMapping, MutableMapping, dict, Generic[KT, VT]): # type:ignore
    """A dictionary object that supports key chaining and attribute access.
    """

    def  __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            new_value = self.__class__()
            new_value.update(value)
            value = new_value

        object.__setattr__(self, name, value)

    def __setitem__(self, key, val):
        def recursive_convert_list(list_val):
            for i, element in enumerate(list_val):
                if isinstance(element, dict):
                    list_val[i] = self.__class__(element)
                elif isinstance(element, list):
                    recursive_convert_list(element)

        keys = key.strip(self._delim).split(self._delim)

        if isinstance(val, Mapping) and not isinstance(val, FlatMapping):
            new_val = self.__class__()
            new_val.update(val)
            val = new_val
        elif isinstance(val, list):
            recursive_convert_list(val)

        if len(keys) == 1:
            setattr(self, keys[0], val)
            return

        subgroup = self
        remainder = keys
        for i, subkey in enumerate(keys[:-1]):
            base = remainder[0]
            remainder = keys[i+1:]

            if isinstance(subgroup, list):
                subgroup = subgroup[int(subkey)]
            elif isinstance(subgroup, FlatMapping):
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
            base = int(base) if isinstance(subgroup, list) else base
            if isinstance(subgroup, (list, FlatMapping)):
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
        subgroup.__setitem__(base, Parameters({key: val}))

    def __delitem__(self, key):
        key = key.split(self._delim) if isinstance(key, str) else [key]
        
        val = self
        for subkey in key[:-1]:
            val = getattr(val, subkey)
        
        delattr(val, key[-1])

    def __iter__(self):
        yield from self.__dict__.__iter__()

    def __len__(self):
        return self.__dict__.__len__()

    def todict(self):
        """Recursively converts the `Parameters` object to a dictionary.

        Returns:
            dict: The converted `Parameters`.
        """
        return {k: v.todict() if isinstance(v, Parameters) else v for k, v in self.items()}

    def toflatdict(self):
        """Converts the flattenned `Parameters` object to a dictionary.

        Returns:
            dict: The flattened `Parameters`.
        """
        return {k: v for k, v in self.flatitems()}
    
    @contextmanager
    def context(self, update=None):
        """Context manager for temporarily changing parameters."""
        orig = self.copy()
        try:
            if update is not None:
                self.update(update)
            yield
        finally:
            self.update(orig)