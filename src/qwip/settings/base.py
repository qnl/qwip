from collections.abc import Mapping, KeysView, ValuesView, ItemsView

from copy import deepcopy
from contextlib import contextmanager

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
        key = key.strip(self._delim).split(self._delim)
        val = self
        for subkey in key:
            if isinstance(val, list):
                val = val[int(subkey)]
            else:
                val = getattr(val, subkey)
        return val

    def __iter__(self):
        for name in self.__slots__.__iter__():
            if not name.startswith('_'):
                yield name

    def __len__(self):
        return self.__slots__.__len__()

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

class SettingsBase(FlatMapping):
    __slots__ = tuple()