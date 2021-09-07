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

class SettingsBase(FlatMapping):
    __slots__ = tuple()