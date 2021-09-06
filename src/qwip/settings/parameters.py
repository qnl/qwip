from collections.abc import MutableMapping, Collection, KeysView, ValuesView, ItemsView
from copy import deepcopy
from contextlib import contextmanager

class FlatKeysView(KeysView):
    def __iter__(self):
        yield from type(self._mapping).__flatiter__(self._mapping)
    
    def __repr__(self):
        return f'{self.__class__.__name__}({list(self.__iter__())}))'
        
class FlatValuesView(ValuesView):
    def __iter__(self):
        for key in self._mapping.flatkeys():
            yield self._mapping[key]
    
    def __repr__(self):
        return f'{self.__class__.__name__}({list(self.__iter__())})'
        
class FlatItemsView(ItemsView):
    def __iter__(self):
        for key in self._mapping.flatkeys():
            yield (key, self._mapping[key])

    def __repr__(self):
        return f'{self.__class__.__name__}({list(self.__iter__())})'

class Parameters(MutableMapping, dict): # type:ignore
    """
    A dictionary object that supports key chaining and attribute access.
    """
    _delim: str = '/'

    def  __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            new_value = self.__class__()
            new_value.update(value)
            value = new_value

        object.__setattr__(self, name, value)

    def __setitem__(self, key, val):
        key = key.strip(self._delim)
        if isinstance(val, dict):
            new_val = self.__class__()
            new_val.update(val)
            val = new_val
        
        if isinstance(key, str) and self._delim in key:
            base, subkey = key.split(self._delim, maxsplit=1)

            if base not in self:
                new_val = self.__class__()
                new_val.update({subkey:val})
                setattr(self, base, new_val)
            elif isinstance(self.__getitem__(base), Parameters):
                self.__getitem__(base).__setitem__(subkey, val)
            elif isinstance(self.__getitem__(base), list):
                subkey = subkey.split(self._delim, maxsplit=1)

                if len(subkey) > 1:
                    nextbase = self.__getitem__(base).__getitem__(int(subkey[0]))
                    nextbase.__setitem__(subkey[1], val)
                else:
                    self.__getitem__(base).__setitem__(int(subkey[0]), val)
            else:
                baseval = self.__getitem__(base)
                raise TypeError(
                    f'Cannot assign subkey {subkey} to base {base} of type {type(baseval)}.'
                )
        elif isinstance(val, list):
            new_val = []
            for e in val:
                new_val.append(self.__class__() if isinstance(e, dict) else e)
                if hasattr(new_val[-1], 'update'):
                    new_val[-1].update(e)
            val = new_val
            setattr(self, key, val)
        else:
            setattr(self, key, val)

    def __getitem__(self, key):
        key = key.strip(self._delim).split(self._delim)
        val = self
        for subkey in key:
            if isinstance(val, list):
                val = val[int(subkey)]
            else:
                val = getattr(val, subkey)
        return val

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

    def __flatiter__(self, base=None):
        for k, v in self.items():
            if hasattr(v, '__flatiter__') and v:
                next_base = k if base is None else  f'{base}{self._delim}{k}'
                yield from v.__flatiter__(base=next_base)
            elif isinstance(v, list) and v:
                for i, next_v in enumerate(v):
                    if hasattr(next_v, '__flatiter__') and next_v:
                        next_base = f'{k}{self._delim}{i}'
                        if base:
                            next_base = base + self._delim + next_base
                        yield from next_v.__flatiter__(base=next_base)
                    else:
                        yield f'{k}{self._delim}{i}' if base is None else f'{base}{self._delim}{k}{self._delim}{i}'
            else:
                yield k if base is None else f'{base}{self._delim}{k}' 

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

    def copy(self):
        """Returns a deep copy of the parameters object"""
        return deepcopy(self)
    
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