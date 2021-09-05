from collections.abc import MutableMapping, KeysView, ValuesView, ItemsView

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
    A dictionary object that supports key chaining.
    """

    _delim: str = '/'

    def  __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = self.__class__(value)

        object.__setattr__(self, name, value)

    def __setitem__(self, key, val):
        if isinstance(val, dict):
            val = self.__class__(val)
        
        if isinstance(key, str) and self._delim in key:
            base, subkey = key.split(self._delim, maxsplit=1)

            if base not in self:
                setattr(self, base, self.__class__({subkey:val}))
            elif not isinstance(self.__getitem__(base), Parameters):
                baseval = self.__getitem__(base)
                raise TypeError(
                    f'Cannot assign subkey {subkey} to base {base} of type {type(baseval)}.'
                )
            else:
                self.__getitem__(base).__setitem__(subkey, val)
        elif isinstance(val, list):
            val = [self.__class__(e) if isinstance(e, dict) else e for e in val]
            setattr(self, key, val)
        else:
            setattr(self, key, val)

    def __getitem__(self, key):
        key = key.split(self._delim) if isinstance(key, str) else [key]
        val = self
        for subkey in key:
            val = getattr(val, subkey)
        return val

    def __delitem__(self, key):
        key = key.split(self._delim) if isinstance(key, str) else [key]
        
        val = self
        for subkey in key[:-1]:
            val = getattr(val, subkey)
        
        delattr(self, val, key[-1])

    def __iter__(self):
        yield from self.__dict__.__iter__()

    def __len__(self):
        return self.__dict__.__len__()

    def __flatiter__(self, base=None):
        for k, v in self.items():
            if isinstance(v, Parameters) and v:
                next_base = k if base is None else  f'{base}{self._delim}{k}'
                yield from v.__flatiter__(base=next_base)
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