"""A module for implementing a Parameters object."""

from collections.abc import Mapping, MutableMapping, KeysView, ValuesView, ItemsView
from copy import deepcopy
from contextlib import contextmanager

from qwip.settings.base import FlatMapping, FlatKeysView, FlatValuesView, FlatItemsView

class Parameters(FlatMapping, MutableMapping, dict): # type:ignore
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
    def context(self, settings=None, validate=True):
        """Context manager for temporarily changing parameters."""
        orig = self.copy()
        try:
            if settings:
                self.update(settings)
            yield
        finally:
            self.update(orig)