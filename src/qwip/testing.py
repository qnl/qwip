"""Module for test helpers.
"""

from attrs import define
from collections.abc import Iterable

class UnorderedList(list):
    """A class for implementing unordered comparison of lists.
    
    This is necessary for testing serialization, since sets are unstructured as
    lists in qwip, and may be serialized in any order. With an UnorderedList, 
    we can then compare the resulting list or other ordered container, based only
    on the elements they contain.
    """
    def __init__(
        self,
        iterable,
        /,
        ignore_type=False
    ):
        self.original_type = type(iterable)
        self.ignore_type = ignore_type

        super().__init__(iterable)

    def __eq__(self, other: Iterable) -> bool:
        """Implements eq for UnorderedList.
        
        TODO: Make a shortcut comparison for hashable items. This can be done
        in O(n) by using `collections.Counter`. See this StackOverflow 
        [post](https://stackoverflow.com/a/8866661).

        Args:
            other: The iterable to compare to.

        Returns:
            `True` if self and other have the same elements.
        """
        return self.compare_unhashable(other)

    def __repr__(self) -> str:
        """Implements repr for UnorderedList."""
        return (
            f'Unordered{self.original_type.__name__.capitalize()}'
            f'({super().__repr__()})'
        )

    def compare_unhashable(self, other: Iterable):
        """Compares the iterable other to self, ignoring order.

        If `self.ignore_type is True`, then only the elements will be compared.
        
        Args:
            other: The iterable to compare to.
        """
        # First check container type
        if not self.ignore_type and type(other) != self.original_type:
            return False

        # Then check length if supported
        try:
            if len(other) != len(self):
                return False
        except TypeError:
            ...

        # Create a copy of self and try removing all elements in other
        unmatched = list(self)
        for element in other:
            try:
                unmatched.remove(element)
            except ValueError:
                return False

        return not unmatched

def ignore_order(
    iterable: Iterable,
    ignore_type: bool = False
) -> UnorderedList:
    """Returns an UnorderedList for making unordered comparisons.
    
    Args:
        iterable: The iterable to construct the UnorderedList from.
        ignore_type: Whether or not to ignore the container type when comparing.

    Returns:
        The UnorderedList object.
    """
    return UnorderedList(iterable, ignore_type=ignore_type)