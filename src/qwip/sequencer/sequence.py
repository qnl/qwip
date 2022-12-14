from collections.abc import Sequence as TSequence
from typing_extensions import Self
import itertools as it

from numpy.typing import NDArray

import numpy as np

from qwip.settings.settings import qdefine
from qwip.sequencer.elements import SequenceElement

SEQUENCE_FUNCTIONS = {}

def sequence_implements(np_function):
    def decorator(func):
        SEQUENCE_FUNCTIONS[np_function] = func
        return func
    
    return decorator

@qdefine(init=False, slots=False, repr=False, eq=False, order=False)
class Sequence(np.ndarray):
    names: tuple[str | None, ...]
    labels: dict[str, np.ndarray]

    def __new__(
        cls,
        array: NDArray[SequenceElement],
        names: tuple[str, ...] | None = None,
        **labels
    ):
        # Turn array into ndarray and return view as Sequence
        # If array is already a subclass of ndarray, it will pass through
        # np.asanyarray unchanged.
        obj = np.asanyarray(array, dtype=object).view(cls)

        if names is None and not labels:
            return obj

        # Validate names
        if names is not None:
            names = _expand_names(names, obj.shape)

            if len(names) != len(obj.shape):
                raise ValueError(
                    f'Length of axis names {names} does not match shape {obj.shape}.'
                )

            unique_names = set()
            for n in names:
                if n in unique_names:
                    raise ValueError(f'{names} contains a duplicate name!')
                if n is not None:
                    unique_names.add(n)

            obj.names = names 

        if labels:
            obj.labels = dict()
            _set_labels(obj, labels, should_raise=True)

        obj.__array_finalize__()

        return obj

    def __array_finalize__(
        self,
        obj: NDArray[SequenceElement] | None = None,
        /
    ) -> None:
        # No additional cleanup necessary if this is explicit construction
        if obj is None: return

        if not hasattr(self, 'names'):
             # We copy names from obj if it exists and obj matches the correct shape
            if hasattr(obj, 'names') and self.shape == obj.shape:
                self.names = obj.names
            # otherwise set to default
            else: 
                self.names = (None,) * len(self.shape)

        if not hasattr(self, 'labels'):
            self.labels = dict()

            if hasattr(obj, 'labels'):
                _set_labels(self, obj.labels, should_raise=False)

    def __repr__(self) -> str:
        names = '' if all(n is None for n in self.names) else f'names={self.names}' + ', '
        prefix = '    '
        arr = prefix + np.array2string(self, prefix=prefix)
        return f'Sequence({names}shape={self.shape}\n{arr}\n)'

    def _expand_basic_index(self, index: tuple) -> tuple:
        """Expands out ellipses in numpy basic indices.

        This expands the index according to the shape of self. According to 
        the numpy [documentation](https://numpy.org/doc/stable/user/basics.indexing.html#dimensional-indexing-tools)
        ... will expand out to as many slice(None) as needed to match the
        shape of the ndarray.
        
        Args:
            index: The numpy index to expand that may include ...
        
        Returns:
            An equivalent index with every dimension expanded out.
        """

        if not isinstance(index, tuple):
            index = (index,) if index is ... else (index, ...) 
        elif ... not in index:
            return index

        # Number of slices to create
        n = len(self.shape) - len(tuple(i for i in index if i not in (..., np.newaxis)))
        
        # Must return a single element interable if not ellipse for itertools.chain
        replace_ellipsis = lambda i: (slice(None) for _ in range(n)) if i is ... else (i,)

        return tuple(it.chain(*(replace_ellipsis(i) for i in index)))

    def _get_names_from_index(self, index: tuple) -> tuple:
        """Determines new names for a view of self given the index.
        
        Args:
            index: The expanded index with no ellipses.
        
        Returns:
            A tuple of updated names
        """
        names = []

        dim = 0
        for idx in index:
            if isinstance(idx, slice):
                names.append(self.names[dim])
                dim += 1
            elif idx is None:
                names.append(None)
            else:
                dim += 1

        return tuple(names)

    
    def __getitem__(self, key):
        if _is_advanced_index(key):
            return NotImplemented

        obj = super().__getitem__(key)

        # If indexing leads to a single valued sequence element
        if not isinstance(obj, type(self)):
            return obj

        expanded = self._expand_basic_index(key)
        obj.names = self._get_names_from_index(expanded) # Set names

        # Copy over label views
        for n in self.labels:
            if n not in obj.names:
                continue

            obj.labels[n] = self.labels[n][expanded[obj.names.index(n)]]

        return obj

    def __array_ufunc__(
        self,
        ufunc,
        method,
        *inputs,
        out=None,
        **kwargs
    ):
        print('ufunc:', ufunc, 'method:', method, 'inputs:', *inputs, 'out:', out, 'kwargs:', kwargs)
        raise Exception
        outputs = out if out else (None,) * ufunc.nout

        results = getattr(ufunc, method)(*inputs, **kwargs)

        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            return
        
        if ufunc.nout == 1:
            results = (results,)

        results = tuple(
            (np.asarray(result).view(type(self)) if output is None else output)
            for result, output in zip(results, outputs)
        )

        return results[0] if len(results) == 1 else results

    def __array_function__(self, func, types, args, kwargs):
        if func not in SEQUENCE_FUNCTIONS:
            return NotImplemented

        if not all (issubclass(t, type(self)) for t in types):
            return NotImplemented

        return SEQUENCE_FUNCTIONS[func](*args, **kwargs)

    @classmethod
    def empty(
        cls,
        shape: tuple[int, ...], 
        names: tuple[str, ...] | None = None,
        **labels: np.ndarray
    ) -> Self:
        """Creates a Sequence of the specified shape with empty SequenceElements.
        
        Args:
            shape: The desired shape of the output sequence.
            names: Names to attach to the axis dimensions.
            labels: Labels to attach to the axis dimensions.
        """
        
        arr = np.empty(shape, dtype=object)

        for index in np.ndindex(*arr.shape):
            arr[index] = SequenceElement()

        return cls(arr, names, **labels)

    def transpose(self, *axes):
        """Reverses or permutes axis of the sequence.
    
        Args:
            axes: Specifies the permutation of the axes. If None, the axes are
                reversed. Can be a tuple of ints or n ints.

        Returns:
            Returns a view of the sequence with the permuted axes.
        """
        seq = super().transpose(*axes)

        if len(axes) == 1:
            axes = axes[0]

        if axes is None or not len(axes):
            seq.names = tuple(reversed(self.names))
            return seq

        seq.names = tuple(self.names[i] for i in axes)

        return seq

    @property
    def T(self):
        """Returns the transpose of a sequence.
        
        This is necessary to ensure that seq.T.names has the correct
        ordering of axis names.

        Returns:
            A transposed view of the of the sequence.
        """
        return self.transpose()


def _expand_names(names: tuple, shape: tuple[int]) -> tuple:
        """Expands out ellipses in names to match shape.

        This function expands out any tuples of names containing ... to match 
        the number of dimensions specified by shape. Replace ... with as many
        None values as necessary. The name tuple is also validated to ensure
        that the total length matches the number of dimensions (unless ...
        is included) and that no duplicate names exist (except for None).
        
        Args:
            names: The tuple of names passed to the constructor.
            shape: The shape of the array that the names will be attached to.
        
        Returns:
            An equivalent tuple of names with every dimension expanded out.

        Raises:
            IndexError: If names contains more than one ellipsis.
        """

        if ... not in names:
            return names

        if names.count(...) > 1:
            raise IndexError('names can only contain a single ellipsis (\'...\')')

        num_dims = len(shape)
        num_to_expand = num_dims - len(names) + 1

        def replace_ellipsis(n):
            return (None for _ in range(num_to_expand)) if n is ... else (n,)

        return tuple(it.chain(*(replace_ellipsis(n) for n in names)))

def _is_advanced_index(index: TSequence) -> True:
    """Determines if an index triggers numpy advanced indexing.
    
    See the numpy [documentation](https://numpy.org/doc/stable/user/basics.indexing.html)
    for more details on indexing.

    Args:
        index: An index passed to `ndarray.__getitem__`.
    
    Returns:
        True if the index would trigger advanced indexing under numpy rules.
    """

    # Check if index is a non-tuple sequence object

    int_or_bool = 'biu'

    def is_sequence_or_np_index(i):
        return (
            isinstance(i, TSequence) or
            (isinstance(i, np.ndarray) and i.dtype.kind in int_or_bool)
        )

    return (
        (is_sequence_or_np_index(index) and not isinstance(index, tuple)) or
        (isinstance(index, tuple) and any(is_sequence_or_np_index(i) for i in index))
    )

def _set_labels(
    obj: Sequence,
    labels: dict[str, TSequence],
    should_raise: bool = False
) -> Sequence:
    """Sets labels on a Sequence.

    This function os used in both the explicit constructor and 
    `__array_finalize__` to copy labels from a source dict to the Sequence
    object being created. Label arrays are copied by reference when the
    Sequence object is a view of another ndarray or Sequence. Labels are
    automatically converted to ndarrays.

    Args:
        obj: The Sequence object to attach the labels to.
        labels: The source dictionary from which labels should be copied.
        should_raise: Whether or not to raise an exception or silently pass.
    
    Returns:
        The Sequence object.
    """
    for n, arr in labels.items():
        try:
            idx = obj.names.index(n)
        
        except ValueError as e:
            if should_raise:
                raise ValueError(
                    f'\'{n}\' is not an axis name. names = {obj.names}'
                ) from e

            continue
            
        if len(arr) != obj.shape[idx]:
            if should_raise:
                raise ValueError(
                    f'{arr} has shape {arr.shape} which does not match shape '
                    f'{obj.shape} for dimension {idx}.'
                )

            continue

        obj.labels[n] = np.asarray(arr)

    return obj

@sequence_implements(np.array2string)
def array2string(a, **kwargs):
    return np.array2string(np.asarray(a), **kwargs)

@sequence_implements(np.concatenate)
def concatenate(
    sequences,
    axis=0,
    **kwargs
):  
    """Concatenates sequences along the specified axis.
    
    Args:
        sequences: An iterable of sequences to concatenate.
        axis: The axis along which to concatenate the sequences.

    Returns:
        The concatenated sequences.
    """
    arr_views = tuple(np.asarray(arr) for arr in sequences)
    seq = np.concatenate(arr_views, axis=axis, **kwargs).view(Sequence)

    arr_names = tuple(
        tuple(arr.names[dim] for arr in sequences) for dim in range(len(seq.shape))
    )

    if axis < 0:
        axis = len(seq.shape) + axis

    names = []
    for dim, axis_names in enumerate(arr_names):
        unique_names = set(n for n in axis_names if n is not None)
        if len(unique_names) == 0:
            names.append(None)
            continue
        elif len(unique_names) > 1:
            raise ValueError(
                f'All names along axis {dim} must be the same. {arr_names[dim]}'
            )
        
        name = next(iter(unique_names))  # Get the axis name
        names.append(name)

        labels = []

        for s in sequences:
            label = s.labels.get(name)

            # append labels for each array to labels if it exists
            if label is not None:
                labels.append(label)
            # if labels is None, break if concatenation axis or continue otherwise
            elif dim == axis:
                break
        else:
            if dim == axis:
                seq.labels[name] = np.concatenate(labels)
            elif labels:
                unique_values = np.unique(np.stack(labels), axis=0)
                # Assign labels only if all labels along non-concatenation axis are the same
                if unique_values.shape[0] == 1:
                    seq.labels[name] = labels[0]
    
    seq.names = tuple(names)

    return seq

@sequence_implements(np.stack)
def stack(
    sequences,
    axis=0,
    name: str | None = None,
    label: np.ndarray | None = None,
    **kwargs
):
    """Joins sequences along a new axis.
    
    Args:
        sequences: An iterable of sequences to concatenate.
        axis: Specifies the new axis in the stacked sequences.
        name: A name for the new axis.
        label: Labels for the new axis. Must match the number of sequences.

    Returns:
        The joined sequences.
    """
    arr_views = tuple(np.asarray(arr) for arr in sequences)
    seq = np.stack(arr_views, axis=axis, **kwargs).view(Sequence)

    if axis < 0:
        axis = len(seq.shape) + axis

    arr_names = tuple(
        tuple(arr.names[dim] for arr in sequences) for dim in range(len(seq.shape) - 1)
    )

    new_name = name
    new_label = label

    names = []
    for dim, axis_names in enumerate(arr_names):
        unique_names = set(n for n in axis_names if n is not None)
        if len(unique_names) == 0:
            names.append(None)
            continue
        elif len(unique_names) > 1:
            raise ValueError(
                f'All names along axis {dim} must be the same. {arr_names[dim]}'
            )
        
        name = next(iter(unique_names))  # Get the axis name
        names.append(name)

        labels = []

        for s in sequences:
            label = s.labels.get(name)

            # append labels for each array to labels if it exists
            if label is not None:
                labels.append(label)

        if labels:
            unique_values = np.unique(np.stack(labels), axis=0)
            # Assign labels only if all labels along non-concatenation axis are the same
            if unique_values.shape[0] == 1:
                seq.labels[name] = labels[0]

    if new_name in names and new_name is not None:
        raise ValueError(f'Axis name {name} is already in names. {names}')
    else:
        names.insert(axis, new_name)

    if new_label is not None:
        if new_name is None:
            raise ValueError(
                f'Cannot add a label for an axis with no name.'
            )

        if new_label.shape[0] != seq.shape[axis]:
            raise ValueError(
                f'Label has shape {new_label.shape} that is not compatible with '
                f'shape {seq.shape} on axis {axis}.'
            )

        seq.labels[new_name] = new_label

    seq.names = tuple(names)

    return seq

@sequence_implements(np.reshape)
def reshape(
    seq,
    shape,
    **kwargs
):
    """Reshapes the sequence.

    Reshaping a sequence creates a view of the sequence but does not
    copy over the axis names and labels in most cases, since this is
    not a sensible thing to do most of the time.
    
    Args:
        seq: The sequence to reshape.
        shape: The new shape of the sequence.
        **kwargs: Keyword arguments are passed to numpy reshape.

    Returns:
        A view of the sequence with the specified shape.
    """
    return seq.reshape(shape, **kwargs)

@sequence_implements(np.transpose)
def transpose(
    seq,
    axes=None,
):
    """Reverses or permutes axis of a sequence.
    
    Args:
        seq: The sequence to transpose.
        axes: Specifies the permutation of the axes. If None, the axes are
            reversed.

    Returns:
        The transposed sequence. A view is returned if possible.
    """
    return seq.transpose(axes)